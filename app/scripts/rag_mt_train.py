from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen RAG-MT LoRA.")
    parser.add_argument("--train_jsonl", default="app/rag_mt_data/train.jsonl", type=str)
    parser.add_argument("--eval_jsonl", default="app/rag_mt_data/eval.jsonl", type=str)
    parser.add_argument("--model_name_or_path", default="model/Qwen3.5-0.8B-Base", type=str)
    parser.add_argument("--output_dir", default="app/rag_mt_lora/qwen35_0p8b_basic", type=str)
    parser.add_argument("--max_length", default=384, type=int)
    parser.add_argument("--max_steps", default=80, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--grad_accumulation", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


class JsonlSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        text = self.format_messages(row["messages"])
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"][:]
        return {key: torch.tensor(value) for key, value in encoded.items()}

    def format_messages(self, messages: list[dict]) -> str:
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = JsonlSFTDataset(Path(args.train_jsonl), tokenizer, max_length=args.max_length)
    tokenized_eval = None
    if Path(args.eval_jsonl).exists():
        tokenized_eval = JsonlSFTDataset(Path(args.eval_jsonl), tokenizer, max_length=args.max_length)

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        quantization_config=quantization_config,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accumulation,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "logging_steps": 5,
        "save_steps": max(args.max_steps, 1),
        "save_total_limit": 1,
        "report_to": [],
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "seed": args.seed,
    }
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if tokenized_eval is not None:
        if "eval_strategy" in ta_params:
            training_kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in ta_params:
            training_kwargs["evaluation_strategy"] = "steps"
        training_kwargs["eval_steps"] = max(args.max_steps, 1)

    training_args = TrainingArguments(**training_kwargs)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_eval,
        "data_collator": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
