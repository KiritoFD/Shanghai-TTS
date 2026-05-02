from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Distill top-match ID selector")
    parser.add_argument("--train_jsonl", default=str(base / "data" / "processed" / "match_train_sft.jsonl"), type=str)
    parser.add_argument(
        "--model_name_or_path",
        default=str((Path(__file__).resolve().parent.parent / "Qwen3.5-2B").resolve()),
        type=str,
    )
    parser.add_argument("--output_dir", default=str(base / "model_lora"), type=str)
    parser.add_argument("--max_length", default=768, type=int)
    parser.add_argument("--per_device_train_batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=32, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=float)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset("json", data_files={"train": args.train_jsonl})["train"]
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def preprocess(row: dict) -> dict:
        text = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        encoded = tok(text, truncation=True, max_length=args.max_length, padding="max_length")
        encoded["labels"] = encoded["input_ids"][:]
        return encoded

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    quant = None
    if args.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    tr_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        fp16=not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
    )

    trainer_kwargs = {
        "model": model,
        "args": tr_args,
        "train_dataset": ds,
        "data_collator": DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
    }
    sign = inspect.signature(Trainer.__init__)
    if "tokenizer" in sign.parameters:
        trainer_kwargs["tokenizer"] = tok
    elif "processing_class" in sign.parameters:
        trainer_kwargs["processing_class"] = tok
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    Path(args.output_dir, "train_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
