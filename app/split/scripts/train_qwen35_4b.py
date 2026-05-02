from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-4B on structured split task")
    parser.add_argument("--train_jsonl", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-4B", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--grad_accumulation", default=16, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser.parse_args()


def format_example(messages: list[dict]) -> str:
    return "".join(f"<|{msg['role']}|>\n{msg['content']}\n" for msg in messages)


def main() -> None:
    args = parse_args()
    dataset = load_dataset("json", data_files=args.train_jsonl)["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(row: dict) -> dict:
        text = format_example(row["messages"])
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"][:]
        return encoded

    tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with Path(args.output_dir, "train_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
