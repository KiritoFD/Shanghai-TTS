from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe max train micro-batch size for Qwen split LoRA training")
    parser.add_argument("--train_jsonl", required=True, type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output_json", default="", type=str)
    return parser.parse_args()


def load_texts(path: Path, limit: int) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def format_messages(tokenizer: AutoTokenizer, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)


def build_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        model_kwargs["torch_dtype"] = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
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
    model.train()
    return tokenizer, model


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe")

    max_batch = max(args.batch_sizes)
    rows = load_texts(Path(args.train_jsonl), max_batch)
    tokenizer, model = build_model(args)
    device = next(model.parameters()).device

    prompts = [format_messages(tokenizer, row["messages"]) for row in rows]
    results: list[dict] = []

    for batch_size in args.batch_sizes:
        batch_prompts = prompts[:batch_size]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            encoded = tokenizer(
                batch_prompts,
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            encoded["labels"] = encoded["input_ids"].clone()
            outputs = model(**encoded)
            loss = outputs.loss
            loss.backward()
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            result = {
                "batch_size": batch_size,
                "ok": True,
                "loss": round(float(loss.detach().cpu()), 4),
                "peak_reserved_gb": round(peak_reserved_gb, 3),
                "peak_allocated_gb": round(peak_allocated_gb, 3),
            }
            print(json.dumps(result, ensure_ascii=False))
            results.append(result)
            model.zero_grad(set_to_none=True)
            del outputs, loss, encoded
        except RuntimeError as exc:
            oom = "out of memory" in str(exc).lower()
            result = {
                "batch_size": batch_size,
                "ok": False,
                "oom": oom,
                "error": str(exc).splitlines()[0],
            }
            print(json.dumps(result, ensure_ascii=False))
            results.append(result)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if oom:
                break
        torch.cuda.empty_cache()

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
