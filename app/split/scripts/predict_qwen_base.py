from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from split_distill.rules import clean_prediction, parse_teacher_json  # noqa: E402


SYSTEM_PROMPT = (
    "You normalize Chinese query text into one JSON object with keys "
    "core_text, type, predicate, object, keywords. "
    "type must be one of ['词项','动作短语']. "
    "Return JSON only."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base Qwen model on split task inputs")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B", type=str)
    parser.add_argument("--adapter_path", default=None, type=str)
    parser.add_argument("--max_new_tokens", default=160, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--limit", default=None, type=int)
    return parser.parse_args()


def read_rows(path: str, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            rows.append(json.loads(line))
    return rows


def build_prompt(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "User query: "
                f"{query}\n"
                "Return a JSON object with fields core_text, type, predicate, object, keywords."
            ),
        },
    ]


def render_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return "".join(f"{msg['role']}: {msg['content']}\n" for msg in messages) + "assistant: "


def load_model(args: argparse.Namespace):
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer: AutoTokenizer, query: str, args: argparse.Namespace) -> tuple[dict, str]:
    prompt = render_prompt(tokenizer, build_prompt(query))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["top_p"] = args.top_p
        outputs = model.generate(**inputs, **generate_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    generated = outputs[0][prompt_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    payload = parse_teacher_json(text)
    return payload, text


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input, args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args)

    ok_count = 0
    error_count = 0
    with output_path.open("w", encoding="utf-8") as dst:
        for row in tqdm(rows, desc="predict"):
            query = str(row["query"])
            record = {"query": query}
            raw_text = None
            try:
                payload, raw_text = generate_one(model, tokenizer, query, args)
                cleaned, warnings = clean_prediction(query, payload)
                if cleaned is None:
                    raise ValueError(",".join(warnings) or "invalid_prediction")
                record.update(cleaned.to_dict())
                if warnings:
                    record["warnings"] = warnings
                record["raw_output"] = raw_text
                ok_count += 1
            except Exception as exc:  # noqa: BLE001
                record["error"] = str(exc)
                if raw_text is not None:
                    record["raw_output"] = raw_text
                error_count += 1
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input": args.input,
        "output": args.output,
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "count": len(rows),
        "ok": ok_count,
        "errors": error_count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
