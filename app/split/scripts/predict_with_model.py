from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "你是中文查询规范化助手。"
    "给定用户输入后，只输出一个JSON对象，字段必须为："
    "core_text,type,predicate,object,keywords。"
    "type只能是“词项”或“动作短语”。"
    "keywords是中文词列表，不要输出任何解释。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction with base model + LoRA adapter")
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--adapter_path", default=None, type=str)
    parser.add_argument("--input_jsonl", required=True, type=str)
    parser.add_argument("--output_jsonl", required=True, type=str)
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_output", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def extract_json(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return "{}"


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    rows = [json.loads(line) for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_queries: set[str] = set()
    if args.resume_output and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            query = str(record.get("query", "")).strip()
            if query:
                done_queries.add(query)

    pending_rows = []
    for row in rows:
        query = str(row.get("query", "")).strip()
        if query and query not in done_queries:
            pending_rows.append(row)

    if not pending_rows:
        print("All rows already predicted. Nothing to do.")
        return

    open_mode = "a" if args.resume_output and out_path.exists() else "w"
    with out_path.open(open_mode, encoding="utf-8") as handle:
        for row in pending_rows:
            query = str(row.get("query", "")).strip()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"用户输入：{query}"},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=args.use_cache,
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            parsed = extract_json(response)
            record = {"query": query, "raw_response": response, "parsed_text": parsed}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
