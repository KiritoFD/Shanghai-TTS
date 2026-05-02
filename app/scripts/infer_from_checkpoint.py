from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from modelscope.hub.snapshot_download import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "你是中文查询规范化助手。"
    "给定用户输入后，只输出一个JSON对象，字段必须为："
    "core_text,type,predicate,object,keywords。"
    "type只能是“词项”或“动作短语”。"
    "keywords是中文词列表，不要输出任何解释。"
)


KNOWN_MODELSCOPE_IDS = {
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
    "Qwen3.5-0.8B-Base": "Qwen/Qwen3.5-0.8B-Base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer with a LoRA checkpoint and auto-download base model from ModelScope")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path like train/model_lora/checkpoint-237")
    parser.add_argument("--base_model_path", default=None, type=str, help="Optional local base model path. If set, skip auto resolve/download.")
    parser.add_argument("--query", action="append", default=None, help="Single query. Can repeat multiple times.")
    parser.add_argument("--input_jsonl", default=None, type=str, help='JSONL with field "query".')
    parser.add_argument("--output_jsonl", default=None, type=str, help="Prediction output path.")
    parser.add_argument("--modelscope_cache_dir", default=None, type=str, help="Optional ModelScope cache dir.")
    parser.add_argument("--modelscope_model_id", default=None, type=str, help="Force ModelScope model id.")
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Interactive mode. Default on when --query/--input_jsonl is not provided.",
    )
    return parser.parse_args()


def extract_json(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return "{}"


def infer_model_name_hint(checkpoint_path: Path, adapter_config: dict) -> str:
    base_path = str(adapter_config.get("base_model_name_or_path", "")).strip()
    candidates = [base_path, checkpoint_path.as_posix(), checkpoint_path.parent.as_posix()]
    for value in candidates:
        if "Qwen3.5-4B" in value:
            return "Qwen3.5-4B"
        if "Qwen3.5-2B" in value:
            return "Qwen3.5-2B"
        if "Qwen3.5-0.8B-Base" in value:
            return "Qwen3.5-0.8B-Base"
    match = re.search(r"Qwen3\.5-(?:4B|2B|0\.8B-Base)", base_path)
    if match:
        return match.group(0)
    return "Qwen3.5-4B"


def resolve_adapter_base_model_dir(checkpoint_path: Path, adapter_config: dict) -> Path | None:
    base_path = str(adapter_config.get("base_model_name_or_path", "")).strip()
    if not base_path:
        return None
    candidate = Path(base_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    resolved = (checkpoint_path / candidate).resolve()
    return resolved if resolved.exists() else None


def resolve_base_model_dir(args: argparse.Namespace, checkpoint_path: Path, adapter_config: dict) -> tuple[Path, str]:
    if args.base_model_path:
        local_model_dir = Path(args.base_model_path)
        if not local_model_dir.exists():
            raise FileNotFoundError(f"base_model_path not found: {local_model_dir}")
        return local_model_dir, "manual"

    adapter_local_dir = resolve_adapter_base_model_dir(checkpoint_path, adapter_config)
    if adapter_local_dir is not None:
        return adapter_local_dir, "adapter_config"

    script_dir = Path(__file__).resolve().parent

    if args.modelscope_model_id:
        model_id = args.modelscope_model_id
        local_name = model_id.split("/")[-1]
    else:
        hint = infer_model_name_hint(checkpoint_path, adapter_config)
        model_id = KNOWN_MODELSCOPE_IDS.get(hint, "Qwen/Qwen3.5-4B")
        local_name = hint

    # Default resolution: same directory as this inference script.
    # Example: train/infer_from_checkpoint.py -> prefer train/Qwen3.5-2B or train/Qwen3.5-4B
    local_model_dir = script_dir / local_name
    if local_model_dir.exists() and any(local_model_dir.glob("*.safetensors")):
        return local_model_dir, model_id

    # Secondary local fallback: parent of checkpoint tree (read-only check for compatibility).
    train_dir = checkpoint_path.parents[2] if len(checkpoint_path.parents) >= 3 else script_dir
    fallback_dir = train_dir / local_name
    if fallback_dir.exists() and any(fallback_dir.glob("*.safetensors")):
        return fallback_dir, model_id

    # Download target is always the same directory as this inference script.
    local_model_dir = script_dir / local_name

    snapshot_download(
        model_id=model_id,
        local_dir=str(local_model_dir),
        cache_dir=args.modelscope_cache_dir,
    )
    return local_model_dir, model_id


def collect_queries(args: argparse.Namespace) -> list[str]:
    queries: list[str] = []
    if args.query:
        for value in args.query:
            q = str(value).strip()
            if q:
                queries.append(q)
    if args.input_jsonl:
        for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = str(row.get("query", "")).strip()
            if q:
                queries.append(q)
    dedup: list[str] = []
    seen: set[str] = set()
    for q in queries:
        if q not in seen:
            seen.add(q)
            dedup.append(q)
    return dedup


def infer_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query: str,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"用户输入：{query}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return {
        "query": query,
        "raw_response": response,
        "parsed_text": extract_json(response),
    }


def scan_available_checkpoints(script_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    search_roots = [script_dir, script_dir / "model_lora"]
    for root in search_roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name.lower()
            if name.startswith("checkpoint-") and (child / "adapter_config.json").exists():
                candidates.append(child.resolve())
    candidates = sorted(set(candidates), key=lambda p: p.as_posix())
    return candidates


def choose_checkpoint_interactive(candidates: list[Path]) -> Path:
    print("Available checkpoints:")
    for idx, ckpt in enumerate(candidates, start=1):
        print(f"[{idx}] {ckpt}")
    while True:
        value = input("Select checkpoint number> ").strip()
        if not value.isdigit():
            print("Please input a number.")
            continue
        index = int(value)
        if 1 <= index <= len(candidates):
            return candidates[index - 1]
        print("Out of range, try again.")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    else:
        candidates = scan_available_checkpoints(script_dir)
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found under {script_dir} or {script_dir / 'model_lora'}"
            )
        checkpoint_path = choose_checkpoint_interactive(candidates)

    adapter_cfg_path = checkpoint_path / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found: {adapter_cfg_path}")
    adapter_config = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))

    base_model_dir, model_id = resolve_base_model_dir(args, checkpoint_path, adapter_config)
    print(f"base_model_dir={base_model_dir}")
    print(f"modelscope_model_id={model_id}")

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
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

    model = AutoModelForCausalLM.from_pretrained(str(base_model_dir), **model_kwargs)
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()
    print(f"model_load_s={time.perf_counter() - load_started:.2f}")

    queries = collect_queries(args)
    interactive = args.interactive if args.interactive is not None else (len(queries) == 0)
    if not interactive and len(queries) == 0:
        raise ValueError("No query provided. Use --query/--input_jsonl or enable interactive mode.")
    out_path = Path(args.output_jsonl) if args.output_jsonl else checkpoint_path / "inference_outputs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if interactive:
        print("Interactive mode started. Type query and press Enter. Type 'exit' or 'quit' to stop.")
        with out_path.open("a", encoding="utf-8") as handle:
            while True:
                try:
                    user_text = input("query> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nbye")
                    break
                if not user_text:
                    continue
                if user_text.lower() in {"exit", "quit"}:
                    print("bye")
                    break
                t0 = time.perf_counter()
                record = infer_one(
                    model=model,
                    tokenizer=tokenizer,
                    query=user_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(record["parsed_text"])
                print(f"infer_s={time.perf_counter() - t0:.2f}")
        print(f"wrote={out_path}")
        return

    total_started = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            t0 = time.perf_counter()
            record = infer_one(
                model=model,
                tokenizer=tokenizer,
                query=query,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"query: {query}")
            print(record["parsed_text"])
            print(f"infer_s={time.perf_counter() - t0:.2f}")
            print()

    print(f"wrote={out_path}")
    print(f"count={len(queries)}")
    print(f"total_infer_s={time.perf_counter() - total_started:.2f}")


if __name__ == "__main__":
    main()
