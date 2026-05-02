from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call teacher model via NVIDIA API")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--system_prompt", default="split/prompts/teacher_batch_system_prompt.txt", type=str)
    parser.add_argument("--api_base", default="https://integrate.api.nvidia.com/v1", type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_tokens", default=2200, type=int)
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--sleep_s", default=0.1, type=float)
    parser.add_argument("--timeout_s", default=300, type=int)
    parser.add_argument("--max_retries", default=3, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_text_from_response(data: dict) -> str | None:
    message = data["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip().startswith(("[", "{")):
        return reasoning_content
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip().startswith(("[", "{")):
        return reasoning
    return None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    api_key = config.get("api_key") or os.environ.get("NVAPI_KEY", "")
    if not api_key:
        raise RuntimeError("NVAPI_KEY is not set")

    model = args.model or config.get("model") or os.environ.get("SPLIT_TEACHER_MODEL", "moonshotai/kimi-k2.5")
    fallback_model = config.get("fallback_model")
    api_base = config.get("api_base", args.api_base)
    temperature = config.get("temperature", args.temperature)
    max_tokens = config.get("max_tokens", args.max_tokens)
    batch_size = int(config.get("batch_size", args.batch_size))
    sleep_s = float(config.get("sleep_s", args.sleep_s))
    timeout_s = int(config.get("timeout_s", args.timeout_s))
    max_retries = int(config.get("max_retries", args.max_retries))
    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")

    input_rows = read_jsonl(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_queries: set[str] = set()
    if args.resume and output_path.exists():
        for row in read_jsonl(str(output_path)):
            if isinstance(row.get("batch_queries"), list):
                for query in row["batch_queries"]:
                    value = str(query).strip()
                    if value:
                        completed_queries.add(value)
            else:
                value = str(row.get("query", "")).strip()
                if value:
                    completed_queries.add(value)

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    )

    pending_rows = [row for row in input_rows if str(row["query"]) not in completed_queries]

    with output_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for batch in chunked(pending_rows, batch_size):
            user_text = "请逐条标注以下查询，并按原顺序返回 JSON 数组：\n" + "\n".join(
                f"{index + 1}. {row['query']}" for index, row in enumerate(batch)
            )
            def invoke(target_model: str) -> tuple[dict, str | None]:
                payload = {
                    "model": target_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                data = None
                for attempt in range(max_retries):
                    try:
                        response = session.post(f"{api_base}/chat/completions", json=payload, timeout=timeout_s)
                        response.raise_for_status()
                        data = response.json()
                        return data, extract_text_from_response(data)
                    except Exception:
                        if attempt + 1 >= max_retries:
                            raise
                        time.sleep(min(8.0, 1.5 * (attempt + 1)))
                raise RuntimeError("teacher_request_failed")

            data, text = invoke(model)
            used_model = model
            if (text is None or not text.strip()) and fallback_model:
                data, text = invoke(fallback_model)
                used_model = fallback_model
            if text is None:
                raise RuntimeError("teacher_response_has_no_text")
            output_row = {
                "batch_queries": [str(row["query"]) for row in batch],
                "batch_sources": [row.get("source", "") for row in batch],
                "teacher_model": used_model,
                "raw_response": text,
            }
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            handle.flush()
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
