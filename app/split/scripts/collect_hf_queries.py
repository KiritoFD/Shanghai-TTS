from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect query-like text from a Hugging Face dataset")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--subset", default=None, type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--field", required=True, type=str, help="Field containing the query text")
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--limit", default=5000, type=int)
    parser.add_argument("--min_chars", default=1, type=int)
    parser.add_argument("--max_chars", default=24, type=int)
    parser.add_argument("--only_chinese_like", action="store_true")
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def is_chinese_like(text: str) -> bool:
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    return len(cjk) >= max(1, len(text) // 2)


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    rows: list[dict] = []

    for row in ds:
        if args.field not in row:
            continue
        query = normalize(str(row[args.field]))
        if not query:
            continue
        if len(query) < args.min_chars or len(query) > args.max_chars:
            continue
        if args.only_chinese_like and not is_chinese_like(query):
            continue
        if query in seen:
            continue
        seen.add(query)
        rows.append({"query": query, "source": f"hf:{args.dataset}:{args.split}"})
        if len(rows) >= args.limit:
            break

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
