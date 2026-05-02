from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import plain text queries into JSONL")
    parser.add_argument("--input_txt", required=True, type=str)
    parser.add_argument("--output_jsonl", required=True, type=str)
    parser.add_argument("--source", default="plain_txt", type=str)
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--min_chars", default=1, type=int)
    parser.add_argument("--max_chars", default=48, type=int)
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_txt)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    rows: list[dict] = []

    for line in input_path.read_text(encoding="utf-8").splitlines():
        query = normalize(line)
        if not query:
            continue
        if len(query) < args.min_chars or len(query) > args.max_chars:
            continue
        if args.dedup and query in seen:
            continue
        seen.add(query)
        rows.append({"query": query, "source": args.source})

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
