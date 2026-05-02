from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exclude queries found in another JSONL file")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--exclude", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    input_rows = read_jsonl(args.input)
    exclude_queries = {row["query"] for row in read_jsonl(args.exclude)}
    output_rows = [row for row in input_rows if row["query"] not in exclude_queries]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
