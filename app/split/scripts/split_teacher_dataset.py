from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split cleaned teacher dataset into train/dev/test")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--train_out", required=True, type=str)
    parser.add_argument("--dev_out", required=True, type=str)
    parser.add_argument("--test_out", required=True, type=str)
    parser.add_argument("--train_ratio", default=0.9, type=float)
    parser.add_argument("--dev_ratio", default=0.05, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def write_jsonl(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    random.seed(args.seed)
    random.shuffle(rows)

    n = len(rows)
    train_end = int(n * args.train_ratio)
    dev_end = train_end + int(n * args.dev_ratio)

    write_jsonl(args.train_out, rows[:train_end])
    write_jsonl(args.dev_out, rows[train_end:dev_end])
    write_jsonl(args.test_out, rows[dev_end:])


if __name__ == "__main__":
    main()
