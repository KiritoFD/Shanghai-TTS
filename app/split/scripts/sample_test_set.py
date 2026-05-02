from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a frozen 200-example test set from cleaned teacher data")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--size", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    random.seed(args.seed)

    hard_cases = [row for row in rows if row.get("source") == "hard_case"]
    word_rows = [row for row in rows if row.get("type") == "词项" and row.get("source") != "hard_case"]
    phrase_rows = [row for row in rows if row.get("type") == "动作短语" and row.get("source") != "hard_case"]

    random.shuffle(hard_cases)
    random.shuffle(word_rows)
    random.shuffle(phrase_rows)

    selected: list[dict] = []
    seen_queries: set[str] = set()

    def add_rows(candidates: list[dict], limit: int) -> None:
        for row in candidates:
            if len(selected) >= args.size or limit <= 0:
                return
            query = row["query"]
            if query in seen_queries:
                continue
            selected.append(row)
            seen_queries.add(query)
            limit -= 1

    hard_target = min(40, len(hard_cases))
    add_rows(hard_cases, hard_target)

    remaining = args.size - len(selected)
    word_target = remaining // 2
    phrase_target = remaining - word_target
    add_rows(word_rows, word_target)
    add_rows(phrase_rows, phrase_target)

    if len(selected) < args.size:
        leftovers = hard_cases + word_rows + phrase_rows
        random.shuffle(leftovers)
        for row in leftovers:
            if len(selected) >= args.size:
                break
            query = row["query"]
            if query in seen_queries:
                continue
            selected.append(row)
            seen_queries.add(query)

    if len(selected) < args.size:
        raise RuntimeError(f"not_enough_rows_for_test_set: wanted={args.size} got={len(selected)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected[: args.size]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
