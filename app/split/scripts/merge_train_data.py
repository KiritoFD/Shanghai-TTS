from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge new cleaned data into existing train set by query")
    parser.add_argument("--base_train", required=True, type=str)
    parser.add_argument("--new_clean", required=True, type=str)
    parser.add_argument("--out_train", required=True, type=str)
    parser.add_argument("--stats_out", required=True, type=str)
    parser.add_argument("--prefer_new", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    base_rows = read_jsonl(args.base_train)
    new_rows = read_jsonl(args.new_clean)

    merged: dict[str, dict] = {}
    for row in base_rows:
        query = str(row.get("query", "")).strip()
        if query:
            merged[query] = row

    inserted = 0
    replaced = 0
    for row in new_rows:
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        if query in merged:
            if args.prefer_new:
                merged[query] = row
                replaced += 1
        else:
            merged[query] = row
            inserted += 1

    out_rows = list(merged.values())
    write_jsonl(args.out_train, out_rows)
    stats = {
        "base_rows": len(base_rows),
        "new_rows": len(new_rows),
        "merged_rows": len(out_rows),
        "inserted": inserted,
        "replaced": replaced,
    }
    Path(args.stats_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
