from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize teacher JSONL files by filling missing batch_index.")
    parser.add_argument("--root", default="split/data/raw", type=str, help="Directory to scan for JSONL files.")
    parser.add_argument("--write", action="store_true", help="Write changes in-place.")
    return parser.parse_args()


def should_fix(row: dict) -> bool:
    return isinstance(row, dict) and "raw_response" in row and "batch_queries" in row


def fix_file(path: Path, write: bool) -> tuple[int, int, int]:
    rows: list[dict] = []
    fixed = 0
    bad = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                bad += 1
                continue
            if should_fix(row) and "batch_index" not in row:
                row["batch_index"] = len(rows)
                fixed += 1
            rows.append(row)

    if write and fixed > 0:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(rows), fixed, bad


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    files = sorted(root.rglob("*.jsonl"))
    touched = 0
    for path in files:
        total, fixed, bad = fix_file(path, write=args.write)
        if fixed > 0 or bad > 0:
            touched += 1
            print(f"{path}\ttotal={total}\tfixed={fixed}\tbad={bad}")
    print(f"scanned={len(files)} touched={touched} write={args.write}")


if __name__ == "__main__":
    main()
