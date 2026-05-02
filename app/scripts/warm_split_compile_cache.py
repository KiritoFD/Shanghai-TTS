from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from app import LocalQueryPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm and persist split torch.compile cache")
    parser.add_argument("--base_model_path", required=True, type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--cache_dir", required=True, type=str)
    parser.add_argument("--compile_mode", default="reduce-overhead", type=str)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--queries_jsonl", default="", type=str)
    parser.add_argument("--limit", default=16, type=int)
    return parser.parse_args()


def load_queries(path: Path, limit: int) -> list[str]:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query = str(row.get("query", "")).strip()
        if query:
            rows.append(query)
        if len(rows) >= limit:
            break
    return rows


def main() -> None:
    args = parse_args()
    preprocessor = LocalQueryPreprocessor(
        checkpoint_path=Path(args.checkpoint_path),
        base_model_path=Path(args.base_model_path),
        load_in_4bit=bool(args.load_in_4bit),
        compile_enabled=True,
        compile_mode=args.compile_mode,
        compile_cache_dir=Path(args.cache_dir),
    )

    queries = ["阿拉上海闲话", "叉开", "炝饼", "两头作虎状"]
    if args.queries_jsonl:
        loaded = load_queries(Path(args.queries_jsonl), args.limit)
        if loaded:
            queries = loaded

    preprocessor.preprocess_batch(queries[: min(len(queries), args.limit)])
    print(f"compile cache warmed under {Path(args.cache_dir).resolve()}")


if __name__ == "__main__":
    main()
