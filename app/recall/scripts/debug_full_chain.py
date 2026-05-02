from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "recall"))

import app as app_mod  # noqa: E402


DEFAULT_QUERIES = [
    "你好怎么说",
    "我爱你怎么说",
    "我喜欢你怎么说",
    "很累怎么说",
    "夸人帅怎么说",
    "没用怎么说",
    "开心怎么说",
    "帮我一下怎么说",
    "我要回家怎么说",
    "吃饭了吗怎么说",
    "谢谢怎么说",
    "不要紧怎么说",
]


def compact_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results:
        rows.append(
            {
                "shanghai": str(item.get("shanghai", "")).strip(),
                "definition": str(item.get("definition", "")).strip(),
                "score": round(float(item.get("score", 0.0)), 4),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug splitter + recall full chain")
    parser.add_argument("--query", action="append", default=[], help="Query to test; can be repeated")
    parser.add_argument("--top_k", default=30, type=int)
    parser.add_argument("--top_n", default=5, type=int)
    parser.add_argument("--no_lora", action="store_true", help="Skip LoRA and use normalized query only")
    parser.add_argument("--force_local", action="store_true", help="Bypass HTTP recall service")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = args.query or DEFAULT_QUERIES
    if args.force_local:
        app_mod.recall_manager.ensure_running = lambda: False

    preprocessor = None if args.no_lora else app_mod.ensure_preprocessor()
    for query in queries:
        if preprocessor is None:
            parsed = {"core_text": app_mod.normalize_query(query), "keywords": []}
        else:
            parsed = preprocessor.preprocess(query)
        terms = app_mod.build_search_terms(query, parsed)
        results, source = app_mod.call_recall_service(query, variants=terms, top_k=args.top_k, top_n=args.top_n)
        payload = {
            "query": query,
            "parsed": parsed,
            "terms": terms,
            "source": source,
            "results": compact_results(results),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
