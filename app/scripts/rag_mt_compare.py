from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


CASES = [
    "那个... 麻烦帮我查一下，自行车坏了，怎么用上海话表达？",
    "我想问下，今天不想去上学上海话怎么讲",
    "你好用上海话怎么说",
    "谢谢你上海话怎么讲",
    "吃饭了吗怎么用上海话说",
]


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare legacy lookup with RAG-MT pipeline.")
    parser.add_argument("--cases_jsonl", default="", type=str)
    parser.add_argument("--model_size", default="0.8b", choices=["0.8b", "2b"])
    parser.add_argument("--output_json", default="app/rag_mt_data/compare_results.json", type=str)
    parser.add_argument("--skip_legacy_model", action="store_true")
    return parser.parse_args()


def load_cases(path: str) -> list[str]:
    if not path:
        return CASES
    rows: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(str(row.get("query") or row.get("core_text") or row.get("message") or "").strip())
    return [row for row in rows if row]


def summarize_legacy_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    return " | ".join(str(row.get("shanghai", "")).strip() for row in results[:3])


def run_legacy(query: str, app_mod: Any, skip_model: bool) -> dict[str, Any]:
    start = time.perf_counter()
    if skip_model:
        parsed = {"core_text": app_mod.normalize_query(query), "keywords": [], "segments": []}
    else:
        parsed = app_mod.ensure_preprocessor().preprocess(query)
    terms = app_mod.build_search_terms(query, parsed)
    results, source = app_mod.call_recall_service(query, variants=terms, top_k=20, top_n=10)
    results = app_mod.rerank_recall_results(query, parsed, results)[:3]
    elapsed = time.perf_counter() - start
    return {
        "latency_ms": round(elapsed * 1000, 1),
        "parsed": parsed,
        "output": summarize_legacy_results(results),
        "source": source,
        "results": app_mod.summarize_results(results),
    }


def run_rag_mt(query: str, app_mod: Any, model_size: str) -> dict[str, Any]:
    start = time.perf_counter()
    result = app_mod.run_rag_mt_pipeline(query, model_size=model_size)
    elapsed = time.perf_counter() - start
    return {
        "latency_ms": round(elapsed * 1000, 1),
        "output": result.get("text", ""),
        "core_text": result.get("core_text", ""),
        "keywords": result.get("keywords", []),
        "lexicon_context": result.get("lexicon_context", ""),
        "matched_source": result.get("matched_source", ""),
    }


def main() -> None:
    args = parse_args()
    import app as app_mod
    import torch

    cases = load_cases(args.cases_jsonl)
    legacy_by_query: dict[str, dict[str, Any]] = {}
    for query in cases:
        legacy_by_query[query] = run_legacy(query, app_mod, skip_model=args.skip_legacy_model)

    app_mod.preprocessor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for query in cases:
        rag_mt = run_rag_mt(query, app_mod, model_size=args.model_size)
        row = {"query": query, "legacy": legacy_by_query[query], "rag_mt": rag_mt}
        rows.append(row)
        safe_print(json.dumps(row, ensure_ascii=False, indent=2))

    report = {
        "model_size": args.model_size,
        "n": len(rows),
        "legacy_avg_ms": round(statistics.mean(row["legacy"]["latency_ms"] for row in rows), 1) if rows else 0,
        "rag_mt_avg_ms": round(statistics.mean(row["rag_mt"]["latency_ms"] for row in rows), 1) if rows else 0,
        "cases": rows,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    safe_print(f"wrote={out}")


if __name__ == "__main__":
    main()
