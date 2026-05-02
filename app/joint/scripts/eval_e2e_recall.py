from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "recall"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end recall evaluation: raw vs qwen_lora vs llama_cpp")
    parser.add_argument("--input_jsonl", default="app/joint/data/test.jsonl", type=str)
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--top_n", default=10, type=int)
    parser.add_argument("--output_json", default="app/joint/outputs/e2e_recall_eval.json", type=str)
    parser.add_argument("--mode", default="all", choices=["all", "qwen_lora", "llama_cpp", "raw"], type=str)
    parser.add_argument("--apply_rerank", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def clean_headword(text: str) -> str:
    return re.sub(r"[【】\[\]\s]", "", str(text)).strip()


def eval_raw_recall(rows: list[dict], recall_engine, top_k: int, top_n: int) -> tuple[dict, list[dict]]:
    hits = {1: 0, 3: 0, 10: 0}
    records: list[dict] = []
    for row in rows:
        query = str(row["query"])
        target = clean_headword(row["headword"])
        result = recall_engine.search(query, top_k=top_k, top_n=top_n)
        ranked = [clean_headword(item.get("shanghai", "")) for item in result["results"]]
        for k in hits:
            hits[k] += int(target in ranked[:k])
        records.append({
            "query": query,
            "target": target,
            "mode": "raw",
            "variants": result.get("query_variants", []),
            "ranked": ranked[:10],
        })
    n = max(1, len(rows))
    metrics = {"mode": "raw", "count": len(rows), "hit@1": hits[1] / n, "hit@3": hits[3] / n, "hit@10": hits[10] / n}
    return metrics, records


def eval_with_preprocessor(
    rows: list[dict],
    preprocessor,
    recall_engine,
    app_mod,
    top_k: int,
    top_n: int,
    mode_name: str,
    apply_rerank: bool,
) -> tuple[dict, list[dict]]:
    hits = {1: 0, 3: 0, 10: 0}
    records: list[dict] = []
    for row in rows:
        query = str(row["query"])
        target = clean_headword(row["headword"])
        parsed = preprocessor.preprocess(query)
        terms = app_mod.build_search_terms(query, parsed)
        results, source = app_mod.call_recall_service(query, variants=terms, top_k=top_k, top_n=top_n)
        if apply_rerank:
            results = app_mod.rerank_recall_results(query, parsed, results)
        ranked = [clean_headword(item.get("shanghai", "")) for item in results]
        for k in hits:
            hits[k] += int(target in ranked[:k])
        records.append({
            "query": query,
            "target": target,
            "mode": mode_name,
            "parsed": parsed,
            "terms": terms,
            "ranked": ranked[:10],
        })
    n = max(1, len(rows))
    metrics = {"mode": mode_name, "count": len(rows), "hit@1": hits[1] / n, "hit@3": hits[3] / n, "hit@10": hits[10] / n}
    return metrics, records


def main() -> None:
    args = parse_args()
    import app as app_mod
    app_mod.recall_manager.ensure_running = lambda: False

    rows = read_jsonl(Path(args.input_jsonl), args.limit)
    all_metrics: list[dict] = []
    all_records: list[dict] = []

    if args.mode in ("all", "raw"):
        print("[eval] raw recall (no splitter)...", flush=True)
        from recall.engine import RecallEngine
        engine = RecallEngine(index_dir=app_mod.RECALL_INDEX_DIR, ann="hnsw", ef_search=64)
        metrics, records = eval_raw_recall(rows, engine, args.top_k, args.top_n)
        all_metrics.append(metrics)
        all_records.extend(records)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    if args.mode in ("all", "qwen_lora"):
        print("[eval] qwen_lora + recall...", flush=True)
        backend_bak = app_mod.PREPROCESSOR_BACKEND
        app_mod.PREPROCESSOR_BACKEND = "qwen_lora"
        app_mod.preprocessor = None
        preprocessor = app_mod.ensure_preprocessor()
        metrics, records = eval_with_preprocessor(
            rows, preprocessor, None, app_mod, args.top_k, args.top_n, "qwen_lora", args.apply_rerank
        )
        all_metrics.append(metrics)
        all_records.extend(records)
        app_mod.PREPROCESSOR_BACKEND = backend_bak
        app_mod.preprocessor = None
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    if args.mode in ("all", "llama_cpp"):
        print("[eval] llama_cpp + recall...", flush=True)
        backend_bak = app_mod.PREPROCESSOR_BACKEND
        app_mod.PREPROCESSOR_BACKEND = "llama_cpp"
        app_mod.preprocessor = None
        metrics = {"mode": "llama_cpp", "count": 0, "hit@1": 0.0, "hit@3": 0.0, "hit@10": 0.0}
        try:
            preprocessor = app_mod.ensure_preprocessor()
            metrics, records = eval_with_preprocessor(
                rows, preprocessor, None, app_mod, args.top_k, args.top_n, "llama_cpp", args.apply_rerank
            )
            all_metrics.append(metrics)
            all_records.extend(records)
        except FileNotFoundError:
            print("[eval] llama.cpp split runtime not ready, skipping", flush=True)
        app_mod.PREPROCESSOR_BACKEND = backend_bak
        app_mod.preprocessor = None
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    output = {"metrics": all_metrics, "records": all_records}
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Summary ===", flush=True)
    for m in all_metrics:
        print(f"  {m['mode']}: hit@1={m['hit@1']:.2%}  hit@3={m['hit@3']:.2%}  hit@10={m['hit@10']:.2%}", flush=True)


if __name__ == "__main__":
    main()
