"""Evaluate BiLSTM seg+POS model on test set and compare with qwen_lora."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "recall"))
sys.path.insert(0, str(APP_ROOT / "seg_pos"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate seg+POS preprocessor with recall")
    parser.add_argument("--input_jsonl", default="app/seg_pos/data/test.jsonl", type=str)
    parser.add_argument("--limit", default=200, type=int)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--top_n", default=10, type=int)
    parser.add_argument("--output_json", default="app/seg_pos/outputs/seg_pos_recall_eval.json", type=str)
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


def main() -> None:
    args = parse_args()
    import app as app_mod
    app_mod.recall_manager.ensure_running = lambda: False

    from recall.engine import RecallEngine
    engine = RecallEngine(index_dir=app_mod.RECALL_INDEX_DIR, ann="hnsw", ef_search=64)

    rows = read_jsonl(Path(args.input_jsonl), args.limit)

    backends_to_test = []

    # BiLSTM seg+POS
    try:
        from seg_pos_runtime import SegPosPreprocessor
        seg_pos_ckpt = Path(APP_ROOT / "seg_pos" / "outputs" / "bilstm_seg_pos" / "best")
        if not seg_pos_ckpt.exists():
            seg_pos_ckpt = Path(APP_ROOT / "seg_pos" / "outputs" / "bilstm_seg_pos" / "final")
        seg_pos_prep = SegPosPreprocessor(checkpoint_dir=seg_pos_ckpt)
        backends_to_test.append(("bilstm_seg_pos", seg_pos_prep))
    except Exception as e:
        print(f"[seg_pos] failed to load: {e}", flush=True)

    # BiLSTM joint (old)
    try:
        from joint.bilstm_runtime import BilstmJointPreprocessor
        bilstm_ckpt = Path(APP_ROOT / "joint" / "outputs" / "bilstm_joint_app" / "final")
        if bilstm_ckpt.exists():
            bilstm_prep = BilstmJointPreprocessor(checkpoint_dir=bilstm_ckpt)
            backends_to_test.append(("bilstm_joint", bilstm_prep))
    except Exception as e:
        print(f"[bilstm_joint] failed to load: {e}", flush=True)

    # Raw (no splitter)
    backends_to_test.append(("raw", None))

    all_results: dict[str, dict] = {}

    for backend_name, preprocessor in backends_to_test:
        print(f"\n=== Evaluating {backend_name} ===", flush=True)
        hits = {1: 0, 3: 0, 10: 0}
        records: list[dict] = []

        for idx, row in enumerate(rows, start=1):
            query = str(row["query"])
            target = clean_headword(row["headword"])

            if preprocessor is None:
                parsed = {"core_text": app_mod.normalize_query(query), "keywords": [], "segments": [], "pos_tags": []}
            else:
                parsed = preprocessor.preprocess(query)

            terms = app_mod.build_search_terms(query, parsed)
            result = engine.search(query, top_k=args.top_k, top_n=args.top_n, extra_variants=terms)
            ranked = [clean_headword(item.get("shanghai", "")) for item in result["results"]]
            for k in hits:
                hits[k] += int(target in ranked[:k])

            if idx <= 20 or idx % 50 == 0:
                print(f"  [{idx}/{len(rows)}] Q={query[:30]:30s} T={target:10s} h1={'Y' if target in ranked[:1] else 'N'} top1={ranked[0][:15] if ranked else '-'}", flush=True)

            records.append({
                "query": query, "target": target, "parsed": parsed,
                "terms": terms[:6], "ranked": ranked[:10],
            })

        n = max(1, len(rows))
        metrics = {
            "backend": backend_name, "count": len(rows),
            "hit@1": hits[1] / n, "hit@3": hits[3] / n, "hit@10": hits[10] / n,
        }
        all_results[backend_name] = {"metrics": metrics, "records": records[:30]}
        print(f"  => hit@1={metrics['hit@1']:.2%}  hit@3={metrics['hit@3']:.2%}  hit@10={metrics['hit@10']:.2%}", flush=True)

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for name, result in all_results.items():
        m = result["metrics"]
        print(f"  {name:20s}  h@1={m['hit@1']:.2%}  h@3={m['hit@3']:.2%}  h@10={m['hit@10']:.2%}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
