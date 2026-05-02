"""Grid search over RecallEngine reranking weights.

Tests combinations of dense_weight, field_weight, fused_weight,
penalty_threshold, and penalty_factor to maximize hit@1.

Usage:
    python scripts/tune_recall_weights.py --max_queries 170
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "recall"))

from recall.engine import RecallEngine, encode_queries_batch, merge_query_variants

RECALL_DATA = APP_ROOT / "data" / "test_clean_200_recall.jsonl"
RECALL_INDEX = APP_ROOT / "recall" / "index_local_bge_m3"


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
                if 0 < limit <= len(rows):
                    break
    return rows


def evaluate_weights(
    engine: RecallEngine,
    rows: list[dict],
    vec_map: dict[str, np.ndarray],
    dense_w: float,
    field_w: float,
    fused_w: float,
    penalty_thresh: float,
    penalty_factor: float,
) -> dict[str, float]:
    """Evaluate recall metrics with given reranking weights."""
    # Monkey-patch the rerank weights
    original_rerank = engine._rerank_results

    def patched_rerank(fused_rows, variants, qv, top_n):
        if not fused_rows:
            return []
        fused_scores = [float(row["score"]) for row in fused_rows]
        min_fused = min(fused_scores)
        max_fused = max(fused_scores)
        fused_span = max(max_fused - min_fused, 1e-9)

        reranked = []
        for row in fused_rows:
            row_id = int(row["id"])
            dense_score = engine._row_dense_score(row_id, qv)
            field_score = engine._field_match_score(variants, row)
            fused_score_norm = (float(row["score"]) - min_fused) / fused_span
            final_score = dense_w * dense_score + field_w * field_score + fused_w * fused_score_norm
            if field_score < penalty_thresh:
                final_score *= penalty_factor
            item = dict(row)
            item["score"] = float(final_score)
            reranked.append(item)
        reranked.sort(key=lambda x: float(x["score"]), reverse=True)
        return reranked[:top_n]

    engine._rerank_results = patched_rerank

    hits = {1: 0, 3: 0, 10: 0}
    mrrs = []
    n_valid = 0

    for row in rows:
        target_ids = set(int(tid) for tid in row.get("target_ids", []) if tid is not None)
        if not target_ids:
            continue
        n_valid += 1

        q = str(row.get("user_query", "") or row.get("query", "")).strip()
        result = engine.search(q, top_k=20, top_n=20, precomputed_vecs=vec_map)
        result_ids = [int(r["id"]) for r in result.get("results", [])]

        for k in (1, 3, 10):
            if any(rid in target_ids for rid in result_ids[:k]):
                hits[k] += 1

        rr = 0.0
        for rank, rid in enumerate(result_ids[:10], start=1):
            if rid in target_ids:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    # Restore original
    engine._rerank_results = original_rerank

    return {
        "hit@1": hits[1] / max(n_valid, 1),
        "hit@3": hits[3] / max(n_valid, 1),
        "hit@10": hits[10] / max(n_valid, 1),
        "mrr@10": sum(mrrs) / max(n_valid, 1),
        "n_valid": n_valid,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune recall reranking weights")
    parser.add_argument("--max_queries", type=int, default=170)
    args = parser.parse_args()

    rows = read_jsonl(RECALL_DATA, limit=args.max_queries)
    print(f"Loaded {len(rows)} queries")

    engine = RecallEngine(index_dir=RECALL_INDEX, ann="hnsw", ef_search=64)
    print(f"Engine loaded ({engine.load_s:.1f}s)")

    # Pre-encode
    all_variants = []
    variant_map = []
    for row in rows:
        q = str(row.get("user_query", "") or row.get("query", "")).strip()
        variants = merge_query_variants(q)
        variant_map.append(variants)
        all_variants.extend(variants)
    unique_variants = list(dict.fromkeys(all_variants))
    t0 = time.perf_counter()
    vecs = encode_queries_batch(unique_variants, engine.tokenizer, engine.model, engine.device, batch_size=64)
    print(f"Encoded {len(unique_variants)} variants in {time.perf_counter() - t0:.1f}s")
    vec_map = {v: vecs[i] for i, v in enumerate(unique_variants)}

    # Grid search
    # Current: dense=0.80, field=0.40, fused=0.40, penalty_thresh=0.25, penalty_factor=0.60
    param_grid = {
        "dense_w": [0.60, 0.70, 0.80, 0.90, 1.00],
        "field_w": [0.20, 0.30, 0.40, 0.50, 0.60],
        "fused_w": [0.20, 0.30, 0.40, 0.50],
        "penalty_thresh": [0.15, 0.20, 0.25, 0.30],
        "penalty_factor": [0.40, 0.50, 0.60, 0.70],
    }

    # First pass: coarse grid with fewer combos
    coarse_grid = list(product(
        param_grid["dense_w"],
        param_grid["field_w"],
        param_grid["fused_w"],
        [0.25],  # fixed penalty_thresh
        [0.60],  # fixed penalty_factor
    ))

    print(f"\nCoarse grid: {len(coarse_grid)} combinations")
    best_hit1 = 0
    best_params = None
    results = []

    for i, (dw, fw, fuw, pt, pf) in enumerate(coarse_grid):
        metrics = evaluate_weights(engine, rows, vec_map, dw, fw, fuw, pt, pf)
        results.append({"dense_w": dw, "field_w": fw, "fused_w": fuw, "penalty_thresh": pt, "penalty_factor": pf, **metrics})

        if metrics["hit@1"] > best_hit1:
            best_hit1 = metrics["hit@1"]
            best_params = (dw, fw, fuw, pt, pf)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(coarse_grid)}] best hit@1={best_hit1:.4f}")

    print(f"\nCoarse best: hit@1={best_hit1:.4f}  params={best_params}")

    # Second pass: fine grid around best
    if best_params:
        dw0, fw0, fuw0, pt0, pf0 = best_params
        fine_grid = list(product(
            [max(0.3, dw0 - 0.1), dw0, min(1.2, dw0 + 0.1)],
            [max(0.1, fw0 - 0.1), fw0, min(0.8, fw0 + 0.1)],
            [max(0.1, fuw0 - 0.1), fuw0, min(0.7, fuw0 + 0.1)],
            [max(0.1, pt0 - 0.05), pt0, min(0.4, pt0 + 0.05)],
            [max(0.3, pf0 - 0.1), pf0, min(0.8, pf0 + 0.1)],
        ))
        # Remove duplicates from coarse
        fine_set = set(fine_grid) - set(coarse_grid)
        fine_list = sorted(fine_set)

        print(f"\nFine grid: {len(fine_list)} new combinations")

        for i, (dw, fw, fuw, pt, pf) in enumerate(fine_list):
            metrics = evaluate_weights(engine, rows, vec_map, dw, fw, fuw, pt, pf)
            results.append({"dense_w": dw, "field_w": fw, "fused_w": fuw, "penalty_thresh": pt, "penalty_factor": pf, **metrics})

            if metrics["hit@1"] > best_hit1:
                best_hit1 = metrics["hit@1"]
                best_params = (dw, fw, fuw, pt, pf)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(fine_list)}] best hit@1={best_hit1:.4f}")

    # Sort by hit@1
    results.sort(key=lambda x: x["hit@1"], reverse=True)

    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION:")
    print(f"  dense_w={best_params[0]}, field_w={best_params[1]}, fused_w={best_params[2]}")
    print(f"  penalty_thresh={best_params[3]}, penalty_factor={best_params[4]}")
    print(f"  hit@1={best_hit1:.4f}")

    # Show top 5
    print(f"\nTop 5 configurations:")
    for r in results[:5]:
        print(f"  hit@1={r['hit@1']:.4f}  hit@3={r['hit@3']:.4f}  hit@10={r['hit@10']:.4f}  "
              f"mrr={r['mrr@10']:.4f}  "
              f"dense={r['dense_w']} field={r['field_w']} fused={r['fused_w']} "
              f"pen_t={r['penalty_thresh']} pen_f={r['penalty_factor']}")

    # Save
    out_path = Path(__file__).parent / "weight_tuning_results.json"
    out_path.write_text(json.dumps({
        "best_params": {
            "dense_w": best_params[0],
            "field_w": best_params[1],
            "fused_w": best_params[2],
            "penalty_thresh": best_params[3],
            "penalty_factor": best_params[4],
        },
        "best_hit1": best_hit1,
        "top10": results[:10],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
