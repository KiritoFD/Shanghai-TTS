"""Diagnose ANN vs embedding quality for recall failures.

For each failing query, check:
1. Is the correct result in ANN top-k? (retrieval issue)
2. What's the dense similarity score? (embedding quality issue)
3. Compare ANN ranking vs reranked ranking
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "recall"))

from recall.engine import RecallEngine, encode_queries_batch, merge_query_variants

RECALL_DATA = APP_ROOT / "data" / "test_clean_200_recall.jsonl"
RECALL_INDEX = APP_ROOT / "recall" / "index_local_bge_m3"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def main():
    rows = read_jsonl(RECALL_DATA)
    print(f"Loaded {len(rows)} labeled queries")

    engine = RecallEngine(index_dir=RECALL_INDEX, ann="hnsw", ef_search=64)
    print(f"Records: {len(engine.records)}, dim: {engine.matrix.shape[1]}")

    # Pre-encode
    all_variants = []
    variant_map = []
    for row in rows:
        q = str(row.get("user_query", "") or row.get("query", "")).strip()
        variants = merge_query_variants(q)
        variant_map.append(variants)
        all_variants.extend(variants)
    unique_variants = list(dict.fromkeys(all_variants))
    vecs = encode_queries_batch(unique_variants, engine.tokenizer, engine.model, engine.device, batch_size=64)
    vec_map = {v: vecs[i] for i, v in enumerate(unique_variants)}

    # Diagnose each query
    results = []
    for i, row in enumerate(rows):
        target_ids = set(int(tid) for tid in row.get("target_ids", []) if tid is not None)
        if not target_ids:
            continue

        q = str(row.get("user_query", "") or row.get("query", "")).strip()
        variants = variant_map[i]

        # Get query vector (weighted average of variants)
        all_vecs = np.stack([vec_map[v] for v in variants], axis=0)
        if len(variants) > 1:
            weights = np.array([0.6] + [0.4 / max(len(variants) - 1, 1)] * (len(variants) - 1), dtype=np.float32)
            qv = (all_vecs * weights[:, None]).sum(axis=0).astype(np.float32)
        else:
            qv = all_vecs[0]
        qv = qv / (np.linalg.norm(qv) + 1e-12)

        # ANN search (raw)
        ann_k = 50
        labels, distances = engine.hnsw_index.knn_query(qv.reshape(1, -1), k=ann_k)
        ann_ids = [int(l) for l in labels[0]]
        ann_dists = [float(d) for d in distances[0]]

        # Check if targets are in ANN results
        ann_rank = {}
        for rank, (rid, dist) in enumerate(zip(ann_ids, ann_dists)):
            if rid in target_ids:
                ann_rank[rid] = rank + 1

        # Dense similarity for target entries
        target_sims = {}
        for tid in target_ids:
            pos = engine.record_pos_by_id.get(tid)
            if pos is not None:
                sim = float(np.dot(engine.matrix[pos], qv))
                target_sims[tid] = sim

        # Dense similarity for top-5 ANN results
        top5_sims = []
        for rank, (rid, dist) in enumerate(zip(ann_ids[:5], ann_dists[:5])):
            pos = engine.record_pos_by_id.get(rid)
            sim = float(np.dot(engine.matrix[pos], qv)) if pos is not None else 0.0
            hw = str(engine.records[pos].get("shanghai", "")) if pos is not None else ""
            top5_sims.append({"rank": rank + 1, "id": rid, "sim": round(sim, 4), "cosine_dist": round(dist, 4), "headword": hw})

        # Full pipeline result
        result = engine.search(q, top_k=20, top_n=20, precomputed_vecs=vec_map)
        pipeline_ids = [int(r["id"]) for r in result.get("results", [])]
        hit1 = any(rid in target_ids for rid in pipeline_ids[:1])

        entry = {
            "query": q,
            "variants": variants[:3],
            "target_ids": sorted(target_ids),
            "hit1": hit1,
            "ann_rank_of_targets": ann_rank,
            "target_dense_sims": {str(k): round(v, 4) for k, v in target_sims.items()},
            "top5_ann": top5_sims,
        }
        results.append(entry)

    # Summary
    n = len(results)
    hit1_count = sum(1 for r in results if r["hit1"])
    ann_has_target = sum(1 for r in results if r["ann_rank_of_targets"])
    ann_top1 = sum(1 for r in results if any(rank == 1 for rank in r["ann_rank_of_targets"].values()))
    ann_top5 = sum(1 for r in results if any(rank <= 5 for rank in r["ann_rank_of_targets"].values()))
    ann_top10 = sum(1 for r in results if any(rank <= 10 for rank in r["ann_rank_of_targets"].values()))

    print(f"\n{'='*60}")
    print(f"ANN DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"Total queries: {n}")
    print(f"Pipeline hit@1: {hit1_count}/{n} = {hit1_count/max(n,1):.3f}")
    print(f"ANN has target in top-50: {ann_has_target}/{n} = {ann_has_target/max(n,1):.3f}")
    print(f"ANN target at rank 1: {ann_top1}/{n} = {ann_top1/max(n,1):.3f}")
    print(f"ANN target at rank ≤5: {ann_top5}/{n} = {ann_top5/max(n,1):.3f}")
    print(f"ANN target at rank ≤10: {ann_top10}/{n} = {ann_top10/max(n,1):.3f}")

    # For failing queries, show ANN details
    failures = [r for r in results if not r["hit1"]]
    print(f"\n--- Failing queries ANN analysis ---")
    for f in failures[:15]:
        print(f"\n  Q: {f['query']}")
        print(f"  Variants: {f['variants']}")
        print(f"  Target ANN ranks: {f['ann_rank_of_targets']}")
        print(f"  Target dense sims: {f['target_dense_sims']}")
        if f['top5_ann']:
            top1 = f['top5_ann'][0]
            print(f"  ANN top-1: id={top1['id']} sim={top1['sim']} dist={top1['cosine_dist']} hw={top1['headword']}")

    # Save
    out_path = Path(__file__).parent / "ann_diagnostic.json"
    out_path.write_text(json.dumps({
        "summary": {
            "total": n,
            "pipeline_hit1": hit1_count,
            "ann_in_top50": ann_has_target,
            "ann_at_rank1": ann_top1,
            "ann_at_rank5": ann_top5,
            "ann_at_rank10": ann_top10,
        },
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
