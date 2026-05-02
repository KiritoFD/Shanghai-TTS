"""Analyze recall failures to identify improvement opportunities.

Outputs:
  - Failure patterns (query length, type, error category)
  - Top failing queries with details
  - Actionable recommendations
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

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

    # Analyze each query
    failures = []
    successes = []
    no_target = 0

    for i, row in enumerate(rows):
        target_ids = set(int(tid) for tid in row.get("target_ids", []) if tid is not None)
        if not target_ids:
            no_target += 1
            continue

        q = str(row.get("user_query", "") or row.get("query", "")).strip()
        result = engine.search(q, top_k=20, top_n=20, precomputed_vecs=vec_map)
        result_ids = [int(r["id"]) for r in result.get("results", [])]
        result_items = result.get("results", [])

        hit1 = any(rid in target_ids for rid in result_ids[:1])
        hit3 = any(rid in target_ids for rid in result_ids[:3])
        hit10 = any(rid in target_ids for rid in result_ids[:10])

        # Find rank of first correct result
        first_correct_rank = -1
        for rank, rid in enumerate(result_ids[:20]):
            if rid in target_ids:
                first_correct_rank = rank + 1
                break

        entry = {
            "query": q,
            "variants": variant_map[i],
            "target_ids": sorted(target_ids),
            "result_ids_top5": result_ids[:5],
            "result_top5_shanghai": [
                r.get("shanghai", "") for r in result_items[:5]
            ],
            "hit1": hit1,
            "hit3": hit3,
            "hit10": hit10,
            "first_correct_rank": first_correct_rank,
            "query_len": len(re.sub(r"\s+", "", q)),
        }

        if hit1:
            successes.append(entry)
        else:
            failures.append(entry)

    # Summary
    n_valid = len(rows) - no_target
    hit1_count = len(successes)
    print(f"\n{'='*60}")
    print(f"RECALL FAILURE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total queries: {len(rows)}")
    print(f"With targets: {n_valid}")
    print(f"No targets (skipped): {no_target}")
    print(f"Hit@1: {hit1_count}/{n_valid} = {hit1_count/max(n_valid,1):.3f}")
    print(f"Failures: {len(failures)}")

    # Categorize failures
    print(f"\n--- Failure Categories ---")

    # 1. Query length analysis
    short_fails = [f for f in failures if f["query_len"] <= 4]
    medium_fails = [f for f in failures if 5 <= f["query_len"] <= 8]
    long_fails = [f for f in failures if f["query_len"] > 8]
    print(f"\nBy query length:")
    print(f"  Short (<=4 chars): {len(short_fails)} failures")
    print(f"  Medium (5-8 chars): {len(medium_fails)} failures")
    print(f"  Long (>8 chars): {len(long_fails)} failures")

    # 2. Rank distribution of correct result
    ranks = [f["first_correct_rank"] for f in failures if f["first_correct_rank"] > 0]
    print(f"\nCorrect result rank distribution (failures only):")
    rank_buckets = Counter()
    for r in ranks:
        if r <= 3:
            rank_buckets["rank 2-3"] += 1
        elif r <= 10:
            rank_buckets["rank 4-10"] += 1
        else:
            rank_buckets["rank 11+"] += 1
    for bucket, count in sorted(rank_buckets.items()):
        print(f"  {bucket}: {count}")

    not_found = sum(1 for f in failures if f["first_correct_rank"] == -1)
    print(f"  not in top-20: {not_found}")

    # 3. Top failing queries
    print(f"\n--- Top 20 Failing Queries ---")
    for f in failures[:20]:
        print(f"\n  Q: {f['query']}")
        print(f"  Variants: {f['variants'][:3]}")
        print(f"  Targets: {f['target_ids']}")
        print(f"  Top-5 results: {f['result_ids_top5']}")
        print(f"  Top-5 shanghai: {f['result_top5_shanghai']}")
        print(f"  Correct rank: {f['first_correct_rank']}")

    # 4. Pattern analysis
    print(f"\n--- Pattern Analysis ---")

    # Queries with shell patterns not stripped
    shell_patterns = ["怎么说", "怎么讲", "怎么表达", "是什么意思", "啥意思"]
    shell_fails = [f for f in failures if any(p in f["query"] for p in shell_patterns)]
    print(f"Queries with shell patterns: {len(shell_fails)}")

    # Queries with pronouns/noise
    noise_tokens = ["我", "你", "他", "她", "它", "请看", "麻烦", "这个", "那个"]
    noise_fails = [f for f in failures if any(t in f["query"] for t in noise_tokens)]
    print(f"Queries with noise tokens: {len(noise_fails)}")

    # Multi-concept queries (two distinct concepts)
    multi_concept = [f for f in failures if f["query_len"] > 10]
    print(f"Long queries (>10 chars): {len(multi_concept)}")

    # 5. Save detailed results
    output = {
        "summary": {
            "total": len(rows),
            "with_targets": n_valid,
            "hit1": hit1_count,
            "hit1_rate": round(hit1_count / max(n_valid, 1), 4),
            "failures": len(failures),
        },
        "failures": failures[:50],
        "successes_sample": successes[:10],
    }
    out_path = Path(__file__).parent / "recall_failure_analysis.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
