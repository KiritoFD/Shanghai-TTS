# Shanghai-TTS Experiment Log

## Baseline Benchmark — 2026-04-28

### Setup
- **Code**: Phase 1 refactoring complete (shared utils, bug fixes, bench.py rewrite)
- **Test data**: 200 queries (`app/data/test_clean_200.jsonl`), 200 seg_pos samples (`app/seg_pos/data/test.jsonl`)
- **Recall labels**: 179/200 generated via NVIDIA API (kimi-k2-instruct-0905), 21 missing due to rate limiting
- **Hardware**: RTX 4070 Laptop, WSL2 + .venv-wsl
- **Models**: Qwen3.5-2B LoRA (split), bge-m3 (recall), BiLSTM-CRF v3 (seg_pos)

### Results

| Module | Metric | Value | Target | Status |
|--------|--------|-------|--------|--------|
| **split** | valid_json_rate | 1.000 | — | PASS |
| **split** | core_text_em | 0.865 | 0.80 | PASS |
| **split** | type_acc | 0.810 | 0.85 | FAIL |
| **split** | keywords_f1 | 0.731 | 0.70 | PASS |
| **recall** | hit@1 | 0.641 | 0.70 | FAIL |
| **recall** | hit@3 | 0.829 | 0.85 | FAIL |
| **recall** | hit@10 | 0.977 | 0.90 | PASS |
| **recall** | mrr@10 | 0.756 | 0.80 | FAIL |
| **recall** | ndcg@10 | 0.780 | 0.80 | FAIL |
| **seg_pos** | seg_f1 | 0.837 | 0.85 | FAIL |
| **seg_pos** | seg_exact_match | 0.820 | 0.50 | PASS |
| **seg_pos** | pos_acc | 0.778 | 0.75 | PASS |
| **seg_pos** | pos_acc_head | 0.708 | 0.80 | FAIL |

**Total time**: ~44min (split 40min + recall 1.5min + seg_pos 0.2min)

### Analysis

#### Split (Qwen3.5-2B LoRA) — core_text_em=0.865

The Qwen LoRA model performs well:
- 100% valid JSON output
- 86.5% core_text exact match (target 80%)
- 81% type accuracy (target 85%, close)
- 73% keywords F1 (target 70%)

Main weakness: type classification at 81%. Some "动作短语" queries are classified as "词项".

**Performance**: 11.2s per query (40min for 200 queries). Could be optimized with batch inference or KV-caching.

#### Recall — hit@1=0.641, hit@10=0.977

The hybrid retrieval engine works well for hit@10 (0.977) but hit@1 is below target. Issues:
- **Query variant quality**: `merge_query_variants()` generates variants that may dilute the query signal
- **Rerank weights**: dense(0.45) + field(0.85) + fused(0.55) may not be optimal
- **Short query handling**: 2-3 char queries get boosted ANN results that may add noise

**Improvement options**:
1. Tune RRF weights and rerank coefficients systematically
2. Add dialect-specific query expansion (Mandarin → Shanghai mappings)
3. Improve field matching score for definition-based queries

#### SegPos (BiLSTM-CRF v3) — seg_f1=0.837, pos_acc=0.778

Close to targets:
- **seg_f1=0.837** vs target 0.85 — only 1.3% gap
- **pos_acc=0.778** passes target 0.75
- **pos_acc_head=0.708** — POS accuracy on headword chars (non-X) is lower
- **seg_exact_match=0.820** — 82% of samples have perfect segmentation

**Improvement options**:
1. Two-stage training (seg only → POS on frozen encoder)
2. Pretrained char embeddings from bge-m3
3. Self-attention layer after BiLSTM
4. Data augmentation for rare POS tags

### Priority Order

1. **Recall**: hit@1 needs 6% improvement — systematic weight tuning
2. **SegPos**: seg_f1 needs 1.3% improvement — architectural changes (v4)
3. **Split**: type_acc needs 4% improvement — training data augmentation or prompt tuning

### Data Generation Notes

- NVIDIA API (kimi-k2-instruct-0905) generates recall labels from engine candidates
- Rate limiting: 429 errors at 5 concurrent requests, need max_in_flight=1-2
- SSL errors in WSL: can't reach external APIs, must run from Windows
- 179/200 labels generated (21 missing), sufficient for evaluation
- Split labels from existing test_clean_200.jsonl (manual gold labels)

### Key Fixes Applied

1. **bench.py**: Rewrote to 3-module structure (split, recall, seg_pos)
2. **seg_pos_runtime.py**: Fixed v3 import (layer_norm keys)
3. **bench.py split**: Changed from rule-based `normalize_query()` to Qwen3.5-2B LoRA model
4. **bench.py seg_pos**: Fixed to use `segment_text` instead of `query` + `_predict_tags()` directly
5. **engine.py**: Added batch encoding, inverted BM25 index, fast lexical matching
