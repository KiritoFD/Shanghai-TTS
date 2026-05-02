# Recall (Encoder + HNSW)

This directory can run on its own without absolute paths.

## Local Assets

- Dictionary: `processed_results.csv`
- Encoder models:
  - `../../model/bge-m3`

## Index Text

The dense index is not built from the headword alone.

By default, `scripts/build_vector_index.py` embeds:

`词条 + "。释义：" + 释义`

This is controlled by `--text_mode`:

- `headword`
- `definition`
- `headword_definition` (default)

So if recall quality is poor, the first thing to check is usually not "did we only index the word itself",
but whether the online path and offline test path are using the same retrieval logic.

## Quick Start

1. Install dependencies

```bash
cd /mnt/g/GitHub/Shanghai-TTS
source .venv-wsl/bin/activate
```

2. Build a vector index

```bash
python app/recall/scripts/build_vector_index.py \
  --dict_csv app/recall/processed_results.csv \
  --out_dir app/recall/index_local_bge_m3 \
  --model_name_or_path model/bge-m3 \
  --id_col 0 --sh_col 0 --def_col 4 --header none \
  --text_mode headword_definition
```

3. Build an HNSW index

```bash
python app/recall/scripts/build_hnsw_index.py \
  --index_dir app/recall/index_local_bge_m3 \
  --m 32 --ef_construction 200 --ef_search 64
```

4. Run the unified interface

Default mode is interactive:

```bash
python app/recall/recall_service.py
```

If `model/bge-m3` is missing, startup auto-downloads it from ModelScope into the root `model/` directory.

`recall_service.py` is the current online retrieval path. It includes:

- query normalization
- query variants
- lexical retrieval
- sparse BM25 retrieval
- ANN retrieval
- rank fusion
- lightweight reranking

Older helper scripts such as `infer.py` may show weaker results because they do not fully mirror
the online retrieval logic.

API mode:

```bash
python app/recall/recall_service.py --mode api --port 8088
```

Request example:

```bash
curl -X POST "http://127.0.0.1:8088/recall" \
  -H "Content-Type: application/json" \
  -d '{"query":"你好怎么说","top_k":20,"top_n":3}'
```

One-shot mode:

```bash
python app/recall/recall_service.py --mode once --query "你好怎么说"
```

## Full Chain Debug

Use this script to inspect how the LoRA splitter cooperates with recall:

```bash
python app/recall/scripts/debug_full_chain.py --force_local --query "你好怎么说" --query "我爱你怎么说"
```
