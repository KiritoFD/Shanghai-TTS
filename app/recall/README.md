# Recall (Shanghai + Shaoxing on 8088)

`app/recall/recall_service.py` now serves both recall indexes behind the same HTTP service on port `8088`.

## Index Layout

- Shanghai index: `app/recall/index_local_bge_m3`
- Shaoxing index: `app/recall/index_shaoxing_bge_m3`
- Shanghai source CSV: `app/recall/processed_results.csv`
- Shaoxing source CSV: `app/data/generated/shaoxing_processed.csv`
- Embedding model: `model/bge-m3`

Each index directory contains:

- `meta.json`
- `records.jsonl`
- `embeddings.npy`
- `index_hnsw.bin` when HNSW was built
- `hnsw_meta.json` when HNSW was built

## How vector text is built

`scripts/build_vector_index.py` does not embed the headword alone.

Default `--text_mode headword_definition` builds the dense text as:

`词条 + "。释义：" + 释义`

Available modes:

- `headword`
- `definition`
- `headword_definition` (default)

That same text construction should be kept consistent between Shanghai and Shaoxing, otherwise recall quality becomes hard to compare.

## How HNSW is built

`scripts/build_hnsw_index.py` reads `embeddings.npy`, L2-normalizes vectors, then builds a cosine-space HNSW index.

Useful knobs:

- `--m`: graph degree, default higher means larger index and usually better recall
- `--ef_construction`: build-time search breadth
- `--ef_search`: query-time search breadth

The current runtime usually uses:

- `m = 32`
- `ef_construction = 200`
- `ef_search = 64`

## Build Shanghai index

```bash
python app/recall/scripts/build_vector_index.py \
  --dict_csv app/recall/processed_results.csv \
  --out_dir app/recall/index_local_bge_m3 \
  --model_name_or_path model/bge-m3 \
  --id_col 0 --sh_col 0 --def_col 4 --header none \
  --text_mode headword_definition
```

```bash
python app/recall/scripts/build_hnsw_index.py \
  --index_dir app/recall/index_local_bge_m3 \
  --m 32 --ef_construction 200 --ef_search 64
```

## Build Shaoxing index

If `app/data/generated/shaoxing_processed.csv` is missing, generate it first from `merged_result.xlsx` through the app pipeline.

```bash
python app/recall/scripts/build_vector_index.py \
  --dict_csv app/data/generated/shaoxing_processed.csv \
  --out_dir app/recall/index_shaoxing_bge_m3 \
  --model_name_or_path model/bge-m3 \
  --id_col 0 --sh_col 0 --def_col 4 --header infer \
  --text_mode headword_definition
```

```bash
python app/recall/scripts/build_hnsw_index.py \
  --index_dir app/recall/index_shaoxing_bge_m3 \
  --m 32 --ef_construction 200 --ef_search 64
```

## Run the unified service

```bash
python app/recall/recall_service.py --mode api --port 8088
```

By default it tries to load:

- `--index_dir app/recall/index_local_bge_m3`
- `--shaoxing_index_dir app/recall/index_shaoxing_bge_m3`

Health check:

```bash
curl http://127.0.0.1:8088/health
```

The response now lists both loaded indexes.

## Query examples

Shanghai:

```bash
curl -X POST "http://127.0.0.1:8088/recall" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"你好怎么说\",\"source\":\"shanghai\",\"top_k\":20,\"top_n\":3}"
```

Shaoxing:

```bash
curl -X POST "http://127.0.0.1:8088/recall" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"太阳\",\"source\":\"shaoxing\",\"top_k\":20,\"top_n\":3}"
```

Accepted source aliases:

- `shanghai`
- `shanghai_csv`
- `shaoxing`
- `shaoxing_xlsx`

## Interactive / one-shot

Shanghai interactive:

```bash
python app/recall/recall_service.py --mode interactive --source shanghai
```

Shaoxing interactive:

```bash
python app/recall/recall_service.py --mode interactive --source shaoxing
```

One-shot:

```bash
python app/recall/recall_service.py --mode once --query "你好怎么说"
```
