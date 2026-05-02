# Joint Segmentation + Embedding Training

This subproject trains the recall encoder with two signals from the local
dictionary spreadsheets:

- retrieval contrastive loss: user-style query -> dictionary entry text
- segmentation auxiliary loss: word-internal chunk labels from `|` in `rare2oo.xlsx`

The segmentation head is an auxiliary training signal. Production recall still
uses dense embeddings plus the existing lexical/BM25/rerank path.

## Build data

```powershell
.\.venv\Scripts\python.exe app\joint\scripts\build_joint_data.py `
  --rare_xlsx app\data\rare2oo.xlsx `
  --sentences_xlsx app\data\output.xlsx `
  --out_dir app\joint\data
```

## Train BiLSTM joint baseline

```powershell
.\.venv\Scripts\python.exe app\joint\scripts\train_bilstm_joint.py `
  --train_jsonl app\joint\data\train.jsonl `
  --dev_jsonl app\joint\data\dev.jsonl `
  --test_jsonl app\joint\data\test.jsonl `
  --corpus_jsonl app\joint\data\corpus.jsonl `
  --output_dir app\joint\outputs\bilstm_joint `
  --epochs 1 `
  --save_every_steps 200 `
  --eval_every_steps 1000
```

The trained checkpoint can be selected by the app with:

```powershell
$env:WUU_PREPROCESSOR_BACKEND='bilstm_joint'
.\.venv\Scripts\python.exe app\app_text.py
```

The default app backend remains `qwen_lora`; change `app/configs/runtime_paths.json` only if you want BiLSTM
to be the persistent default.

## Evaluate Qwen LoRA + existing recall on the same query style

```powershell
.\.venv\Scripts\python.exe app\joint\scripts\evaluate_qwen_lora_recall.py `
  --input_jsonl app\joint\data\test.jsonl `
  --limit 200 `
  --output_json app\joint\outputs\qwen_lora_recall_eval.json
```

## Smoke train

```powershell
.\.venv\Scripts\python.exe app\joint\scripts\train_joint_encoder.py `
  --train_jsonl app\joint\data\train.jsonl `
  --dev_jsonl app\joint\data\dev.jsonl `
  --model_name_or_path model\bge-m3 `
  --output_dir app\joint\outputs\bge_m3_joint `
  --max_steps 10 `
  --batch_size 4
```

## Build a recall index from the trained encoder

```powershell
.\.venv\Scripts\python.exe app\recall\scripts\build_vector_index.py `
  --dict_csv app\recall\processed_results.csv `
  --out_dir app\recall\index_bge_m3_joint `
  --model_name_or_path app\joint\outputs\bge_m3_joint `
  --id_col 0 --sh_col 0 --def_col 4 --header none `
  --text_mode headword_definition

.\.venv\Scripts\python.exe app\recall\scripts\build_hnsw_index.py `
  --index_dir app\recall\index_bge_m3_joint `
  --m 32 --ef_construction 200 --ef_search 64
```
