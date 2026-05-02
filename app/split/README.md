# split

Query normalization and keyword extraction distillation pipeline.

This subproject implements a teacher-student workflow for the task:

- strip question-shell noise such as `怎么说/怎么讲`
- classify the core input as `词项` or `动作短语`
- extract normalized retrieval keywords
- render the result back to the required production format

## Task definition

The task is treated as structured prediction rather than plain text generation.

Teacher and student both target:

```json
{
  "core_text": "开心",
  "type": "词项",
  "predicate": "",
  "object": "",
  "keywords": ["快乐", "高兴", "愉悦"]
}
```

The final production string is rendered afterwards:

```text
【单词】
快乐,高兴,愉悦
```

## Directory layout

```text
split/
  configs/
  data/
  docs/
  prompts/
  scripts/
  src/
```

## Pipeline

1. Generate seed queries and hard cases
2. Optionally ingest real query-like text from public datasets
2. Use a teacher model through NVIDIA's OpenAI-compatible API
3. Clean and validate teacher outputs with rules
4. Build SFT JSONL for a Qwen3.5-4B student
5. Fine-tune with LoRA
6. Evaluate structured accuracy and rule-violation rate
7. Evaluate retrieval metrics with qrels and corpus data

## Environment

Required environment variables:

- `NVAPI_KEY`: NVIDIA API key
- `SPLIT_TEACHER_MODEL`: optional, defaults to `moonshotai/kimi-k2.5`

The exact teacher model name is configurable because upstream model names can change.
The current default is `moonshotai/kimi-k2.5`.
For JSON distillation stability, the local config also supports a fallback model such as `moonshotai/kimi-k2-instruct-0905`.
You can also store local teacher settings in `split/configs/local.teacher.json`.

## Quick start

### 1. Generate seed queries

```powershell
.venv\Scripts\python.exe split\scripts\generate_seed_queries.py `
  --output split\data\raw\seed_queries.jsonl `
  --count 500
```

### 2. Optionally ingest public datasets

```powershell
.venv\Scripts\python.exe split\scripts\collect_hf_queries.py `
  --dataset DMetaSoul/chinese-semantic-textual-similarity `
  --subset BUSTM `
  --split train `
  --field sentence1 `
  --output split\data\raw\hf_queries.jsonl `
  --limit 5000 `
  --only_chinese_like
```

### 2. Generate teacher labels

```powershell
.venv\Scripts\python.exe split\scripts\teacher_generate.py `
  --input split\data\raw\seed_queries.jsonl `
  --output split\data\raw\teacher_outputs.jsonl `
  --config split\configs\local.teacher.json
```

### 3. Clean teacher labels

```powershell
.venv\Scripts\python.exe split\scripts\clean_teacher_dataset.py `
  --input split\data\raw\teacher_outputs.jsonl `
  --output split\data\processed\teacher_clean.jsonl `
  --reject_output split\data\processed\teacher_rejects.jsonl
```

### 4. Build student SFT data

```powershell
.venv\Scripts\python.exe split\scripts\build_sft_dataset.py `
  --input split\data\processed\teacher_clean.jsonl `
  --output split\data\processed\student_sft.jsonl
```

### 5. Split train/dev/test

```powershell
.venv\Scripts\python.exe split\scripts\split_teacher_dataset.py `
  --input split\data\processed\teacher_clean.jsonl `
  --train_out split\data\processed\teacher_train.jsonl `
  --dev_out split\data\processed\teacher_dev.jsonl `
  --test_out split\data\processed\teacher_test.jsonl
```

### 6. Train student

```powershell
.venv\Scripts\python.exe split\scripts\train_qwen35_4b.py `
  --train_jsonl split\data\processed\student_sft.jsonl `
  --output_dir split\outputs\qwen35_4b_lora `
  --model_name Qwen/Qwen3.5-4B `
  --load_in_4bit
```

### 7. Structured evaluation

```powershell
$env:PYTHONPATH='split/src'
.venv\Scripts\python.exe split\scripts\evaluate_predictions.py `
  --gold split\data\processed\teacher_test.jsonl `
  --pred split\data\processed\student_pred.jsonl
```

### 8. Retrieval evaluation

```powershell
.venv\Scripts\python.exe split\scripts\evaluate_retrieval.py `
  --queries split\data\processed\student_pred.jsonl `
  --corpus split\data\retrieval\corpus.jsonl `
  --qrels split\data\retrieval\qrels.jsonl `
  --k 10
```

## Notes

- A strong rule layer is intentionally kept after the model.
- The student learns structured JSON, not the final formatted string.
- If the teacher response is invalid JSON or violates rules, it is rejected rather than repaired silently.
- Teacher generation is batched on purpose to fit low-RPM API limits.
