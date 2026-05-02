$ErrorActionPreference = "Stop"

$env:PYTHONHOME = ""
$env:PYTHONPATH = ""

uv run --no-project --python 3.12 --with aiohttp python -u split/scripts/teacher_generate_async.py `
  --input split/data/raw/query_train_candidates_merged.jsonl `
  --output split/data/raw/query_train_teacher_5000.jsonl `
  --config split/configs/trainset.teacher.json `
  --model moonshotai/kimi-k2-instruct-0905 `
  --resume `
  --launch_interval_s 2 `
  --max_in_flight 10 `
  --verbose
