$ErrorActionPreference = "Stop"

$env:PYTHONHOME = ""
$env:PYTHONPATH = ""

uv run --no-project --python 3.12 --with aiohttp python -u recall/scripts/teacher_generate_recall_async.py `
  --input recall/data/raw/teacher_tasks_20000.jsonl `
  --output recall/data/raw/teacher_labels_20000.jsonl `
  --config split/configs/trainset.teacher.json `
  --model moonshotai/kimi-k2-instruct-0905 `
  --resume `
  --batch_size 8 `
  --launch_interval_s 2 `
  --max_in_flight 10 `
  --verbose
