$ErrorActionPreference = "Stop"

$env:PYTHONHOME = ""
$env:PYTHONPATH = ""

uv run --no-project --python 3.12 --with aiohttp python -u split/scripts/teacher_generate_async.py `
  --input split/data/raw/query_testset_20000_clean_v2.jsonl `
  --output split/data/raw/query_testset_20000_clean_v2_teacher.jsonl `
  --config split/configs/trainset.teacher.json `
  --model moonshotai/kimi-k2-instruct-0905 `
  --resume `
  --launch_interval_s 2 `
  --max_in_flight 10 `
  --verbose
