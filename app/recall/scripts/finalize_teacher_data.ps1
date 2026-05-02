$ErrorActionPreference = "Stop"

$repo = "G:\GitHub\Shanghai-TTS"
$tasks = "$repo\recall\data\raw\teacher_tasks_20000.jsonl"
$teacher = "$repo\recall\data\raw\teacher_labels_20000.jsonl"
$clean = "$repo\recall\data\processed\match_train_clean.jsonl"
$reject = "$repo\recall\data\processed\match_train_rejects.jsonl"
$sft = "$repo\recall\data\processed\match_train_sft.jsonl"

function Count-Tasks {
  param([string]$path)
  if (!(Test-Path $path)) { return 0 }
  return (Get-Content $path | Measure-Object -Line).Lines
}

function Count-Covered {
  param([string]$path)
  if (!(Test-Path $path)) { return 0 }
  $v = py -3.12 "$repo\split\scripts\count_teacher_covered.py" "$path"
  return [int]$v
}

$target = Count-Tasks -path $tasks
while ($true) {
  $covered = Count-Covered -path $teacher
  Write-Output "covered=$covered target=$target"
  if ($covered -ge $target) { break }
  Start-Sleep -Seconds 30
}

$env:PYTHONHOME = ""
$env:PYTHONPATH = ""
uv run python "$repo\recall\scripts\parse_teacher_labels.py" `
  --tasks_jsonl "$tasks" `
  --teacher_jsonl "$teacher" `
  --output "$clean" `
  --reject_output "$reject"

uv run python "$repo\recall\scripts\build_sft_dataset.py" `
  --input "$clean" `
  --output "$sft"

Write-Output "finalize_done"
