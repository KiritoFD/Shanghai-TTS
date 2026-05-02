$ErrorActionPreference = "Stop"

$repo = "G:\GitHub\Shanghai-TTS"
$inputJsonl = "$repo\split\data\raw\query_testset_20000_clean_v2.jsonl"
$teacherOut = "$repo\split\data\raw\query_testset_20000_clean_v2_teacher.jsonl"
$cleanOut = "$repo\split\data\processed\query_testset_20000_clean_v2_clean.jsonl"
$rejectOut = "$repo\split\data\processed\query_testset_20000_clean_v2_rejects.jsonl"
$mergeStats = "$repo\train\merge_stats_from_20k_v2.json"

function Get-CoveredCount {
  param([string]$path)
  if (!(Test-Path $path)) { return 0 }
  $value = py -3.12 "$repo\split\scripts\count_teacher_covered.py" "$path"
  return [int]$value
}

$target = (Get-Content $inputJsonl | Measure-Object -Line).Lines
while ($true) {
  $covered = Get-CoveredCount -path $teacherOut
  Write-Output "covered=$covered target=$target"
  if ($covered -ge $target) { break }
  Start-Sleep -Seconds 30
}

$env:PYTHONHOME = ""
$env:PYTHONPATH = ""

py -3.12 "$repo\split\scripts\clean_teacher_dataset.py" `
  --input "$teacherOut" `
  --output "$cleanOut" `
  --reject_output "$rejectOut"

py -3.12 "$repo\split\scripts\merge_train_data.py" `
  --base_train "$repo\train\train_clean.jsonl" `
  --new_clean "$cleanOut" `
  --out_train "$repo\train\train_clean.jsonl" `
  --stats_out "$mergeStats" `
  --prefer_new

py -3.12 "$repo\split\scripts\build_sft_dataset.py" `
  --input "$repo\train\train_clean.jsonl" `
  --output "$repo\train\train_sft.jsonl"

Write-Output "finalize_done"
