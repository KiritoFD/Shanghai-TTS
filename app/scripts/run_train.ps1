$ErrorActionPreference = "Stop"

$env:PYTHONHOME = ""
$env:PYTHONPATH = ""

& "G:\GitHub\Shanghai-TTS\train\.venv\Scripts\python.exe" `
  "G:\GitHub\Shanghai-TTS\train\train_distill.py" `
  --config "G:\GitHub\Shanghai-TTS\train\train_6gb_lora_config.json" `
  --model_name_or_path "Qwen/Qwen3.5-4B" `
  --output_dir "G:\GitHub\Shanghai-TTS\train\model_lora"
