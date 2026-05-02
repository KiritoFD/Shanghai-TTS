# WSL Setup Guide

Training and GPU-accelerated inference run in WSL2. The Windows side hosts the Flask app and orchestrates the pipeline.

## Prerequisites

- Windows 11 with WSL2 enabled
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed on Windows (WSL2 uses them automatically)

## Create WSL Virtual Environment

```bash
wsl bash -c "cd /mnt/g/GitHub/Shanghai-TTS && python3 -m venv .venv-wsl && source .venv-wsl/bin/activate && pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch torchaudio && pip install transformers peft bitsandbytes hnswlib flask scipy numpy pandas requests"
```

## Running Scripts in WSL

All training and GPU scripts use this pattern:

```bash
wsl bash -c "cd /mnt/g/GitHub/Shanghai-TTS && source .venv-wsl/bin/activate && python <script>"
```

### Examples

**Train segmentation model:**
```bash
wsl bash -c "cd /mnt/g/GitHub/Shanghai-TTS && source .venv-wsl/bin/activate && python app/seg_pos/train_bilstm_seg_pos_v3.py"
```

**Evaluate segmentation model:**
```bash
wsl bash -c "cd /mnt/g/GitHub/Shanghai-TTS && source .venv-wsl/bin/activate && python app/seg_pos/eval_seg_pos.py"
```

**Run recall service:**
```bash
wsl bash -c "cd /mnt/g/GitHub/Shanghai-TTS && source .venv-wsl/bin/activate && python app/recall/recall_service.py --mode api --port 8088"
```

**Start Flask app (Windows):**
```powershell
cd app
python app.py
```

## GPU Configuration

WSL2 automatically uses the Windows NVIDIA driver. Verify with:

```bash
wsl bash -c "nvidia-smi"
```

Key environment variables:
- `CUDA_VISIBLE_DEVICES=0` — select GPU (default)
- `WUU_TTS_MODEL_DIR` — override TTS model path
- `WUU_TEXT_ONLY=1` — disable TTS, text-only mode
- `WUU_PREPROCESSOR_BACKEND` — override preprocessor (qwen_lora/bilstm_joint/bilstm_seg_pos)

## Troubleshooting

### CUDA not found in WSL
- Ensure NVIDIA drivers are up to date on Windows
- Run `wsl bash -c "nvidia-smi"` to verify

### bitsandbytes errors
- Use `bitsandbytes>=0.49.2` with `torch>=2.11`
- If 4-bit loading fails, set `load_in_4bit: false` in runtime_paths.json

### Port conflicts
- Recall service: port 8088 (configurable in runtime_paths.json)
- Flask app: port 8081 (hardcoded in app.py)
