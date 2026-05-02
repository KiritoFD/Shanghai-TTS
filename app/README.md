# App Workspace

This directory contains the runnable web app and its supporting code.

## Entry points

- `app.py`: full pipeline, from user text to recalled entry to final audio.
- `app_text.py`: text-only debug mode, same retrieval path but no TTS synthesis.

## Split backends

The front-stage split runtime is configurable in `configs/runtime_paths.json`:

- `query_preprocessor.backend = "qwen_lora"` uses the PyTorch Transformers + LoRA path.
- `query_preprocessor.backend = "llama_cpp"` is still available as an explicit debug backend for a GGUF export, but is not the default runtime.

You can override it without editing JSON:

```bash
WUU_PREPROCESSOR_BACKEND=qwen_lora python app/app_text.py
WUU_PREPROCESSOR_BACKEND=llama_cpp python app/app_text.py
```

Both backends produce the same structured fields (`core_text`, `type`, `predicate`, `object`, `keywords`, `segments`).
`app.py` sends those segments together with keywords to recall, so downstream lexical/BM25/bge-m3 search sees the
original query, normalized core, generated keywords, and segmented chunks together.

## Shanghai-side segmentation

The later Shanghai word segmentation and POS stage is now referred to as `seg_shanghai` to distinguish it from the
front-stage Mandarin split. The legacy training/runtime files still live under `app/seg_pos/`, but benchmark/runtime
output should use the `seg_shanghai` name.

## Model layout

All large models are now centralized under the repo-root `model/` directory:

- `../model/Qwen3.5-2B`: base LLM used by the LoRA query normalizer
- `../model/Qwen3.5-2B-split-lora-f16.gguf`: exported GGUF for `llama.cpp` split inference
- `../model/Qwen3.5-4B`: secondary archived LLM asset
- `../model/bge-m3`: embedding model used by recall
- `../model/vits`: TTS artifacts (`config.json` and `checkpoint_48000.pth`)

The app-level runtime config is in `configs/runtime_paths.json`.

## WSL startup

Start from the repo root:

```bash
cd /mnt/g/GitHub/Shanghai-TTS
source .venv-wsl/bin/activate
python app/app.py
```

Text-only debug entry:

```bash
cd /mnt/g/GitHub/Shanghai-TTS
source .venv-wsl/bin/activate
python app/app_text.py
```

`app.py` auto-starts the recall service, so you do not need to launch it manually.

## Health check

```bash
curl http://127.0.0.1:8081/api/user/confinfo
```

## Notes

- Runtime paths are resolved relative to `app/`.
- LoRA checkpoints remain under `app/model_lora/`.
- A smaller incomplete legacy `Qwen3.5-4B` copy was moved to `../model/_unused/Qwen3.5-4B_incomplete_legacy`.
