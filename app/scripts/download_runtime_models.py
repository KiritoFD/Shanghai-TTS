from __future__ import annotations

import argparse
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download runtime models for Shanghai-TTS into the local model directory.")
    parser.add_argument("--repo-root", required=True, type=str)
    parser.add_argument("--only", nargs="*", default=[], help="Optional subset: qwen bge")
    return parser.parse_args()


def looks_ready(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    if (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists():
        return True
    return False


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    model_root = repo_root / "model"
    model_root.mkdir(parents=True, exist_ok=True)

    requested = {item.strip().lower() for item in args.only if item.strip()}
    specs = [
        ("qwen", "Qwen/Qwen3.5-2B", model_root / "Qwen3.5-2B"),
        ("bge", "BAAI/bge-m3", model_root / "bge-m3"),
    ]

    for short_name, model_id, target_dir in specs:
        if requested and short_name not in requested:
            continue
        if looks_ready(target_dir):
            print(f"[skip] {short_name} ready: {target_dir}", flush=True)
            continue
        print(f"[download] {short_name} -> {target_dir} ({model_id})", flush=True)
        snapshot_download(
            model_id=model_id,
            local_dir=str(target_dir),
            cache_dir=str((Path.home() / ".cache" / "modelscope").resolve()),
        )
        print(f"[done] {short_name} -> {target_dir}", flush=True)


if __name__ == "__main__":
    main()
