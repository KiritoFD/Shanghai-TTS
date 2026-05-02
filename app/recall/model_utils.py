from __future__ import annotations

from pathlib import Path


def _resolve_local_path(model_name_or_path: str, base_dir: Path) -> Path:
    p = Path(model_name_or_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def ensure_model_path(model_name_or_path: str, base_dir: Path) -> str:
    """
    Ensure encoder model path is available locally.
    For bge-m3, if local path is missing, auto-download from ModelScope
    into <repo_root>/model/bge-m3 and return that local path.
    """
    local_try = _resolve_local_path(model_name_or_path, base_dir)
    if local_try.exists():
        return str(local_try)

    name = model_name_or_path.lower().replace("\\", "/")
    if "bge-m3" not in name:
        return model_name_or_path

    repo_root = base_dir.parent.parent.resolve()
    target_dir = (repo_root / "model" / "bge-m3").resolve()
    if target_dir.exists():
        return str(target_dir)

    print(f"[model] local model not found: {model_name_or_path}")
    print(f"[model] downloading BAAI/bge-m3 from ModelScope to: {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(
        model_id="BAAI/bge-m3",
        local_dir=str(target_dir),
        cache_dir=str((repo_root / "app" / "recall" / "modelscope_cache").resolve()),
    )
    print("[model] download finished.")
    return str(target_dir)
