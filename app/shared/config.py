"""Unified configuration loader for Shanghai-TTS.

Absorbs app/path_config.py — provides load_runtime_config() with the same
interface so existing callers keep working unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent.parent  # = app/
REPO_ROOT = APP_ROOT.parent
CONFIG_PATH = APP_ROOT / "configs" / "runtime_paths.json"


def _resolve_path(value: str | None, base_dir: Path = APP_ROOT) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_runtime_config() -> dict[str, Any]:
    """Load and resolve all paths from configs/runtime_paths.json.

    Identical interface to the original path_config.load_runtime_config().
    """
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    llm = config.setdefault("llm", {})
    pipeline = config.setdefault("pipeline", {})
    rag_mt = config.setdefault("rag_mt", {})
    cleaner = config.setdefault("cleaner", {})
    query_preprocessor = config.setdefault("query_preprocessor", {})
    reranker = config.setdefault("reranker", {})
    tts = config.setdefault("tts", {})
    recall = config.setdefault("recall", {})
    data = config.setdefault("data", {})

    llm["base_model_dir"] = _resolve_path(llm.get("base_model_dir"))
    llm["lora_checkpoint_dir"] = _resolve_path(llm.get("lora_checkpoint_dir"))
    llm["gguf_model_path"] = _resolve_path(llm.get("gguf_model_path"), REPO_ROOT)
    llm["llama_cpp_server_path"] = _resolve_path(llm.get("llama_cpp_server_path"), REPO_ROOT)
    llm["compile_enabled"] = bool(llm.get("compile_enabled", False))
    llm["compile_mode"] = str(llm.get("compile_mode", "reduce-overhead"))
    llm["compile_cache_dir"] = _resolve_path(llm.get("compile_cache_dir"), REPO_ROOT)
    llm["llama_cpp_host"] = str(llm.get("llama_cpp_host", "127.0.0.1"))
    llm["llama_cpp_port"] = int(llm.get("llama_cpp_port", 8091))
    llm["llama_cpp_ctx_size"] = int(llm.get("llama_cpp_ctx_size", 2048))
    llm["llama_cpp_n_gpu_layers"] = int(llm.get("llama_cpp_n_gpu_layers", 99))
    llm["llama_cpp_threads"] = int(llm.get("llama_cpp_threads", 8))
    llm["llama_cpp_batch_size"] = int(llm.get("llama_cpp_batch_size", 256))
    llm["llama_cpp_ubatch_size"] = int(llm.get("llama_cpp_ubatch_size", 256))
    llm["llama_cpp_timeout_seconds"] = int(llm.get("llama_cpp_timeout_seconds", 20))
    llm["llama_cpp_auto_start"] = bool(llm.get("llama_cpp_auto_start", True))
    llm["llama_cpp_flash_attention"] = bool(llm.get("llama_cpp_flash_attention", True))
    pipeline["mode"] = str(pipeline.get("mode", "lookup"))
    rag_mt["model_size"] = str(rag_mt.get("model_size", "0.8b"))
    rag_mt["load_in_4bit"] = bool(rag_mt.get("load_in_4bit", False))
    rag_mt["max_new_tokens"] = int(rag_mt.get("max_new_tokens", 64))
    rag_mt["temperature"] = float(rag_mt.get("temperature", 0.0))
    rag_mt["max_keywords"] = int(rag_mt.get("max_keywords", 8))
    rag_mt["max_context_pairs"] = int(rag_mt.get("max_context_pairs", 12))
    rag_mt["max_candidates_per_keyword"] = int(rag_mt.get("max_candidates_per_keyword", 2))
    rag_mt_models = rag_mt.setdefault("models", {})
    for model_config in rag_mt_models.values():
        if not isinstance(model_config, dict):
            continue
        model_config["model_path"] = _resolve_path(model_config.get("model_path"), APP_ROOT)
        model_config["adapter_path"] = _resolve_path(model_config.get("adapter_path"), APP_ROOT)
        model_config["load_in_4bit"] = bool(model_config.get("load_in_4bit", rag_mt["load_in_4bit"]))
    cleaner["enabled"] = bool(cleaner.get("enabled", False))
    cleaner["checkpoint_dir"] = _resolve_path(cleaner.get("checkpoint_dir"), APP_ROOT)
    query_preprocessor["backend"] = str(query_preprocessor.get("backend", "llama_cpp"))
    reranker["enabled"] = bool(reranker.get("enabled", False))
    reranker["checkpoint_dir"] = _resolve_path(reranker.get("checkpoint_dir"))
    tts["model_dir"] = _resolve_path(tts.get("model_dir"))
    data["dictionary_csv"] = _resolve_path(data.get("dictionary_csv"))

    tts_model_dir = tts["model_dir"] or APP_ROOT
    tts["config_path"] = _resolve_path(tts.get("config_file"), tts_model_dir)
    tts["checkpoint_path"] = _resolve_path(tts.get("checkpoint_file"), tts_model_dir)
    recall["script_path"] = _resolve_path(recall.get("script_path"))
    recall["index_dir"] = _resolve_path(recall.get("index_dir"))
    recall["timeout_seconds"] = int(recall.get("timeout_seconds", 5))
    recall["port"] = int(recall.get("port", 8088))
    recall["auto_start"] = bool(recall.get("auto_start", True))
    return config
