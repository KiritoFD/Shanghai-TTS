from __future__ import annotations

import atexit
import copy
import html
import importlib
import importlib.util
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import torch
from flask import Flask, jsonify, request, send_from_directory, session
from peft import PeftModel
from cleaner_model import CleanerCoreExtractor
from dictionary_sources import (
    PRIMARY_SOURCE_ID,
    SHAOXING_CSV_PATH,
    SHAOXING_INDEX_DIR,
    SHAOXING_SOURCE_ID,
    SHAOXING_XLSX_PATH,
    ensure_dense_index,
    ensure_shaoxing_processed_csv,
    index_complete,
)
from path_config import load_runtime_config
from rag_mt import QwenRAGMTTranslator, build_lexicon_context, clean_query
from rerank.runtime import SplitRecallReranker
from split.llama_cpp_runtime import LlamaCppConfig, LlamaCppQueryPreprocessor
from shared.text import normalize_query
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from werkzeug.exceptions import HTTPException


ZH_PROMPT = (
    "\u4f60\u662f\u4e2d\u6587\u67e5\u8be2\u89c4\u8303\u5316\u52a9\u624b\u3002"
    "\u7ed9\u5b9a\u7528\u6237\u8f93\u5165\u540e\uff0c\u53ea\u8f93\u51fa\u4e00\u4e2a JSON \u5bf9\u8c61\uff0c"
    "\u5b57\u6bb5\u5fc5\u987b\u4e3a\uff1acore_text,type,predicate,object,segments,keywords\u3002"
    "type \u53ea\u80fd\u662f\"\u8bcd\u9879\"\u6216\"\u52a8\u4f5c\u77ed\u8bed\"\u3002"
    "segments \u662f\u539f\u6587\u62c6\u5206\u540e\u7684\u5b50\u77ed\u8bed\uff0c\u5fc5\u987b\u4fdd\u7559\u539f\u59cb\u7528\u8bcd\uff0c\u4e0d\u8981\u540c\u4e49\u66ff\u6362\u3002"
    "\u4f8b\uff1a\u201c\u4e24\u5934\u4f5c\u864e\u72b6\u201d\u2192 segments=[\"两头\",\"作虎状\"], \u800c\u4e0d\u662f[\"厉害\",\"强\"]\u3002"
    "\u4f8b\uff1a\u201c\u53c9\u5f00\u201d\u2192 segments=[\"叉开\"], \u800c\u4e0d\u662f[\"分开\"]\u3002"
    "keywords \u5fc5\u987b\u662f\u4e2d\u6587\u8bcd\u6570\u7ec4\uff0c\u4ec5\u5728\u539f\u59cb\u8bcd\u65e0\u6cd5\u76f4\u63a5\u641c\u7d22\u65f6\u624d\u63d0\u4f9b\u540c\u4e49\u8bcd\uff0c\u540c\u65f6\u4fdd\u7559\u539f\u59cb\u8bcd\u3002"
    "\u4e0d\u8981\u8f93\u51fa\u4efb\u4f55\u89e3\u91ca\u3002"
)

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
STATIC_DIR = ROOT / "static"
FRONTEND_DIR = REPO_ROOT / "frontend"
LOG_DIR = ROOT / "logs"
RUNTIME_CONFIG = load_runtime_config()
CHECKPOINT_PATH = Path(RUNTIME_CONFIG["llm"]["lora_checkpoint_dir"])
BASE_MODEL_PATH = Path(RUNTIME_CONFIG["llm"]["base_model_dir"])
PREPROCESSOR_BACKEND = os.getenv(
    "WUU_PREPROCESSOR_BACKEND",
    str(RUNTIME_CONFIG.get("query_preprocessor", {}).get("backend", "llama_cpp")),
).strip()
RECALL_API_URL = str(RUNTIME_CONFIG["recall"]["api_url"])
RECALL_HEALTH_URL = str(RUNTIME_CONFIG["recall"]["health_url"])
RECALL_SCRIPT_PATH = Path(RUNTIME_CONFIG["recall"]["script_path"])
RECALL_INDEX_DIR = Path(RUNTIME_CONFIG["recall"]["index_dir"])
RECALL_HOST = str(RUNTIME_CONFIG["recall"]["host"])
RECALL_PORT = int(RUNTIME_CONFIG["recall"]["port"])
RECALL_TIMEOUT = int(RUNTIME_CONFIG["recall"]["timeout_seconds"])
RECALL_AUTO_START = bool(RUNTIME_CONFIG["recall"]["auto_start"])
LOAD_IN_4BIT = bool(RUNTIME_CONFIG["llm"].get("load_in_4bit", False))
GGUF_MODEL_PATH = Path(RUNTIME_CONFIG["llm"]["gguf_model_path"]) if RUNTIME_CONFIG["llm"].get("gguf_model_path") else None
LLAMA_CPP_SERVER_PATH = Path(RUNTIME_CONFIG["llm"]["llama_cpp_server_path"]) if RUNTIME_CONFIG["llm"].get("llama_cpp_server_path") else None
LLAMA_CPP_HOST = str(RUNTIME_CONFIG["llm"].get("llama_cpp_host", "127.0.0.1"))
LLAMA_CPP_PORT = int(RUNTIME_CONFIG["llm"].get("llama_cpp_port", 8091))
LLAMA_CPP_CTX_SIZE = int(RUNTIME_CONFIG["llm"].get("llama_cpp_ctx_size", 2048))
LLAMA_CPP_N_GPU_LAYERS = int(RUNTIME_CONFIG["llm"].get("llama_cpp_n_gpu_layers", 99))
LLAMA_CPP_THREADS = int(RUNTIME_CONFIG["llm"].get("llama_cpp_threads", 8))
LLAMA_CPP_BATCH_SIZE = int(RUNTIME_CONFIG["llm"].get("llama_cpp_batch_size", 256))
LLAMA_CPP_UBATCH_SIZE = int(RUNTIME_CONFIG["llm"].get("llama_cpp_ubatch_size", 256))
LLAMA_CPP_TIMEOUT_SECONDS = int(RUNTIME_CONFIG["llm"].get("llama_cpp_timeout_seconds", 20))
LLAMA_CPP_AUTO_START = bool(RUNTIME_CONFIG["llm"].get("llama_cpp_auto_start", True))
LLAMA_CPP_FLASH_ATTENTION = bool(RUNTIME_CONFIG["llm"].get("llama_cpp_flash_attention", True))
RERANKER_ENABLED = bool(RUNTIME_CONFIG.get("reranker", {}).get("enabled", False))
RERANKER_CHECKPOINT_DIR = RUNTIME_CONFIG.get("reranker", {}).get("checkpoint_dir")
PIPELINE_MODE = os.getenv(
    "WUU_PIPELINE_MODE",
    str(RUNTIME_CONFIG.get("pipeline", {}).get("mode", "lookup")),
).strip().lower()
RAG_MT_CONFIG = RUNTIME_CONFIG.get("rag_mt", {})
RAG_MT_MODEL_SIZE = os.getenv("WUU_RAG_MT_MODEL_SIZE", str(RAG_MT_CONFIG.get("model_size", "2b"))).strip().lower()
PRELOAD_MODELS = os.getenv("WUU_PRELOAD_MODELS", "1") != "0"
PRELOAD_ALL_RAG_MT_MODELS = os.getenv("WUU_PRELOAD_ALL_RAG_MT_MODELS", "0") == "1"
PRELOAD_LEGACY_PREPROCESSOR = os.getenv("WUU_PRELOAD_LEGACY_PREPROCESSOR", "0") == "1"
CLEANER_CONFIG = RUNTIME_CONFIG.get("cleaner", {})
TEXT_ONLY_MODE = os.getenv("WUU_TEXT_ONLY", "0") == "1"
_synthesize_fn = None
local_recall_engine = None
exact_dictionary_index: dict[str, list[dict[str, Any]]] | None = None
shaoxing_recall_engine = None
shaoxing_exact_index: dict[str, list[dict[str, Any]]] | None = None
COMPILE_ENABLED = bool(RUNTIME_CONFIG["llm"].get("compile_enabled", False))
COMPILE_MODE = str(RUNTIME_CONFIG["llm"].get("compile_mode", "reduce-overhead"))
COMPILE_CACHE_DIR = Path(RUNTIME_CONFIG["llm"].get("compile_cache_dir") or (REPO_ROOT / ".cache" / "torchinductor_split"))


def resolve_dictionary_path() -> Path | None:
    configured = Path(RUNTIME_CONFIG["data"]["dictionary_csv"])
    if configured.exists():
        return configured
    logging.warning("[dict] primary dictionary missing: %s", configured)
    return None


DICT_PATH = resolve_dictionary_path()
SHAOXING_PROCESSED_PATH = ensure_shaoxing_processed_csv()


def _tts_module() -> Any:
    return importlib.import_module("tts_engine")


def synthesize(text: str, output_path: str, model_name: str | None = None) -> None:
    module = _tts_module()
    synthesize_fn = getattr(module, "synthesize")
    synthesize_fn(text, output_path, model_name=model_name)


def ensure_tts_model_loaded(model_name: str | None = None, device: str | None = None) -> dict[str, Any] | None:
    if TEXT_ONLY_MODE:
        return None
    module = _tts_module()
    load_model = getattr(module, "load_model", None)
    if callable(load_model):
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return load_model(model_name, device=target_device)
    return None


def set_tts_model_device(model_name: str | None = None, device: str = "cpu") -> dict[str, Any] | None:
    if TEXT_ONLY_MODE:
        return None
    module = _tts_module()
    load_model = getattr(module, "load_model", None)
    if callable(load_model):
        return load_model(model_name, device=device)
    return None


def tts_runtime_status() -> dict[str, Any]:
    if TEXT_ONLY_MODE:
        return {"cuda_available": torch.cuda.is_available(), "default_model": None, "active_direct_model": None, "models": {}}
    module = _tts_module()
    status_fn = getattr(module, "tts_status", None)
    if callable(status_fn):
        return status_fn()
    return {"cuda_available": torch.cuda.is_available(), "default_model": None, "active_direct_model": None, "models": {}}


def tts_model_ready(model_name: str) -> bool:
    module = _tts_module()
    ready_fn = getattr(module, "model_ready", None)
    if callable(ready_fn):
        return bool(ready_fn(model_name))
    return False


def set_tts_active_direct_model(model_name: str) -> str:
    module = _tts_module()
    setter = getattr(module, "set_active_direct_model_name")
    return str(setter(model_name))


def active_direct_tts_model() -> str:
    module = _tts_module()
    getter = getattr(module, "get_active_direct_model_name")
    return str(getter())


def tts_ready() -> bool:
    if TEXT_ONLY_MODE:
        return False
    status = tts_runtime_status()
    models = status.get("models", {})
    return bool(models) and all(bool(item.get("ready")) for item in models.values())


class RecallServiceManager:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.start_attempted = False
        self.warning_emitted = False

    def is_healthy(self) -> bool:
        try:
            response = requests.get(RECALL_HEALTH_URL, timeout=min(RECALL_TIMEOUT, 3))
            return response.status_code == 200
        except Exception:
            return False

    def ensure_running(self) -> bool:
        if self.is_healthy():
            return True
        if not RECALL_AUTO_START:
            return False
        if self.process is not None and self.process.poll() is None:
            return self._wait_until_healthy(timeout_s=45.0)
        return self.start(wait=True)

    def start(self, wait: bool = True) -> bool:
        self.start_attempted = True
        if not RECALL_SCRIPT_PATH.exists():
            logging.warning("[recall] script not found: %s", RECALL_SCRIPT_PATH)
            return False

        recall_log_dir = ROOT / "recall" / "logs"
        recall_log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stdout_path = recall_log_dir / f"recall_service_{stamp}.out.log"
        stderr_path = recall_log_dir / f"recall_service_{stamp}.err.log"
        stdout_file = open(stdout_path, "a", encoding="utf-8")
        stderr_file = open(stderr_path, "a", encoding="utf-8")
        cmd = [
            sys.executable,
            str(RECALL_SCRIPT_PATH),
            "--mode",
            "api",
            "--host",
            RECALL_HOST,
            "--port",
            str(RECALL_PORT),
        ]
        logging.info("[recall] starting service: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        if not wait:
            return True
        return self._wait_until_healthy(timeout_s=45.0)

    def _wait_until_healthy(self, timeout_s: float = 20.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.is_healthy():
                logging.info("[recall] service is healthy")
                self.warning_emitted = False
                return True
            if self.process is not None and self.process.poll() is not None:
                break
            time.sleep(0.5)
        if not self.warning_emitted:
            logging.warning("[recall] service is unavailable after startup wait")
            self.warning_emitted = True
        return False

    def shutdown(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is not None:
            return
        logging.info("[recall] stopping child process")
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


recall_manager = RecallServiceManager()
atexit.register(recall_manager.shutdown)


class Tee:
    def __init__(self, filename: Path, mode: str = "a", encoding: str = "utf-8"):
        self.file = open(filename, mode, encoding=encoding, buffering=1)
        self.stdout = sys.stdout

    def write(self, message: str) -> None:
        try:
            self.stdout.write(message)
        except OSError:
            pass
        self.file.write(message)

    def flush(self) -> None:
        try:
            self.stdout.flush()
        except OSError:
            pass
        self.file.flush()


class LocalQueryPreprocessor:
    def __init__(
        self,
        checkpoint_path: Path,
        base_model_path: Path,
        load_in_4bit: bool = True,
        compile_enabled: bool = False,
        compile_mode: str = "reduce-overhead",
        compile_cache_dir: Path | None = None,
    ):
        self.checkpoint_path = checkpoint_path.resolve()
        self.base_model_dir = base_model_path.resolve()
        self.compile_enabled = bool(compile_enabled)
        self.compile_mode = str(compile_mode)
        self.compile_cache_dir = compile_cache_dir.resolve() if compile_cache_dir else None

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.base_model_dir.exists():
            raise FileNotFoundError(f"Base model not found: {self.base_model_dir}")

        print(f"[model] base: {self.base_model_dir}")
        print(f"[model] lora: {self.checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.base_model_dir), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.json_end_token_ids = self.tokenizer.encode("}", add_special_tokens=False)
        self.json_eos_token_id: int | list[int] = self.tokenizer.eos_token_id
        if len(self.json_end_token_ids) == 1:
            self.json_eos_token_id = [self.tokenizer.eos_token_id, self.json_end_token_ids[0]]

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            model_kwargs["dtype"] = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        base_model = AutoModelForCausalLM.from_pretrained(str(self.base_model_dir), **model_kwargs)
        self.model = PeftModel.from_pretrained(base_model, str(self.checkpoint_path))
        if not load_in_4bit and hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
        self._maybe_configure_compile_cache()
        self._maybe_compile_model(load_in_4bit=load_in_4bit)
        self.model.eval()
        self._warmed_up = False
        print("[model] loaded")

    def _maybe_configure_compile_cache(self) -> None:
        if self.compile_cache_dir is None:
            return
        self.compile_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(self.compile_cache_dir))
        try:
            import torch._inductor.config as inductor_config

            if hasattr(inductor_config, "fx_graph_cache"):
                inductor_config.fx_graph_cache = True
        except Exception:
            pass

    def _maybe_compile_model(self, load_in_4bit: bool) -> None:
        if not self.compile_enabled or load_in_4bit or not torch.cuda.is_available():
            return
        if importlib.util.find_spec("triton") is None:
            print("[model] torch.compile skipped: Triton is not installed")
            return
        if not hasattr(torch, "compile"):
            return
        try:
            self.model.forward = torch.compile(
                self.model.forward,
                mode=self.compile_mode,
                fullgraph=False,
            )
            print("[model] torch.compile enabled for split forward")
        except Exception as exc:
            print(f"[model] torch.compile skipped: {exc}")

    def _ensure_warmup(self) -> None:
        if self._warmed_up:
            return
        try:
            warm_prompt = self._build_prompt("阿拉上海闲话")
            self._generate_from_prompts([warm_prompt], max_new_tokens=16, temperature=0.0, skip_warmup=True)
            print("[model] split warmup complete")
        except Exception as exc:
            print(f"[model] split warmup skipped: {exc}")
        self._warmed_up = True

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return self._extract_legacy_output(text)

    def _extract_legacy_output(self, text: str) -> dict[str, Any]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return {}

        header = lines[0]
        if "\u77ed\u8bed" in header:
            type_name = "\u52a8\u4f5c\u77ed\u8bed"
        elif "\u5355\u8bcd" in header or "\u8bcd\u9879" in header:
            type_name = "\u8bcd\u9879"
        else:
            type_name = ""

        keyword_line = ""
        for line in lines[1:]:
            if re.search(r"[\u4e00-\u9fff]", line):
                keyword_line = line
                break

        keywords = [part.strip() for part in re.split(r"[,，]", keyword_line) if part.strip()]
        segments = keywords[:3] if keywords else []
        return {
            "core_text": keywords[0] if keywords else "",
            "type": type_name,
            "predicate": keywords[0] if type_name == "\u52a8\u4f5c\u77ed\u8bed" and keywords else "",
            "object": keywords[1] if type_name == "\u52a8\u4f5c\u77ed\u8bed" and len(keywords) > 1 else "",
            "segments": segments,
            "keywords": keywords,
        }

    def _build_prompt(self, query: str) -> str:
        messages = [
            {"role": "system", "content": ZH_PROMPT},
            {"role": "user", "content": f"\u7528\u6237\u8f93\u5165\uff1a{query}"},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def _generate_from_prompts(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        skip_warmup: bool = False,
    ) -> list[dict[str, Any]]:
        if not skip_warmup:
            self._ensure_warmup()
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.json_eos_token_id,
            "use_cache": True,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            outputs = self.model.generate(**encoded, **gen_kwargs)
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        return [
            self._extract_json(
                self.tokenizer.decode(outputs[row_index][int(input_len) :], skip_special_tokens=True)
            )
            for row_index, input_len in enumerate(input_lengths)
        ]

    def preprocess(self, query: str, max_new_tokens: int = 64, temperature: float = 0.0) -> dict[str, Any]:
        return self._generate_from_prompts([self._build_prompt(query)], max_new_tokens=max_new_tokens, temperature=temperature)[0]

    def preprocess_batch(self, queries: list[str], max_new_tokens: int = 64, temperature: float = 0.0) -> list[dict[str, Any]]:
        if not queries:
            return []
        prompts = [self._build_prompt(query) for query in queries]
        return self._generate_from_prompts(prompts, max_new_tokens=max_new_tokens, temperature=temperature)


def empty_dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(range(7)))


def load_dataframe(path: Path | None = None) -> pd.DataFrame:
    # header=0: CSV now has a header row (entry,entry_alt,romanization,ipa,definition,tone_notation,notes)
    # We read with header=0 so pandas assigns column names, but downstream code
    # uses integer column positions (row[0], row[4], row[5]) via .iloc — keep that
    # working by resetting to positional int columns after load.
    csv_path = path or DICT_PATH
    if csv_path is None or not csv_path.exists():
        return empty_dictionary_frame()
    _df = pd.read_csv(csv_path, header=0, encoding="utf-8-sig")
    _df.columns = list(range(len(_df.columns)))
    return _df




def call_recall_service(
    query: str,
    variants: list[str] | None = None,
    top_k: int = 20,
    top_n: int = 3,
    source: str = "shanghai",
) -> tuple[list[dict[str, Any]], str]:
    normalized_source = "shaoxing" if str(source).strip().lower() in {"shaoxing", SHAOXING_SOURCE_ID} else "shanghai"
    target_index_dir = SHAOXING_INDEX_DIR if normalized_source == "shaoxing" else RECALL_INDEX_DIR
    if normalized_source == "shanghai":
        if DICT_PATH is None or not DICT_PATH.exists() or not index_complete(target_index_dir):
            return [], ""
    elif not index_complete(target_index_dir):
        return [], ""

    def call_local_recall_engine() -> tuple[list[dict[str, Any]], str]:
        global local_recall_engine, shaoxing_recall_engine
        try:
            if normalized_source == "shaoxing":
                if shaoxing_recall_engine is None:
                    logging.info("[recall] loading in-process shaoxing fallback engine from %s", SHAOXING_INDEX_DIR)
                    from recall.engine import RecallEngine

                    shaoxing_recall_engine = RecallEngine(index_dir=SHAOXING_INDEX_DIR, ann="hnsw", ef_search=64)
                result = shaoxing_recall_engine.search(query, top_k=top_k, top_n=top_n, extra_variants=variants or [])
                return result.get("results", []), "local_recall_engine_shaoxing"
            if local_recall_engine is None:
                logging.info("[recall] loading in-process fallback engine from %s", RECALL_INDEX_DIR)
                from recall.engine import RecallEngine

                local_recall_engine = RecallEngine(index_dir=RECALL_INDEX_DIR, ann="hnsw", ef_search=64)
            result = local_recall_engine.search(query, top_k=top_k, top_n=top_n, extra_variants=variants or [])
            return result.get("results", []), "local_recall_engine_shanghai"
        except Exception as exc:
            logging.warning("[recall] local engine fallback failed: %s", exc)
            return [], ""

    if not recall_manager.ensure_running():
        if not recall_manager.warning_emitted:
            logging.warning("[recall] service unavailable, fallback to in-process recall engine")
            recall_manager.warning_emitted = True
        return call_local_recall_engine()
    try:
        response = requests.post(
            RECALL_API_URL,
            json={"query": query, "source": normalized_source, "variants": variants or [], "top_k": top_k, "top_n": top_n},
            timeout=RECALL_TIMEOUT,
        )
        if response.status_code != 200:
            return call_local_recall_engine()
        payload = response.json()
        if not payload.get("ok"):
            return call_local_recall_engine()
        result = payload.get("result", {})
        recall_manager.warning_emitted = False
        return result.get("results", []), f"recall_api_{normalized_source}"
    except Exception as exc:
        if not recall_manager.warning_emitted:
            logging.warning("[recall] request failed: %s", exc)
            recall_manager.warning_emitted = True
        return call_local_recall_engine()


def local_search(df: pd.DataFrame, query: str, top_n: int = 3) -> list[dict[str, Any]]:
    if not query:
        return []

    headword_hits = df[0].astype(str).str.contains(query, na=False, regex=False)
    definition_hits = df[4].astype(str).str.contains(query, na=False, regex=False)
    matches = df[headword_hits | definition_hits].copy()
    if matches.empty:
        return []

    def score_row(row: pd.Series) -> float:
        shanghai = str(row[0]).strip()
        definition = str(row[4]).strip()
        score = 0.0
        if query in shanghai:
            score += 2.0
        if query in definition:
            score += 1.0
        return score - 0.01 * len(shanghai)

    matches["score"] = matches.apply(score_row, axis=1)
    matches = matches.sort_values("score", ascending=False).head(top_n)

    rows: list[dict[str, Any]] = []
    for idx, row in matches.iterrows():
        rows.append(
            {
                "id": int(idx),
                "shanghai": str(row[0]).strip(),
                "definition": str(row[4]).strip(),
                "wu_pinyin": str(row[5]).strip() if len(row) > 5 else "",
                "score": float(row["score"]),
            }
        )
    return rows


def sanitize_headword(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^[\[\u3010]+", "", value)
    value = re.sub(r"[\]\u3011]+\.\d+$", "", value)
    value = re.sub(r"[\]\u3011]+$", "", value)
    return value.strip()


def clean_cell_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() == "nan" else text


def find_headword_row(target: str) -> pd.Series | None:
    if df.empty:
        return None
    clean_target = sanitize_headword(target)
    headwords = df[0].astype(str).map(sanitize_headword)

    exact_matches = df[headwords == clean_target]
    if not exact_matches.empty:
        return exact_matches.iloc[0]

    contains_matches = df[headwords.str.contains(clean_target, na=False, regex=False)]
    if not contains_matches.empty:
        return contains_matches.iloc[0]

    return None


def find_shaoxing_row(target: str) -> pd.Series | None:
    if shaoxing_df.empty:
        return None
    clean_target = sanitize_headword(target)
    headwords = shaoxing_df[0].astype(str).map(sanitize_headword)

    exact_matches = shaoxing_df[headwords == clean_target]
    if not exact_matches.empty:
        return exact_matches.iloc[0]

    contains_matches = shaoxing_df[headwords.str.contains(clean_target, na=False, regex=False)]
    if not contains_matches.empty:
        return contains_matches.iloc[0]
    return None


def exact_shaoxing_hits(term: str, top_n: int = 2) -> list[dict[str, Any]]:
    query = _clean_definition_for_match(term)
    if not query:
        return []
    index = build_exact_shaoxing_index()
    return [dict(item) for item in index.get(query, [])[:top_n]]


def _row_pinyin(row: pd.Series) -> str:
    return clean_cell_text(row[5]) if len(row) > 5 else ""


def source_to_tts_model(source_id: str | None) -> str:
    return "shaoxing" if str(source_id or "").strip() == SHAOXING_SOURCE_ID else "shanghai"


def synthesize_dictionary_audio(
    *,
    source_id: str,
    headword: str,
    pinyin: str,
    output_path: Path,
) -> tuple[str, str]:
    model_name = source_to_tts_model(source_id)
    ensure_tts_model_loaded(model_name=model_name)
    synthesize(pinyin, str(output_path), model_name=model_name)
    return model_name, headword


def call_shaoxing_recall(
    query: str,
    variants: list[str] | None = None,
    top_k: int = 20,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    if not index_complete(SHAOXING_INDEX_DIR):
        return []
    result_rows, _source = call_recall_service(
        query=query,
        variants=variants,
        top_k=top_k,
        top_n=top_n,
        source="shaoxing",
    )
    rows: list[dict[str, Any]] = []
    for item in result_rows:
        headword = clean_cell_text(item.get("shanghai", ""))
        row = find_shaoxing_row(headword)
        rows.append(
            {
                "id": int(item.get("id", len(rows))),
                "shanghai": headword,
                "definition": clean_cell_text(item.get("definition", "")),
                "pinyin": _row_pinyin(row) if row is not None else "",
                "score": float(item.get("score", 0.0)),
                "source": SHAOXING_SOURCE_ID,
            }
        )
    return rows


def shaoxing_local_search(query: str, top_n: int = 3) -> list[dict[str, Any]]:
    if not query or shaoxing_df.empty:
        return []

    headword_hits = shaoxing_df[0].astype(str).str.contains(query, na=False, regex=False)
    definition_hits = shaoxing_df[4].astype(str).str.contains(query, na=False, regex=False)
    matches = shaoxing_df[headword_hits | definition_hits].copy()
    if matches.empty:
        return []

    def score_row(row: pd.Series) -> float:
        headword = clean_cell_text(row[0])
        definition = clean_cell_text(row[4])
        score = 0.0
        if query in headword:
            score += 2.0
        if query in definition:
            score += 1.0
        if _row_pinyin(row):
            score += 0.15
        return score - 0.01 * len(headword)

    matches["score"] = matches.apply(score_row, axis=1)
    matches = matches.sort_values("score", ascending=False).head(top_n)

    rows: list[dict[str, Any]] = []
    for idx, row in matches.iterrows():
        rows.append(
            {
                "id": int(idx),
                "shanghai": clean_cell_text(row[0]),
                "definition": clean_cell_text(row[4]),
                "pinyin": _row_pinyin(row),
                "score": float(row["score"]),
                "source": SHAOXING_SOURCE_ID,
            }
        )
    return rows


def build_reply(results: list[dict[str, Any]]) -> str:
    if not results:
        return "\u672a\u627e\u5230\u5339\u914d\u8bcd\u6761\u3002"

    lines = ["\u627e\u5230\u8fd9\u4e9b\u5019\u9009\uff1a<br><br>"]
    for item in results:
        raw_headword = str(item.get("shanghai", "")).strip()
        shanghai = sanitize_headword(raw_headword)
        definition = str(item.get("definition", "")).strip()
        source = str(item.get("source", PRIMARY_SOURCE_ID)).strip()
        pinyin = str(item.get("pinyin") or item.get("wu_pinyin") or "").strip()
        safe_headword = html.escape(shanghai, quote=True)
        safe_definition = html.escape(definition, quote=True)
        safe_pinyin = html.escape(pinyin, quote=True)
        safe_source = html.escape(source, quote=True)
        if shanghai:
            shanghai = f"\u3010{safe_headword}\u3011"
        badge = "\u4e0a\u6d77\u8bcd\u5178" if source != SHAOXING_SOURCE_ID else "\u7ecd\u5174\u8bcd\u5178"
        lines.append(f"<b>{shanghai}</b> <span class='result-badge'>[{badge}]</span><br>")
        if pinyin:
            lines.append(f"\u62fc\u97f3\uff1a{safe_pinyin}<br>")
        lines.append(f"\u91ca\u4e49\uff1a{safe_definition}<br>")
        if pinyin:
            lines.append(
                "<button type='button' class='voice-btn' "
                f"data-headword='{safe_headword}' data-source='{safe_source}'>\u25b6 \u751f\u6210\u8bed\u97f3</button><br><hr>"
            )
        else:
            lines.append("<span class='result-badge'>\u8be5\u8bcd\u6761\u6682\u65e0\u53ef\u7528\u8bfb\u97f3</span><br><hr>")
    return "".join(lines)


def build_text_only_reply(results: list[dict[str, Any]]) -> str:
    if not results:
        return "\u672a\u627e\u5230\u5339\u914d\u8bcd\u6761\u3002"

    lines: list[str] = []
    for item in results:
        shanghai = sanitize_headword(str(item.get("shanghai", "")).strip())
        definition = str(item.get("definition", "")).strip()
        source = str(item.get("source", PRIMARY_SOURCE_ID)).strip()
        pinyin = str(item.get("pinyin") or item.get("wu_pinyin") or "").strip()
        if shanghai:
            shanghai = f"\u3010{shanghai}\u3011"
        prefix = "[绍兴]" if source == SHAOXING_SOURCE_ID else "[上海]"
        line = f"{prefix} {shanghai}"
        if pinyin:
            line += f" ({pinyin})"
        line += f" - {definition}"
        lines.append(line)
    return "<br>".join(lines)


app = Flask(__name__)
app.secret_key = os.getenv("SHANGHAI_TTS_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    MAX_CONTENT_LENGTH=32 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
df = load_dataframe()
shaoxing_df = load_dataframe(SHAOXING_PROCESSED_PATH)
preprocessor: Any | None = None
reranker: SplitRecallReranker | None = None
rag_mt_translators: dict[str, QwenRAGMTTranslator] = {}
cleaner_core_extractor: CleanerCoreExtractor | None = None

# ── index rebuild state ───────────────────────────────────────────────────────
_rebuild_lock = threading.Lock()
_rebuild_status: dict[str, Any] = {"running": False, "progress": [], "error": None, "done": False}


@app.route("/")
def index():
    index_html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    return app.response_class(index_html, mimetype="text/html")


@app.route("/frontend/<path:filename>")
def frontend_asset(filename: str):
    return send_from_directory(str(FRONTEND_DIR), filename)


@app.after_request
def apply_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; "
        "connect-src 'self'; "
        "font-src 'self' data:; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'self'"
    )
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/rebuild_index", methods=["POST"])
def api_rebuild_index():
    """Trigger a full index rebuild in a background thread.

    POST /api/rebuild_index          — start rebuild (idempotent if already running)
    GET  /api/rebuild_index/status   — poll progress
    """
    global _rebuild_status
    with _rebuild_lock:
        if _rebuild_status["running"]:
            return jsonify({"ok": False, "error": "already_running", "status": _rebuild_status})
        _rebuild_status = {"running": True, "progress": ["开始重建索引…"], "error": None, "done": False}

    def _run() -> None:
        global local_recall_engine, exact_dictionary_index, df, _rebuild_status
        try:
            import sys as _sys
            _script_dir = str(REPO_ROOT / "app" / "recall" / "scripts")
            if _script_dir not in _sys.path:
                _sys.path.insert(0, _script_dir)
            from build_vector_index import run_build  # type: ignore[import]

            meta = json.loads((RECALL_INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
            # resolve model path relative to index_dir (matches how RecallEngine does it)
            _recall_dir = str(REPO_ROOT / "app" / "recall")
            if _recall_dir not in _sys.path:
                _sys.path.insert(0, _recall_dir)
            from model_utils import ensure_model_path  # type: ignore[import]
            model_path = ensure_model_path(str(meta["model_name_or_path"]), RECALL_INDEX_DIR)

            def _cb(msg: str) -> None:
                logging.info("[rebuild] %s", msg)
                with _rebuild_lock:
                    _rebuild_status["progress"].append(msg)

            run_build(
                dict_csv=str(DICT_PATH),
                out_dir=str(RECALL_INDEX_DIR),
                model_name_or_path=model_path,
                id_col="0",
                sh_col="1",
                def_col="4",
                header="infer",
                batch_size=128,
                max_length=128,
                text_mode="headword_definition",
                progress_cb=_cb,
            )

            # hot-reload in-process engine and df
            _cb("热重载本地召回引擎…")
            from recall.engine import RecallEngine
            local_recall_engine = RecallEngine(index_dir=RECALL_INDEX_DIR, ann="hnsw", ef_search=64)
            exact_dictionary_index = None  # force rebuild on next query
            df = load_dataframe()
            _cb("完成！")
            with _rebuild_lock:
                _rebuild_status["running"] = False
                _rebuild_status["done"] = True
        except Exception as exc:  # noqa: BLE001
            logging.error("[rebuild] failed: %s", exc, exc_info=True)
            with _rebuild_lock:
                _rebuild_status["running"] = False
                _rebuild_status["error"] = str(exc)

    threading.Thread(target=_run, daemon=True, name="index-rebuild").start()
    return jsonify({"ok": True, "message": "索引重建已启动，请轮询 /api/rebuild_index/status 查看进度"})


@app.route("/api/rebuild_index/status", methods=["GET"])
def api_rebuild_index_status():
    with _rebuild_lock:
        return jsonify({"ok": True, "status": dict(_rebuild_status)})


@app.route("/api/user/confinfo", methods=["GET"])
def api_user_confinfo():
    tts_status = tts_runtime_status()
    return jsonify(
        {
            "ok": True,
            "service": "shanghai-tts",
            "version": "local",
            "pipeline_mode": PIPELINE_MODE,
            "rag_mt_ready": rag_mt_ready(),
            "legacy_preprocessor_ready": legacy_preprocessor_ready(),
            "tts_ready": tts_ready(),
            "tts_status": tts_status,
            "shaoxing_index_ready": index_complete(SHAOXING_INDEX_DIR),
        }
    )


@app.route("/api/tts/status", methods=["GET"])
def api_tts_status():
    return jsonify({"ok": True, "tts": tts_runtime_status()})


@app.route("/api/tts/load", methods=["POST"])
def api_tts_load():
    payload = request.get_json(silent=True) or {}
    model_name = str(payload.get("model", "")).strip().lower()
    device = str(payload.get("device", "")).strip().lower() or "cpu"
    try:
        status = set_tts_model_device(model_name=model_name, device=device)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    return jsonify({"ok": True, "model": model_name, "status": status, "tts": tts_runtime_status()})


@app.route("/api/tts/direct_model", methods=["POST"])
def api_tts_direct_model():
    payload = request.get_json(silent=True) or {}
    model_name = str(payload.get("model", "")).strip().lower()
    try:
        active_name = set_tts_active_direct_model(model_name)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "active_direct_model": active_name, "tts": tts_runtime_status()})


@app.route("/api/tts/read_headword", methods=["POST"])
def api_tts_read_headword():
    started = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    source_id = str(payload.get("source", PRIMARY_SOURCE_ID)).strip() or PRIMARY_SOURCE_ID
    headword = str(payload.get("headword", "")).strip()
    if not headword:
        return jsonify({"text": "缺少词条名。"}), 400

    if source_id == SHAOXING_SOURCE_ID:
        row = find_shaoxing_row(headword)
        if row is None:
            return jsonify({"text": f"未找到绍兴词条：{headword}"}), 404
        pinyin = _row_pinyin(row)
        definition = clean_cell_text(row[4]) or "暂缺"
        display_headword = clean_cell_text(row[0]) or headword
    else:
        row = find_headword_row(headword)
        if row is None:
            return jsonify({"text": f"未找到上海词条：{headword}"}), 404
        pinyin = clean_cell_text(row[5]) if len(row) > 5 else ""
        definition = clean_cell_text(row[4]) or "暂缺"
        display_headword = clean_cell_text(row[0]) or headword

    if not pinyin:
        return jsonify({"text": f"匹配到：{display_headword}<br>释义：{definition}<br>但该词条暂无可用读音。"})

    model_name = source_to_tts_model(source_id)
    if not tts_model_ready(model_name):
        return jsonify({"text": f"{model_name} TTS 模型未就绪。"}), 503

    unique_fn = f"{uuid.uuid4().hex[:8]}.wav"
    try:
        synthesize_dictionary_audio(
            source_id=source_id,
            headword=display_headword,
            pinyin=pinyin,
            output_path=STATIC_DIR / unique_fn,
        )
    except RuntimeError as exc:
        return jsonify({"text": f"{display_headword}<br>TTS 加载失败：{exc}"}), 503
    session["last_audio"] = {"filename": unique_fn, "word": display_headword, "source": source_id, "model": model_name}
    return jsonify(
        {
            "ok": True,
            "audio": f"/static/{unique_fn}",
            "model": model_name,
            "source": source_id,
            "text": f"匹配到：{display_headword}<br>释义：{definition}",
            "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 1)},
        }
    )


@app.route("/download/<filename>")
def download_file(filename: str):
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".wav"):
        return jsonify({"text": "Invalid file name"}), 400
    return send_from_directory(str(STATIC_DIR), safe_name, as_attachment=True)


@app.errorhandler(404)
def handle_404(_e):
    return jsonify({"text": "Not Found"}), 404


@app.errorhandler(Exception)
def handle_exception(exc):
    if isinstance(exc, HTTPException):
        return exc
    logging.error("Unhandled exception:\n%s", traceback.format_exc())
    return jsonify({"text": f"Server error: {str(exc)}"}), 500


def ensure_preprocessor() -> Any:
    global preprocessor
    if preprocessor is None:
        backend = PREPROCESSOR_BACKEND.lower()
        if backend == "llama_cpp":
            if LLAMA_CPP_SERVER_PATH is None:
                raise FileNotFoundError("llm.llama_cpp_server_path is not configured")
            if GGUF_MODEL_PATH is None:
                raise FileNotFoundError("llm.gguf_model_path is not configured")
            preprocessor = LlamaCppQueryPreprocessor(
                config=LlamaCppConfig(
                    server_path=LLAMA_CPP_SERVER_PATH,
                    gguf_model_path=GGUF_MODEL_PATH,
                    host=LLAMA_CPP_HOST,
                    port=LLAMA_CPP_PORT,
                    ctx_size=LLAMA_CPP_CTX_SIZE,
                    n_gpu_layers=LLAMA_CPP_N_GPU_LAYERS,
                    threads=LLAMA_CPP_THREADS,
                    batch_size=LLAMA_CPP_BATCH_SIZE,
                    ubatch_size=LLAMA_CPP_UBATCH_SIZE,
                    timeout_seconds=LLAMA_CPP_TIMEOUT_SECONDS,
                    auto_start=LLAMA_CPP_AUTO_START,
                    flash_attention=LLAMA_CPP_FLASH_ATTENTION,
                ),
                base_model_path=BASE_MODEL_PATH,
                system_prompt=ZH_PROMPT,
            )
        elif backend in {"qwen", "qwen_lora", "llm", "transformers"}:
            preprocessor = LocalQueryPreprocessor(
                checkpoint_path=CHECKPOINT_PATH,
                base_model_path=BASE_MODEL_PATH,
                load_in_4bit=LOAD_IN_4BIT,
                compile_enabled=COMPILE_ENABLED,
                compile_mode=COMPILE_MODE,
                compile_cache_dir=COMPILE_CACHE_DIR,
            )
        else:
            raise ValueError(f"unknown query preprocessor backend: {PREPROCESSOR_BACKEND}")
    return preprocessor


def legacy_preprocessor_ready() -> bool:
    backend = PREPROCESSOR_BACKEND.lower()
    if backend == "llama_cpp":
        return LLAMA_CPP_SERVER_PATH is not None and GGUF_MODEL_PATH is not None and GGUF_MODEL_PATH.exists()
    if backend in {"qwen", "qwen_lora", "llm", "transformers"}:
        return CHECKPOINT_PATH.exists() and BASE_MODEL_PATH.exists()
    return False


def ensure_rag_mt_translator(model_size: str | None = None) -> QwenRAGMTTranslator:
    selected_size = (model_size or RAG_MT_MODEL_SIZE or "2b").strip().lower()
    if selected_size in rag_mt_translators:
        return rag_mt_translators[selected_size]

    model_configs = RAG_MT_CONFIG.get("models", {})
    model_config = model_configs.get(selected_size)
    if model_config is None:
        available = ", ".join(sorted(model_configs)) or "<none>"
        raise ValueError(f"unknown rag_mt model_size={selected_size}; available={available}")

    model_path = Path(model_config["model_path"])
    adapter_value = model_config.get("adapter_path")
    adapter_path = Path(adapter_value) if adapter_value else None
    translator = QwenRAGMTTranslator(
        model_path=model_path,
        adapter_path=adapter_path,
        load_in_4bit=bool(model_config.get("load_in_4bit", RAG_MT_CONFIG.get("load_in_4bit", False))),
        max_new_tokens=int(RAG_MT_CONFIG.get("max_new_tokens", 64)),
        temperature=float(RAG_MT_CONFIG.get("temperature", 0.0)),
    )
    rag_mt_translators[selected_size] = translator
    logging.info("[rag_mt] loaded model_size=%s model=%s adapter=%s", selected_size, model_path, adapter_path)
    return translator


def rag_mt_ready(model_size: str | None = None) -> bool:
    selected_size = (model_size or RAG_MT_MODEL_SIZE or "2b").strip().lower()
    model_configs = RAG_MT_CONFIG.get("models", {})
    model_config = model_configs.get(selected_size)
    if not isinstance(model_config, dict):
        return False
    model_path = model_config.get("model_path")
    adapter_path = model_config.get("adapter_path")
    return bool(model_path and Path(model_path).exists() and adapter_path and Path(adapter_path).exists())


def ensure_all_rag_mt_translators() -> None:
    model_configs = RAG_MT_CONFIG.get("models", {})
    sizes = sorted(model_configs) if PRELOAD_ALL_RAG_MT_MODELS else [RAG_MT_MODEL_SIZE]
    for size in sizes:
        ensure_rag_mt_translator(model_size=size)


def rag_mt_model_paths() -> set[Path]:
    paths: set[Path] = set()
    for model_config in RAG_MT_CONFIG.get("models", {}).values():
        if isinstance(model_config, dict) and model_config.get("model_path"):
            paths.add(Path(model_config["model_path"]).resolve())
    return paths


def should_preload_legacy_preprocessor() -> bool:
    if not PRELOAD_LEGACY_PREPROCESSOR:
        return False
    if PIPELINE_MODE not in {"rag_mt", "rag-mt", "mt"}:
        return True
    if BASE_MODEL_PATH.resolve() in rag_mt_model_paths():
        logging.info(
            "[preload] skip legacy query preprocessor to avoid loading duplicate base model: %s",
            BASE_MODEL_PATH.resolve(),
        )
        return False
    return True


def ensure_cleaner_core_extractor() -> CleanerCoreExtractor | None:
    global cleaner_core_extractor
    if not bool(CLEANER_CONFIG.get("enabled", False)):
        return None
    checkpoint_dir = CLEANER_CONFIG.get("checkpoint_dir")
    if not checkpoint_dir:
        return None
    path = Path(checkpoint_dir)
    if not (path / "model.pt").exists():
        logging.warning("[cleaner] checkpoint not found: %s", path)
        return None
    if cleaner_core_extractor is None:
        cleaner_core_extractor = CleanerCoreExtractor(path)
        logging.info("[cleaner] loaded core extractor: %s", path)
    return cleaner_core_extractor


def preload_runtime_models() -> None:
    if not PRELOAD_MODELS:
        logging.info("[preload] disabled by WUU_PRELOAD_MODELS=0")
        return

    logging.info("[preload] loading runtime models into memory")
    if not TEXT_ONLY_MODE:
        if recall_manager.ensure_running():
            logging.info("[preload] recall service is ready")
        else:
            logging.warning("[preload] recall service unavailable; app will fallback to local search")

    build_exact_dictionary_index()
    logging.info("[preload] exact dictionary index is ready")

    cleaner = ensure_cleaner_core_extractor()
    if cleaner is not None:
        logging.info("[preload] cleaner core is ready")

    if RERANKER_ENABLED:
        ensure_reranker()
        logging.info("[preload] reranker is ready")

    if PIPELINE_MODE in {"rag_mt", "rag-mt", "mt"}:
        try:
            ensure_all_rag_mt_translators()
            logging.info("[preload] rag-mt translators are ready: %s", ", ".join(sorted(rag_mt_translators)))
        except Exception as exc:
            logging.warning("[preload] rag-mt translators unavailable: %s", exc)
    elif should_preload_legacy_preprocessor():
        try:
            ensure_preprocessor()
            logging.info("[preload] legacy query preprocessor is ready")
        except Exception as exc:
            logging.warning("[preload] legacy query preprocessor unavailable: %s", exc)

    if should_preload_legacy_preprocessor() and PIPELINE_MODE in {"rag_mt", "rag-mt", "mt"}:
        try:
            ensure_preprocessor()
            logging.info("[preload] legacy query preprocessor is ready")
        except Exception as exc:
            logging.warning("[preload] legacy query preprocessor unavailable: %s", exc)

    logging.info("[preload] skip auto-loading tts models; keep runtime state unloaded until requested")


def start_preload_thread() -> None:
    if not PRELOAD_MODELS:
        logging.info("[preload] disabled by WUU_PRELOAD_MODELS=0")
        return
    threading.Thread(target=preload_runtime_models, daemon=True, name="runtime-preload").start()


def bootstrap_dictionary_assets() -> None:
    global shaoxing_df
    try:
        processed_path = ensure_shaoxing_processed_csv()
        if processed_path is not None:
            shaoxing_df = load_dataframe(processed_path)
            logging.info("[dict] shaoxing processed rows=%s", len(shaoxing_df))
    except Exception as exc:
        logging.warning("[dict] failed to prepare shaoxing csv: %s", exc)

    def _build() -> None:
        if DICT_PATH is not None and DICT_PATH.exists() and not index_complete(RECALL_INDEX_DIR):
            try:
                ensure_dense_index(DICT_PATH, RECALL_INDEX_DIR, source_label=PRIMARY_SOURCE_ID)
            except Exception as exc:
                logging.warning("[dict] failed to auto-build primary recall index: %s", exc)
        if SHAOXING_PROCESSED_PATH is not None and SHAOXING_PROCESSED_PATH.exists() and not index_complete(SHAOXING_INDEX_DIR):
            try:
                ensure_dense_index(SHAOXING_PROCESSED_PATH, SHAOXING_INDEX_DIR, source_label=SHAOXING_SOURCE_ID)
            except Exception as exc:
                logging.warning("[dict] failed to auto-build shaoxing recall index: %s", exc)

    threading.Thread(target=_build, daemon=True, name="dict-bootstrap").start()


def clean_query_for_rag_mt(user_msg: str) -> dict[str, Any]:
    extractor = ensure_cleaner_core_extractor()
    if extractor is not None:
        try:
            return extractor.predict(user_msg).as_parsed()
        except Exception as exc:
            logging.warning("[cleaner] model inference failed, fallback to rules: %s", exc)
    return clean_query(user_msg, max_keywords=int(RAG_MT_CONFIG.get("max_keywords", 8))).as_parsed()


def _clean_definition_for_match(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"〈[^〉]+〉", "", value)
    value = value.replace("﹝俚﹞", "").replace("﹝旧﹞", "")
    value = re.split(r"[：:；;。丨◇]", value)[0]
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", value)
    return value


def build_exact_dictionary_index() -> dict[str, list[dict[str, Any]]]:
    global exact_dictionary_index
    if exact_dictionary_index is not None:
        return exact_dictionary_index

    index: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for idx, row in df.iterrows():
        shanghai = sanitize_headword(str(row[0]).strip())
        definition_raw = str(row[4]).strip()
        definition = _clean_definition_for_match(definition_raw)
        if not shanghai:
            continue
        row_payload = {
            "id": int(idx),
            "shanghai": shanghai,
            "definition": definition_raw,
            "wu_pinyin": str(row[5]).strip() if len(row) > 5 else "",
            "source": "exact_dictionary",
        }
        for key, base_score in ((shanghai, 120.0), (definition, 100.0)):
            if not key:
                continue
            marker = (key, shanghai)
            if marker in seen:
                continue
            seen.add(marker)
            item = dict(row_payload)
            item["score"] = base_score - 0.01 * len(shanghai)
            index.setdefault(key, []).append(item)

    for key, values in index.items():
        values.sort(key=lambda item: (float(item["score"]), -len(str(item["shanghai"]))), reverse=True)
    exact_dictionary_index = index
    return index


def build_exact_shaoxing_index() -> dict[str, list[dict[str, Any]]]:
    global shaoxing_exact_index
    if shaoxing_exact_index is not None:
        return shaoxing_exact_index

    index: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    if shaoxing_df.empty:
        shaoxing_exact_index = index
        return index

    for _, row in shaoxing_df.iterrows():
        headword = clean_cell_text(row[0])
        definition_raw = clean_cell_text(row[4])
        definition = _clean_definition_for_match(definition_raw)
        pinyin = _row_pinyin(row)
        if not headword:
            continue
        row_payload = {
            "shanghai": headword,
            "definition": definition_raw,
            "pinyin": pinyin,
            "source": SHAOXING_SOURCE_ID,
        }
        for key, base_score in ((headword, 120.0), (definition, 100.0)):
            clean_key = _clean_definition_for_match(key)
            if not clean_key:
                continue
            marker = (clean_key, headword)
            if marker in seen:
                continue
            seen.add(marker)
            item = dict(row_payload)
            item["score"] = base_score - 0.01 * len(headword)
            index.setdefault(clean_key, []).append(item)

    for key, values in index.items():
        values.sort(key=lambda item: (float(item["score"]), -len(str(item["shanghai"]))), reverse=True)
    shaoxing_exact_index = index
    return index


def exact_dictionary_hits(term: str, top_n: int = 3) -> list[dict[str, Any]]:
    query = _clean_definition_for_match(term)
    if not query:
        return []
    variants = [query]
    for suffix in ("了", "吗", "呢", "啊", "呀"):
        if query.endswith(suffix) and len(query) > len(suffix) + 1:
            variants.append(query[: -len(suffix)])

    index = build_exact_dictionary_index()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for variant in variants:
        for item in index.get(variant, []):
            shanghai = sanitize_headword(str(item.get("shanghai", "")))
            if not shanghai or shanghai in seen:
                continue
            seen.add(shanghai)
            rows.append(dict(item))
    rows.sort(key=lambda item: (float(item["score"]), -len(str(item["shanghai"]))), reverse=True)
    return rows[:top_n]


@lru_cache(maxsize=256)
def cached_rule_parse(user_msg: str) -> dict[str, Any]:
    return clean_query_for_rag_mt(user_msg)


def clone_payload(value: Any) -> Any:
    return copy.deepcopy(value)


def collect_rag_mt_keyword_hits(parsed: dict[str, Any]) -> tuple[list[tuple[str, list[dict[str, Any]]]], str]:
    keywords = parsed.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    core_text = str(parsed.get("core_text", "")).strip()
    terms: list[str] = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    if core_text and len(core_text) <= 4 and core_text not in terms:
        terms.insert(0, core_text)

    hits: list[tuple[str, list[dict[str, Any]]]] = []
    sources: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term or term in seen:
            continue
        if core_text and term == core_text and len(terms) > 1:
            continue
        seen.add(term)
        exact_results = exact_dictionary_hits(term, top_n=3)
        results, source = (exact_results, "exact_dictionary") if exact_results else call_recall_service(term, variants=[term], top_k=20, top_n=5)
        if results and source != "exact_dictionary":
            results = rerank_recall_results(term, {"core_text": term, "keywords": [term], "segments": [term]}, results)[:3]
        if not results:
            results = local_search(df, term, top_n=3)
            if results:
                source = "local_search"
        if results:
            hits.append((term, results))
            sources.append(source or "unknown")
    return hits, ",".join(sorted(set(sources)))


def run_rag_mt_pipeline(user_msg: str, model_size: str | None = None) -> dict[str, Any]:
    timings_ms: dict[str, float] = {}
    total_started = time.perf_counter()

    parse_started = time.perf_counter()
    parsed = cached_rule_parse(user_msg)
    timings_ms["parse"] = round((time.perf_counter() - parse_started) * 1000, 1)

    context_started = time.perf_counter()
    keyword_hits, matched_source = collect_rag_mt_keyword_hits(parsed)
    context = build_lexicon_context(
        keyword_hits,
        max_pairs=int(RAG_MT_CONFIG.get("max_context_pairs", 12)),
        max_candidates_per_keyword=int(RAG_MT_CONFIG.get("max_candidates_per_keyword", 2)),
    )
    timings_ms["context_lookup"] = round((time.perf_counter() - context_started) * 1000, 1)

    translate_started = time.perf_counter()
    translator = ensure_rag_mt_translator(model_size=model_size)
    selected_model_size = (model_size or RAG_MT_MODEL_SIZE or "2b").strip().lower()
    keywords = [str(item) for item in parsed.get("keywords", []) if str(item).strip()]
    translation = translator.translate(
        parsed["core_text"],
        context,
        raw_query=user_msg,
        keywords=keywords,
    )
    timings_ms["translate"] = round((time.perf_counter() - translate_started) * 1000, 1)
    timings_ms["total"] = round((time.perf_counter() - total_started) * 1000, 1)
    logging.info("[rag_mt] parsed=%s", json.dumps(parsed, ensure_ascii=False))
    logging.info("[rag_mt] context=%s", context)
    logging.info("[rag_mt] translation=%s", translation)
    logging.info("[rag_mt] timings_ms=%s", json.dumps(timings_ms, ensure_ascii=False))
    return {
        "text": translation or parsed["core_text"],
        "mode": "rag_mt",
        "translation": translation,
        "core_text": parsed["core_text"],
        "keywords": parsed["keywords"],
        "segments": parsed.get("segments", []),
        "type": parsed.get("type", ""),
        "lexicon_context": context,
        "matched_source": matched_source or "none",
        "model_size": selected_model_size,
        "timings_ms": timings_ms,
        "recall": [
            {
                "keyword": keyword,
                "hits": summarize_results(hits),
            }
            for keyword, hits in keyword_hits
        ],
    }


def run_dictionary_lookup_pipeline(
    user_msg: str,
    parsed: dict[str, Any] | None = None,
    use_legacy_preprocessor: bool = False,
) -> dict[str, Any]:
    timings_ms: dict[str, float] = {}
    total_started = time.perf_counter()
    if parsed is None:
        parse_started = time.perf_counter()
        if use_legacy_preprocessor:
            try:
                parsed = ensure_preprocessor().preprocess(user_msg)
            except Exception as exc:
                logging.warning("[lookup] legacy preprocessor unavailable, fallback to rule cleaner: %s", exc)
                parsed = cached_rule_parse(user_msg)
        else:
            parsed = cached_rule_parse(user_msg)
        timings_ms["parse"] = round((time.perf_counter() - parse_started) * 1000, 1)
    else:
        timings_ms["parse"] = 0.0

    terms_started = time.perf_counter()
    search_terms = build_search_terms(user_msg, parsed)
    timings_ms["build_terms"] = round((time.perf_counter() - terms_started) * 1000, 1)
    logging.info("[lookup] parsed=%s", json.dumps(parsed, ensure_ascii=False))
    logging.info("[lookup] search_terms=%s", json.dumps(search_terms, ensure_ascii=False))

    exact_started = time.perf_counter()
    exact_primary: list[dict[str, Any]] = []
    exact_shaoxing: list[dict[str, Any]] = []
    for term in search_terms[:4]:
        if not exact_primary:
            exact_primary = exact_dictionary_hits(term, top_n=3)
        if not exact_shaoxing:
            exact_shaoxing = exact_shaoxing_hits(term, top_n=2)
        if exact_primary and exact_shaoxing:
            break
    timings_ms["exact_hits"] = round((time.perf_counter() - exact_started) * 1000, 1)

    if exact_primary or exact_shaoxing:
        results = list(exact_primary)
        seen = {f"{item.get('source', PRIMARY_SOURCE_ID)}::{sanitize_headword(str(item.get('shanghai', '')))}" for item in results}
        for item in exact_shaoxing:
            key = f"{item.get('source', SHAOXING_SOURCE_ID)}::{sanitize_headword(str(item.get('shanghai', '')))}"
            if key not in seen:
                seen.add(key)
                results.append(item)
        results = results[:5]
        timings_ms["total"] = round((time.perf_counter() - total_started) * 1000, 1)
        summary = summarize_results(results)
        logging.info(
            "[lookup] matched_source=%s matched_term=%s results=%s timings_ms=%s",
            "exact_dictionary" + (",shaoxing_xlsx" if exact_shaoxing else ""),
            search_terms[0] if search_terms else user_msg,
            json.dumps(summary, ensure_ascii=False),
            json.dumps(timings_ms, ensure_ascii=False),
        )
        return {
            "text": build_text_only_reply(results) if TEXT_ONLY_MODE else build_reply(results),
            "text_only": build_text_only_reply(results),
            "parsed": clone_payload(parsed),
            "search_terms": list(search_terms),
            "matched_source": "exact_dictionary" + (",shaoxing_xlsx" if exact_shaoxing else ""),
            "matched_term": search_terms[0] if search_terms else user_msg,
            "results": summary,
            "timings_ms": timings_ms,
        }

    recall_started = time.perf_counter()
    primary_results, matched_source = call_recall_service(user_msg, variants=search_terms, top_k=20, top_n=10)
    primary_results = rerank_recall_results(user_msg, parsed, primary_results)[:3]
    timings_ms["shanghai_recall"] = round((time.perf_counter() - recall_started) * 1000, 1)
    matched_term = user_msg
    if not primary_results:
        local_started = time.perf_counter()
        for term in search_terms:
            primary_results = local_search(df, term, top_n=3)
            if primary_results:
                matched_term = term
                matched_source = "local_search"
                break
        timings_ms["shanghai_local_fallback"] = round((time.perf_counter() - local_started) * 1000, 1)
    else:
        timings_ms["shanghai_local_fallback"] = 0.0

    shaoxing_results: list[dict[str, Any]] = []
    shaoxing_started = time.perf_counter()
    for term in search_terms:
        shaoxing_results = call_shaoxing_recall(term, variants=[term], top_k=20, top_n=3)
        if not shaoxing_results:
            shaoxing_results = shaoxing_local_search(term, top_n=2)
        if shaoxing_results:
            break
    timings_ms["shaoxing_lookup"] = round((time.perf_counter() - shaoxing_started) * 1000, 1)

    results = list(primary_results)
    seen = {f"{item.get('source', PRIMARY_SOURCE_ID)}::{sanitize_headword(str(item.get('shanghai', '')))}" for item in results}
    for item in shaoxing_results:
        key = f"{item.get('source', SHAOXING_SOURCE_ID)}::{sanitize_headword(str(item.get('shanghai', '')))}"
        if key not in seen:
            seen.add(key)
            results.append(item)
    results = results[:5]

    source_parts = [matched_source] if matched_source else []
    if shaoxing_results:
        source_parts.append(SHAOXING_SOURCE_ID)

    summary = summarize_results(results)
    timings_ms["total"] = round((time.perf_counter() - total_started) * 1000, 1)
    logging.info(
        "[lookup] matched_source=%s matched_term=%s results=%s timings_ms=%s",
        ",".join(source_parts) or "none",
        matched_term,
        json.dumps(summary, ensure_ascii=False),
        json.dumps(timings_ms, ensure_ascii=False),
    )
    return {
        "text": build_text_only_reply(results) if TEXT_ONLY_MODE else build_reply(results),
        "text_only": build_text_only_reply(results),
        "parsed": clone_payload(parsed),
        "search_terms": list(search_terms),
        "matched_source": ",".join(source_parts) or "none",
        "matched_term": matched_term,
        "results": summary,
        "timings_ms": timings_ms,
    }


def build_combined_rag_mt_reply(rag_result: dict[str, Any], lookup_result: dict[str, Any]) -> str:
    translation = str(rag_result.get("translation") or rag_result.get("text") or "").strip()
    lookup_text = str(lookup_result.get("text") or "").strip()
    parts = [
        f"翻译结果：{translation or '未生成翻译。'}",
        "原词典命中：",
        lookup_text or "未找到匹配词条。",
    ]
    return "<br>".join(parts)


def run_combined_rag_mt_pipeline(user_msg: str, model_size: str | None = None) -> dict[str, Any]:
    total_started = time.perf_counter()
    rag_result = run_rag_mt_pipeline(user_msg, model_size=model_size)
    lookup_parsed = {
        "core_text": rag_result.get("core_text", ""),
        "keywords": rag_result.get("keywords", []),
        "segments": rag_result.get("segments", []),
        "type": rag_result.get("type", ""),
    }
    lookup_result = run_dictionary_lookup_pipeline(user_msg, parsed=lookup_parsed)
    timings_ms = {
        "rag_mt_total": float(rag_result.get("timings_ms", {}).get("total", 0.0)),
        "lookup_total": float(lookup_result.get("timings_ms", {}).get("total", 0.0)),
        "total": round((time.perf_counter() - total_started) * 1000, 1),
    }
    return {
        "text": build_combined_rag_mt_reply(rag_result, lookup_result),
        "mode": "rag_mt_with_dictionary",
        "translation": rag_result.get("translation", ""),
        "dictionary_text": lookup_result.get("text", ""),
        "translation_result": rag_result,
        "dictionary_result": lookup_result,
        "timings_ms": timings_ms,
    }


def _extract_substrings(text: str, min_len: int = 2, max_len: int = 6) -> list[str]:
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < min_len:
        return []
    parts: list[str] = []
    for length in range(min(len(cleaned), max_len), min_len - 1, -1):
        for start in range(len(cleaned) - length + 1):
            segment = cleaned[start : start + length]
            if segment not in parts:
                parts.append(segment)
    return parts


def build_search_terms(user_msg: str, parsed: dict[str, Any]) -> list[str]:
    stripped = normalize_query(user_msg)
    core_text = str(parsed.get("core_text", "")).strip()
    keywords = parsed.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    segments = parsed.get("segments", [])
    if not isinstance(segments, list):
        segments = []

    original_substrings = _extract_substrings(stripped)

    candidates = [stripped]
    if core_text and core_text != stripped:
        candidates.append(core_text)
    for seg in segments:
        seg = str(seg).strip()
        if seg and seg not in candidates:
            candidates.append(seg)
    for kw in keywords:
        kw = str(kw).strip()
        if kw and kw not in candidates:
            candidates.append(kw)
    for sub in original_substrings[:8]:
        if sub not in candidates:
            candidates.append(sub)
    candidates.extend(expand_search_terms(candidates))

    seen: set[str] = set()
    terms: list[str] = []
    for term in candidates:
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def expand_search_terms(terms: list[str]) -> list[str]:
    expanded: list[str] = []
    for term in terms:
        normalized = str(term).strip()
        if not normalized:
            continue
        praise_match = re.search(r"(?:夸人|夸|称赞|赞)([\u4e00-\u9fff]{1,3})$", normalized)
        if praise_match:
            attribute = praise_match.group(1).strip()
            if attribute and attribute not in {"人", "别人", "对方"}:
                expanded.append(attribute)
    return expanded


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for item in results:
        summary.append(
            {
                "shanghai": str(item.get("shanghai", "")).strip(),
                "definition": str(item.get("definition", "")).strip(),
                "wu_pinyin": str(item.get("wu_pinyin", "")).strip(),
                "pinyin": str(item.get("pinyin", "")).strip(),
                "source": str(item.get("source", PRIMARY_SOURCE_ID)).strip(),
            }
        )
    return summary


def ensure_reranker() -> SplitRecallReranker | None:
    global reranker
    if not RERANKER_ENABLED or RERANKER_CHECKPOINT_DIR is None:
        return None
    if reranker is None:
        reranker = SplitRecallReranker(Path(RERANKER_CHECKPOINT_DIR))
    return reranker


def _candidate_terms_from_parsed(parsed: dict[str, Any], user_msg: str) -> list[str]:
    terms: list[str] = []
    core_text = str(parsed.get("core_text", "")).strip()
    if core_text:
        terms.append(core_text)
    for field in ("segments", "keywords"):
        values = parsed.get(field, [])
        if isinstance(values, list):
            for value in values:
                term = str(value).strip()
                if term:
                    terms.append(term)
    normalized = normalize_query(user_msg)
    if normalized:
        terms.append(normalized)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def rerank_recall_results(user_msg: str, parsed: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return results
    learned_reranker = ensure_reranker()
    if learned_reranker is not None:
        try:
            return learned_reranker.rerank(user_msg, parsed, results)
        except Exception as exc:
            logging.warning("[rerank] learned reranker failed, fallback to heuristic: %s", exc)

    candidate_terms = _candidate_terms_from_parsed(parsed, user_msg)
    query_type = str(parsed.get("type", "")).strip()

    reranked: list[dict[str, Any]] = []
    for rank, item in enumerate(results):
        shanghai = sanitize_headword(str(item.get("shanghai", "")).strip())
        definition = normalize_query(str(item.get("definition", "")).strip())
        base_score = float(item.get("score", 0.0))

        score = base_score
        overlap_bonus = 0.0
        for term in candidate_terms:
            if not term:
                continue
            term_bonus = 0.0
            if term == shanghai:
                term_bonus += 4.0
            elif term in shanghai:
                term_bonus += 2.5
            if term and term in definition:
                term_bonus += 1.5
            overlap_bonus += term_bonus

        score += overlap_bonus

        # Phrase-like queries are more likely to map to longer Shanghai expressions.
        if query_type == "动作短语":
            score += min(len(shanghai), 6) * 0.08
        elif query_type == "词项":
            score -= max(len(shanghai) - 4, 0) * 0.03

        # Prefer candidates that preserve split chunks from the original query.
        segments = parsed.get("segments", [])
        if isinstance(segments, list):
            matched_segments = sum(1 for seg in segments if str(seg).strip() and str(seg).strip() in shanghai)
            score += matched_segments * 0.6

        reranked_item = dict(item)
        reranked_item["base_score"] = base_score
        reranked_item["rerank_score"] = round(score, 4)
        reranked_item["rerank_overlap_bonus"] = round(overlap_bonus, 4)
        reranked_item["retrieval_rank"] = rank + 1
        reranked.append(reranked_item)

    reranked.sort(
        key=lambda row: (
            float(row.get("rerank_score", row.get("score", 0.0))),
            float(row.get("score", 0.0)),
        ),
        reverse=True,
    )
    return reranked


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_msg = str(payload.get("message", "")).strip()
    if not user_msg:
        return jsonify({"text": "\u8bf7\u8f93\u5165\u5185\u5bb9\u3002"})

    logging.info("[%s] user: %s", datetime.now().strftime("%H:%M:%S"), user_msg)
    last_audio = session.get("last_audio", {})

    if "\u518d\u8bfb\u4e00\u904d" in user_msg:
        if last_audio.get("filename"):
            return jsonify({"text": "\u597d\u7684\u3002", "audio": f"/static/{last_audio['filename']}"})
        return jsonify({"text": "\u8fd8\u6ca1\u6709\u53ef\u91cd\u64ad\u7684\u8bed\u97f3\u3002"})

    if "\u4e0b\u8f7d" in user_msg:
        if last_audio.get("filename"):
            return jsonify(
                {
                    "text": (
                        f"<a href='/download/{last_audio['filename']}' target='_blank'>"
                        "\u70b9\u51fb\u4e0b\u8f7d\u97f3\u9891</a>"
                    )
                }
            )
        return jsonify({"text": "\u8fd8\u6ca1\u6709\u53ef\u4e0b\u8f7d\u7684\u97f3\u9891\u3002"})

    if re.fullmatch(r"[a-zA-Z0-9\s]+", user_msg):
        started = time.perf_counter()
        if TEXT_ONLY_MODE:
            return jsonify({"text": f"text-only mode: {user_msg}"})
        model_name = active_direct_tts_model()
        if not tts_model_ready(model_name):
            return jsonify({"text": f"{model_name} TTS 模型未就绪，缺少 config 或 checkpoint。"})
        logging.info("[tts] direct pinyin input: %s", user_msg)
        unique_fn = f"{uuid.uuid4().hex[:8]}.wav"
        try:
            ensure_tts_model_loaded(model_name=model_name)
            synthesize(user_msg, str(STATIC_DIR / unique_fn), model_name=model_name)
        except RuntimeError as exc:
            return jsonify({"text": f"{model_name} TTS 加载失败：{exc}"})
        session["last_audio"] = {"filename": unique_fn, "word": "custom_pinyin", "model": model_name, "source": "direct"}
        return jsonify(
            {
                "text": f"\u6b63\u5728\u6717\u8bfb\uff1a{user_msg}<br><span style='color:#64748b;'>模型：{model_name}</span>",
                "audio": f"/static/{unique_fn}",
                "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 1)},
            }
        )

    bracket_match = re.search(r"[\u3010\[](.+?)[\u3011\]]", user_msg)
    if bracket_match:
        started = time.perf_counter()
        target = bracket_match.group(1).strip()
        logging.info("[tts] bracket target: %s", target)
        row = find_headword_row(target)
        if row is not None:
            if TEXT_ONLY_MODE:
                return jsonify({"text": f"\u5339\u914d\u5230\uff1a{row[0]}<br>\u91ca\u4e49\uff1a{row[4]}"})

            wu_pinyin = str(row[5]).strip() if len(row) > 5 else ""
            logging.info(
                "[tts] matched headword=%s definition=%s wu_pinyin=%s",
                str(row[0]).strip(),
                str(row[4]).strip(),
                wu_pinyin,
            )
            if not wu_pinyin:
                return jsonify({"text": f"\u5339\u914d\u5230\uff1a{row[0]}<br>\u91ca\u4e49\uff1a{row[4]}<br>\u4f46\u8be5\u8bcd\u6761\u6682\u65e0\u53ef\u7528\u8bfb\u97f3\u3002"})
            if not tts_model_ready("shanghai"):
                return jsonify({"text": f"\u5339\u914d\u5230\uff1a{row[0]}<br>\u91ca\u4e49\uff1a{row[4]}<br>上海话 TTS 模型未就绪，暂时不能朗读。"})

            unique_fn = f"{uuid.uuid4().hex[:8]}.wav"
            try:
                ensure_tts_model_loaded(model_name="shanghai")
                synthesize(wu_pinyin, str(STATIC_DIR / unique_fn), model_name="shanghai")
            except RuntimeError as exc:
                return jsonify({"text": f"\u5339\u914d\u5230\uff1a{row[0]}<br>\u91ca\u4e49\uff1a{row[4]}<br>TTS 加载失败：{exc}"})
            session["last_audio"] = {"filename": unique_fn, "word": str(row[0]), "model": "shanghai", "source": PRIMARY_SOURCE_ID}
            return jsonify(
                {
                    "text": f"\u5339\u914d\u5230\uff1a{row[0]}<br>\u91ca\u4e49\uff1a{row[4]}<br><span style='color:#64748b;'>模型：shanghai</span>",
                    "audio": f"/static/{unique_fn}",
                    "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 1)},
                }
            )
        shaoxing_row = find_shaoxing_row(target)
        if shaoxing_row is not None:
            definition = clean_cell_text(shaoxing_row[4]) or "\u6682\u7f3a"
            pinyin = _row_pinyin(shaoxing_row) or "\u6682\u7f3a"
            if TEXT_ONLY_MODE or pinyin == "\u6682\u7f3a":
                return jsonify(
                    {
                        "text": (
                            f"\u5339\u914d\u5230\uff1a{shaoxing_row[0]}"
                            f"<br>\u62fc\u97f3\uff1a{pinyin}"
                            f"<br>\u91ca\u4e49\uff1a{definition}"
                        )
                    }
                )
            if not tts_model_ready("shaoxing"):
                return jsonify(
                    {
                        "text": (
                            f"\u5339\u914d\u5230\uff1a{shaoxing_row[0]}"
                            f"<br>\u62fc\u97f3\uff1a{pinyin}"
                            f"<br>\u91ca\u4e49\uff1a{definition}"
                            "<br>绍兴话 TTS 模型未就绪。"
                        )
                    }
                )
            unique_fn = f"{uuid.uuid4().hex[:8]}.wav"
            try:
                synthesize_dictionary_audio(
                    source_id=SHAOXING_SOURCE_ID,
                    headword=clean_cell_text(shaoxing_row[0]) or target,
                    pinyin=pinyin,
                    output_path=STATIC_DIR / unique_fn,
                )
            except RuntimeError as exc:
                return jsonify({"text": f"\u5339\u914d\u5230\uff1a{shaoxing_row[0]}<br>TTS 加载失败：{exc}"})
            session["last_audio"] = {"filename": unique_fn, "word": str(shaoxing_row[0]), "model": "shaoxing", "source": SHAOXING_SOURCE_ID}
            return jsonify(
                {
                    "text": (
                        f"\u5339\u914d\u5230\uff1a{shaoxing_row[0]}"
                        f"<br>\u62fc\u97f3\uff1a{pinyin}"
                        f"<br>\u91ca\u4e49\uff1a{definition}"
                        "<br><span style='color:#64748b;'>模型：shaoxing</span>"
                    ),
                    "audio": f"/static/{unique_fn}",
                    "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 1)},
                }
            )
        return jsonify({"text": f"\u672a\u627e\u5230\u8bcd\u6761\uff1a{target}"})

    requested_mode = str(payload.get("pipeline_mode") or "lookup").strip().lower()
    if requested_mode in {"rag_mt", "rag-mt", "mt"} and rag_mt_ready():
        model_size = str(payload.get("rag_mt_model_size", RAG_MT_MODEL_SIZE)).strip().lower()
        try:
            result = run_combined_rag_mt_pipeline(user_msg, model_size=model_size)
        except Exception as exc:
            logging.warning("[rag_mt] fallback to dictionary lookup: %s", exc)
            lookup_result = run_dictionary_lookup_pipeline(user_msg, use_legacy_preprocessor=True)
            return jsonify({"text": lookup_result["text"], "mode": "lookup_fallback", "dictionary_result": lookup_result, "timings_ms": lookup_result.get("timings_ms", {})})
        return jsonify(result)

    lookup_result = run_dictionary_lookup_pipeline(user_msg, use_legacy_preprocessor=True)
    return jsonify({"text": lookup_result["text"], "mode": "lookup", "dictionary_result": lookup_result, "timings_ms": lookup_result.get("timings_ms", {})})


def configure_logging() -> None:
    STATIC_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    log_filename = LOG_DIR / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    sys.stdout = Tee(log_filename, mode="a", encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Log file created: %s", log_filename)


if __name__ == "__main__":
    configure_logging()
    bootstrap_dictionary_assets()
    print(f"Pipeline mode: {PIPELINE_MODE}", flush=True)
    if PIPELINE_MODE in {"rag_mt", "rag-mt", "mt"}:
        print(f"RAG-MT model size: {RAG_MT_MODEL_SIZE}", flush=True)
    else:
        print(f"Loading query preprocessor backend: {PREPROCESSOR_BACKEND}", flush=True)
    start_preload_thread()
    print("Starting Flask app...", flush=True)
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1", port=8081, threaded=True, use_reloader=False)
