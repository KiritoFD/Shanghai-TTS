from __future__ import annotations

import atexit
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


def _strip_thinking(text: str) -> str:
    think_end_tag = "\n\n"
    if think_end_tag in text:
        last_pos = text.rfind(think_end_tag)
        after = text[last_pos + len(think_end_tag):].strip()
        if after.startswith("{") or any(c.isalpha() for c in after[:5]):
            return after
    return text


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = _strip_thinking(text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return _extract_legacy_output(text)


def _extract_legacy_output(text: str) -> dict[str, Any]:
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


@dataclass(slots=True)
class LlamaCppConfig:
    server_path: Path
    gguf_model_path: Path
    host: str = "127.0.0.1"
    port: int = 8091
    ctx_size: int = 2048
    n_gpu_layers: int = 99
    threads: int = 8
    batch_size: int = 256
    ubatch_size: int = 256
    temperature: float = 0.0
    max_new_tokens: int = 128
    timeout_seconds: int = 30
    auto_start: bool = True
    flash_attention: bool = True

    @property
    def completion_url(self) -> str:
        return f"http://{self.host}:{self.port}/completion"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"


class LlamaCppServiceManager:
    def __init__(self, config: LlamaCppConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self.warning_emitted = False

    def is_healthy(self) -> bool:
        try:
            response = requests.get(self.config.health_url, timeout=min(self.config.timeout_seconds, 3))
            return response.status_code == 200
        except Exception:
            return False

    def ensure_running(self) -> bool:
        if self.is_healthy():
            return True
        if not self.config.auto_start:
            return False
        if self.process is not None and self.process.poll() is None:
            return self._wait_until_healthy(timeout_s=45.0)
        return self.start(wait=True)

    def start(self, wait: bool = True) -> bool:
        if not self.config.server_path.exists():
            raise FileNotFoundError(f"llama.cpp server not found: {self.config.server_path}")
        if not self.config.gguf_model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {self.config.gguf_model_path}")

        log_dir = self.config.gguf_model_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stdout_path = log_dir / f"llama_cpp_split_{stamp}.out.log"
        stderr_path = log_dir / f"llama_cpp_split_{stamp}.err.log"
        stdout_file = open(stdout_path, "a", encoding="utf-8")
        stderr_file = open(stderr_path, "a", encoding="utf-8")

        cmd = [
            str(self.config.server_path),
            "-m", str(self.config.gguf_model_path),
            "--host", self.config.host,
            "--port", str(self.config.port),
            "-c", str(self.config.ctx_size),
            "-ngl", str(self.config.n_gpu_layers),
            "-t", str(self.config.threads),
            "-b", str(self.config.batch_size),
            "-ub", str(self.config.ubatch_size),
        ]
        if self.config.flash_attention:
            cmd.extend(["-fa", "on"])

        logging.info("[llama.cpp] starting split server: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.config.server_path.parent),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        if not wait:
            return True
        return self._wait_until_healthy(timeout_s=60.0)

    def _wait_until_healthy(self, timeout_s: float = 60.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.is_healthy():
                self.warning_emitted = False
                return True
            if self.process is not None and self.process.poll() is not None:
                break
            time.sleep(0.5)
        if not self.warning_emitted:
            logging.warning("[llama.cpp] split server unavailable after startup wait")
            self.warning_emitted = True
        return False

    def shutdown(self) -> None:
        if self.process is None or self.process.poll() is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


class LlamaCppQueryPreprocessor:
    def __init__(self, config: LlamaCppConfig, base_model_path: Path, system_prompt: str):
        self.config = config
        self.base_model_path = base_model_path.resolve()
        self.system_prompt = system_prompt
        if not self.base_model_path.exists():
            raise FileNotFoundError(f"Base model not found: {self.base_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.base_model_path), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._thinking_enabled = self._detect_thinking_template()
        self.service = LlamaCppServiceManager(config)
        atexit.register(self.service.shutdown)

    def _detect_thinking_template(self) -> bool:
        try:
            test_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return "\u672c\u6bb5\u601d\u8003" in test_prompt or "<think>" in test_prompt
        except Exception:
            return False

    def _build_prompt(self, query: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"\u7528\u6237\u8f93\u5165\uff1a{query}"},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if self._thinking_enabled:
            if "<think>\n</think>" not in prompt and "<think>" not in prompt:
                prompt = prompt.rstrip()
                if prompt.endswith("\n"):
                    prompt += "<think>\n</think>\n\n"
        return prompt

    def _completion(self, prompt: str, max_new_tokens: int, temperature: float) -> dict[str, Any]:
        if not self.service.ensure_running():
            raise RuntimeError("llama.cpp split server is not available")

        stop_tokens = ["<|im_end|>", "<|end|>", "\n<|im_start|>"]
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_new_tokens,
            "temperature": max(temperature, 0.01),
            "top_k": 1 if temperature <= 0.01 else 40,
            "cache_prompt": True,
            "stop": stop_tokens,
        }
        if temperature <= 0.01:
            payload["top_p"] = 1.0
            payload["top_k"] = 1
            payload["penalty_repeat"] = 1.0

        try:
            response = requests.post(
                self.config.completion_url, json=payload, timeout=self.config.timeout_seconds,
            )
            response.encoding = "utf-8"
            response.raise_for_status()
            data = response.json()
            content = data.get("content", "")
            if not content and isinstance(data, dict):
                content = data.get("stop", "") or ""
            raw_text = str(content)
        except requests.exceptions.Timeout:
            logging.warning("[llama.cpp] completion request timed out")
            return {"core_text": "", "type": "", "segments": [], "keywords": [], "backend": "llama_cpp", "_error": "timeout"}
        except Exception as exc:
            logging.warning("[llama.cpp] completion request failed: %s", exc)
            return {"core_text": "", "type": "", "segments": [], "keywords": [], "backend": "llama_cpp", "_error": str(exc)}

        parsed = _extract_json(raw_text)
        parsed["backend"] = "llama_cpp"
        parsed["_raw"] = raw_text[:200]
        return parsed

    def preprocess(self, query: str, max_new_tokens: int | None = None, temperature: float | None = None) -> dict[str, Any]:
        return self._completion(
            self._build_prompt(query),
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=self.config.temperature if temperature is None else temperature,
        )

    def preprocess_batch(self, queries: list[str], max_new_tokens: int | None = None, temperature: float | None = None) -> list[dict[str, Any]]:
        return [
            self.preprocess(
                query,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=self.config.temperature if temperature is None else temperature,
            )
            for query in queries
        ]
