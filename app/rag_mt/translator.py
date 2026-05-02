from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import jieba.posseg as pseg
except Exception:  # pragma: no cover - optional baseline dependency
    pseg = None


PREFIX_SHELL_RE = re.compile(
    r"^(?:那个|这个|嗯|呃|啊|就是|麻烦|麻烦你|请问|我想问下|我想问一下|"
    r"帮我查一下|帮我看看|请帮我|帮我|想问下|想问一下|查一下)[\s，,。.!！？?；;：:、.]*"
)
SUFFIX_SHELL_PATTERNS = [
    r"(用)?上海话(里)?(怎么|怎样|咋)(说|讲|表达|念)[？?。!！\s]*$",
    r"(怎么|怎样|咋)(用)?上海话(说|讲|表达|念)[？?。!！\s]*$",
    r"(怎么|怎样|咋)(说|讲|表达|念)[？?。!！\s]*$",
    r"(是什么意思|啥意思|什么叫)[？?。!！\s]*$",
]

PUNCT_RE = re.compile(r"^[\s，,。.!！？?；;：:、\"'“”‘’（）()\[\]【】]+|[\s，,。.!！？?；;：:、\"'“”‘’（）()\[\]【】]+$")
STOPWORDS = {
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "一下",
    "请问",
    "麻烦",
    "帮我",
    "上海话",
    "怎么",
    "怎样",
    "表达",
}
KEYWORD_FLAGS = ("n", "v", "a", "t", "s", "l", "i")


@dataclass(slots=True)
class CleanedQuery:
    core_text: str
    keywords: list[str]
    segments: list[str]
    backend: str = "rules_jieba"

    def as_parsed(self) -> dict[str, Any]:
        query_type = "动作短语" if len(self.core_text) > 2 else "词项"
        return {
            "core_text": self.core_text,
            "type": query_type,
            "predicate": self.keywords[0] if query_type == "动作短语" and self.keywords else "",
            "object": self.keywords[-1] if query_type == "动作短语" and len(self.keywords) > 1 else "",
            "segments": self.segments,
            "keywords": self.keywords,
            "backend": self.backend,
        }


def clean_query(query: str, max_keywords: int = 8) -> CleanedQuery:
    core = str(query or "").strip()
    previous = None
    while previous != core:
        previous = core
        core = PREFIX_SHELL_RE.sub("", core)
    for pattern in SUFFIX_SHELL_PATTERNS:
        core = re.sub(pattern, "", core)
    core = PUNCT_RE.sub("", re.sub(r"\s+", "", core))
    if not core:
        core = str(query or "").strip()

    segments = segment_core(core)
    keywords = extract_keywords(core, segments, max_keywords=max_keywords)
    return CleanedQuery(core_text=core, keywords=keywords, segments=segments)


def segment_core(core_text: str) -> list[str]:
    if not core_text:
        return []
    if pseg is None:
        return [core_text]
    segments: list[str] = []
    for word, _flag in pseg.cut(core_text):
        token = str(word).strip()
        if token and token not in STOPWORDS:
            segments.append(token)
    return dedupe_keep_order(segments) or [core_text]


def extract_keywords(core_text: str, segments: list[str], max_keywords: int = 8) -> list[str]:
    candidates: list[str] = []
    if pseg is not None:
        for word, flag in pseg.cut(core_text):
            token = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(word).strip())
            if not token or token in STOPWORDS:
                continue
            if len(token) >= 2 or str(flag).startswith(KEYWORD_FLAGS):
                candidates.append(token)
    else:
        candidates.extend(_fallback_keyword_windows(core_text))
    candidates.extend(token for token in segments if len(token) >= 2 and token not in STOPWORDS)
    if "吗" in core_text:
        candidates.append("吗")
    if core_text and core_text not in candidates:
        candidates.insert(0, core_text)
    return dedupe_keep_order(candidates)[:max_keywords]


def _fallback_keyword_windows(text: str) -> list[str]:
    clean = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    if len(clean) <= 2:
        return [clean] if clean else []
    windows: list[str] = [clean]
    for token in (clean[:3], clean[-2:], clean[-3:], clean[:2]):
        if token and token not in windows and token not in STOPWORDS:
            windows.append(token)
    for size in (4, 3, 2):
        if len(clean) < size:
            continue
        for start in range(0, len(clean) - size + 1):
            token = clean[start : start + size]
            if token not in STOPWORDS:
                windows.append(token)
    return windows


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def sanitize_headword(text: str) -> str:
    return str(text or "").strip().strip("[]").strip("【】")


def build_lexicon_context(
    keyword_hits: list[tuple[str, list[dict[str, Any]]]],
    max_pairs: int = 12,
    max_candidates_per_keyword: int = 2,
) -> str:
    pairs: list[str] = []
    seen: set[tuple[str, str]] = set()
    for keyword, hits in keyword_hits:
        clean_keyword = str(keyword).strip()
        if not clean_keyword:
            continue
        for hit in hits[:max_candidates_per_keyword]:
            shanghai = sanitize_headword(str(hit.get("shanghai", "")))
            if not shanghai:
                continue
            pair = (clean_keyword, shanghai)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(f"{clean_keyword}={shanghai}")
            if len(pairs) >= max_pairs:
                return "; ".join(pairs)
    return "; ".join(pairs)


class QwenRAGMTTranslator:
    def __init__(
        self,
        model_path: Path,
        adapter_path: Path | None = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.adapter_path = Path(adapter_path).resolve() if adapter_path else None
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        if not self.model_path.exists():
            raise FileNotFoundError(f"RAG-MT model not found: {self.model_path}")
        if self.adapter_path is not None and not self.adapter_path.exists():
            raise FileNotFoundError(f"RAG-MT adapter not found: {self.adapter_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
            kwargs["dtype"] = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

        model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs)
        if self.adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(self.adapter_path))
        self.model = model.eval()

    def build_prompt(
        self,
        core_text: str,
        context: str,
        raw_query: str = "",
        keywords: list[str] | None = None,
    ) -> str:
        keyword_text = "、".join(str(item).strip() for item in (keywords or []) if str(item).strip()) or "无"
        messages = [
            {
                "role": "system",
                "content": "你是一个精确的上海话翻译器。请参考给定的词汇对照表，将普通话翻译为上海话。不解释，只输出结果。",
            },
            {
                "role": "user",
                "content": (
                    f"【原始输入】：{raw_query or core_text}\n"
                    f"【清洗后普通话】：{core_text}\n"
                    f"【关键词】：{keyword_text}\n"
                    f"【参考词汇】：{context or '无'}"
                ),
            },
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                return f"{messages[0]['content']}\n\n{messages[1]['content']}\n【上海话】："

    def translate(self, core_text: str, context: str, raw_query: str = "", keywords: list[str] | None = None) -> str:
        prompt = self.build_prompt(core_text, context, raw_query=raw_query, keywords=keywords)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            outputs = self.model.generate(**encoded, **gen_kwargs)
        generated = outputs[0][encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return clean_translation_output(text)


def clean_translation_output(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^【上海话】[:：]?", "", value).strip()
    value = value.split("<|im_end|>")[0].strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if lines:
        value = lines[0]
    return value.strip("\"'“”")
