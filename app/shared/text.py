"""Shared text processing utilities for Shanghai-TTS pipeline.

Single source of truth for:
  - clean_text: whitespace removal (was duplicated in 5 files)
  - normalize_query: shell-pattern stripping + punctuation (was duplicated in 5 files)
  - QUESTION_SHELL_PATTERNS: canonical list of query shell patterns
  - LOW_INFORMATION_TOKENS: stop words for sparse retrieval
"""
from __future__ import annotations

import re


QUESTION_SHELL_PATTERNS: list[str] = [
    "用上海话怎么说",
    "上海话怎么说",
    "上海话怎么讲",
    "上海话里",
    "怎么说",
    "怎么讲",
    "怎么表达",
    "是什么意思",
    "啥意思",
    "这个词",
    "这个",
    "请问",
    "麻烦问下",
    "麻烦",
]

LOW_INFORMATION_TOKENS: set[str] = {
    "我",
    "你",
    "他",
    "她",
    "它",
    "侬",
    "的",
    "了",
    "啊",
    "呀",
    "呢",
    "吗",
    "吧",
    "很",
    "太",
    "最",
    "更",
    "一下",
    "一",
    "下",
    "怎么",
    "说",
    "讲",
}


def clean_text(text: str) -> str:
    """Remove all whitespace from text. Canonical implementation (was duplicated in 5 files)."""
    return "".join(ch for ch in str(text).strip() if not ch.isspace())


def normalize_query(query: str) -> str:
    """Strip question shell patterns and punctuation from a user query.

    Consolidates 5 implementations from app.py, engine.py, bilstm_runtime.py,
    seg_pos_runtime.py, and split/rules.py into one with the superset of all
    patterns and punctuation handling.
    """
    text = str(query).strip()
    for pattern in QUESTION_SHELL_PATTERNS:
        text = text.replace(pattern, "")
    text = re.sub(r"[\"'，,。.!！?？；;：:（）()\[\]【】]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or str(query).strip()


def is_meaningful(text: str, allow_single_char: bool = False) -> bool:
    """Check whether *text* contains enough signal to be a useful search variant.

    Requires at least one non-stopword token (see LOW_INFORMATION_TOKENS).
    """
    normalized = re.sub(r"\s+", "", text.strip())
    if not normalized:
        return False
    if len(normalized) == 1 and not allow_single_char:
        return False
    return bool(_tokenize_for_meaning(normalized))


def _tokenize_for_meaning(text: str) -> list[str]:
    """Lightweight tokenizer used by is_meaningful (avoids importing engine.py)."""
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())
    if not cleaned:
        return []
    tokens: list[str] = []
    current_ascii: list[str] = []
    cjk_chars: list[str] = []
    for ch in cleaned:
        if re.match(r"[a-z0-9_]", ch):
            current_ascii.append(ch)
            if cjk_chars:
                chars = cjk_chars
                tokens.extend(chars)
                if len(chars) >= 2:
                    tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
                cjk_chars = []
        else:
            cjk_chars.append(ch)
            if current_ascii:
                tokens.append("".join(current_ascii))
                current_ascii = []
    if current_ascii:
        tokens.append("".join(current_ascii))
    if cjk_chars:
        tokens.extend(cjk_chars)
        if len(cjk_chars) >= 2:
            tokens.extend("".join(cjk_chars[i : i + 2]) for i in range(len(cjk_chars) - 1))
    return [t for t in tokens if t not in LOW_INFORMATION_TOKENS]
