"""Shared utilities for Shanghai-TTS pipeline.

Re-exports key functions and constants from submodules so callers can do:
    from shared import clean_text, normalize_query, CRF, load_runtime_config
"""
from __future__ import annotations

from .config import APP_ROOT, REPO_ROOT, CONFIG_PATH, load_runtime_config
from .model_defs import (
    BMES_CONSTRAINTS,
    BMES_END_TAGS,
    BMES_START_TAGS,
    CRF,
    PAD,
    POS_ID_TO_TAG,
    POS_TAG_TO_ID,
    POS_TAGS,
    SEG_ID_TO_TAG,
    SEG_TAG_TO_ID,
    UNK,
    encode_chars,
)
from .text import (
    LOW_INFORMATION_TOKENS,
    QUESTION_SHELL_PATTERNS,
    clean_text,
    is_meaningful,
    normalize_query,
)

__all__ = [
    # text
    "clean_text",
    "normalize_query",
    "is_meaningful",
    "QUESTION_SHELL_PATTERNS",
    "LOW_INFORMATION_TOKENS",
    # model_defs
    "CRF",
    "PAD",
    "UNK",
    "SEG_TAG_TO_ID",
    "SEG_ID_TO_TAG",
    "POS_TAGS",
    "POS_TAG_TO_ID",
    "POS_ID_TO_TAG",
    "BMES_CONSTRAINTS",
    "BMES_START_TAGS",
    "BMES_END_TAGS",
    "encode_chars",
    # config
    "APP_ROOT",
    "REPO_ROOT",
    "CONFIG_PATH",
    "load_runtime_config",
]
