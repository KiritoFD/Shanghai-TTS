"""Verb negation expansion: 不→勿 pattern for Shanghai dialect matching.

In Shanghai dialect writing, "不" in verb compounds is often written as "勿".
Examples: 讲不来 → 讲勿来, 做不来 → 做勿来, 对不住 → 对勿住

This module provides targeted expansion for these patterns.
"""
from __future__ import annotations

import re


def expand_verb_negation(query: str) -> list[str]:
    """Expand verb+不 patterns to verb+勿 variants.

    For queries like "讲不来", generates "讲勿来".
    """
    variants = []
    # Pattern: CJK char + 不 + CJK chars
    matches = re.finditer(r"([\u4e00-\u9fff])不([\u4e00-\u9fff])", query)
    for m in matches:
        expanded = query[:m.start()] + m.group(1) + "勿" + m.group(2) + query[m.end():]
        if expanded != query:
            variants.append(expanded)
    return variants


def extract_core_concept(query: str) -> list[str]:
    """Extract the most likely target concept from a multi-concept query.

    Heuristics:
    1. If query has a clear verb+object structure, extract the verb+object
    2. If query has "怎么说X" pattern, extract X
    3. If query has "X是什么意思" pattern, extract X
    """
    concepts = []

    # Pattern: "怎么说X" or "怎么讲X" → X
    m = re.search(r"怎么(?:说|讲|表达)([\u4e00-\u9fff]+)", query)
    if m:
        concepts.append(m.group(1))

    # Pattern: "X是什么意思" or "X啥意思" → X
    m = re.search(r"([\u4e00-\u9fff]+)(?:是什么意思|啥意思)", query)
    if m:
        concepts.append(m.group(1))

    # Pattern: "X怎么说" → X
    m = re.search(r"([\u4e00-\u9fff]{2,})怎么说", query)
    if m:
        concepts.append(m.group(1))

    # Pattern: "X这个词" or "X这个" → X
    m = re.search(r"([\u4e00-\u9fff]{2,})这个", query)
    if m:
        concepts.append(m.group(1))

    return list(dict.fromkeys(concepts))  # dedupe while preserving order
