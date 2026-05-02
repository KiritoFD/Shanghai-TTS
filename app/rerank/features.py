from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


def sanitize_headword(text: str) -> str:
    return re.sub(r"[\[\]【】\s]", "", str(text)).strip()


def normalize_simple(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).strip()


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
    normalized = normalize_simple(user_msg)
    if normalized:
        terms.append(normalized)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def _char_overlap(a: str, b: str) -> tuple[float, float]:
    if not a or not b:
        return 0.0, 0.0
    common = sum(min(a.count(ch), b.count(ch)) for ch in set(a))
    return common / max(len(a), 1), common / max(len(b), 1)


@dataclass
class FeatureRow:
    values: list[float]
    names: list[str]


def build_feature_row(user_msg: str, parsed: dict[str, Any], candidate: dict[str, Any], rank: int) -> FeatureRow:
    shanghai = sanitize_headword(candidate.get("shanghai", ""))
    definition = normalize_simple(candidate.get("definition", ""))
    query = normalize_simple(user_msg)
    core_text = normalize_simple(parsed.get("core_text", ""))
    query_type = str(parsed.get("type", "")).strip()
    terms = _candidate_terms_from_parsed(parsed, user_msg)
    segments = parsed.get("segments", [])
    if not isinstance(segments, list):
        segments = []

    term_exact = 0
    term_in_headword = 0
    term_in_definition = 0
    matched_segments = 0
    longest_term_in_headword = 0
    longest_term_in_definition = 0
    for term in terms:
        norm_term = normalize_simple(term)
        if not norm_term:
            continue
        if norm_term == shanghai:
            term_exact = 1
        if norm_term in shanghai:
            term_in_headword += 1
            longest_term_in_headword = max(longest_term_in_headword, len(norm_term))
        if norm_term in definition:
            term_in_definition += 1
            longest_term_in_definition = max(longest_term_in_definition, len(norm_term))
    for seg in segments:
        norm_seg = normalize_simple(seg)
        if norm_seg and norm_seg in shanghai:
            matched_segments += 1

    query_head_p, head_query_p = _char_overlap(query, shanghai)
    core_head_p, head_core_p = _char_overlap(core_text, shanghai)
    query_def_p, def_query_p = _char_overlap(query, definition)

    values = [
        float(candidate.get("score", 0.0)),
        1.0 / max(rank, 1),
        float(term_exact),
        float(term_in_headword),
        float(term_in_definition),
        float(matched_segments),
        float(longest_term_in_headword),
        float(longest_term_in_definition),
        float(len(shanghai)),
        float(len(definition)),
        query_head_p,
        head_query_p,
        core_head_p,
        head_core_p,
        query_def_p,
        def_query_p,
        1.0 if query_type == "动作短语" else 0.0,
        1.0 if query_type == "词项" else 0.0,
    ]
    names = [
        "base_score",
        "inv_rank",
        "term_exact",
        "term_in_headword",
        "term_in_definition",
        "matched_segments",
        "longest_term_in_headword",
        "longest_term_in_definition",
        "headword_len",
        "definition_len",
        "query_head_precision",
        "head_query_precision",
        "core_head_precision",
        "head_core_precision",
        "query_def_precision",
        "def_query_precision",
        "is_phrase",
        "is_word",
    ]
    return FeatureRow(values=values, names=names)
