from __future__ import annotations

import json
import re

from .schema import StructuredExample

QUESTION_SHELL_PATTERNS = [
    r"怎么说$",
    r"怎么讲$",
    r"怎么表达$",
    r"是什么意思$",
    r"啥意思$",
    r"用上海话怎么说$",
    r"用上海话怎么讲$",
    r"上海话里怎么说$",
    r"上海话怎么说$",
    r"上海话怎么讲$",
]

SPECIAL_WORDS = {
    "没用",
    "有意思",
    "难受",
    "开心",
    "伤心",
    "高兴",
}

STOPWORDS = {
    "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "给", "把", "被", "对", "向", "跟", "在", "从", "和",
    "啊", "呢", "吗", "吧", "呀",
    "的", "地", "得",
}


def normalize_query_text(text: str) -> str:
    value = text.strip()
    for pattern in QUESTION_SHELL_PATTERNS:
        value = re.sub(pattern, "", value)
    value = re.sub(r"\s+", "", value)
    return value.strip("，。！？；： ")


def _is_chinese_word(token: str) -> bool:
    return bool(re.fullmatch(r"[\u4e00-\u9fff]+", token))


def _clean_keyword_list(tokens: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = re.sub(r"[^\u4e00-\u9fff]", "", token or "")
        if not token or token in STOPWORDS:
            continue
        if not _is_chinese_word(token):
            continue
        if token not in seen:
            seen.add(token)
            output.append(token)
    return output


def parse_teacher_json(raw_text: str) -> dict:
    raw_text = raw_text.strip()
    if raw_text.startswith("["):
        match = re.search(r"\[.*\]", raw_text, re.S)
    else:
        match = re.search(r"\{.*\}", raw_text, re.S)
    if not match:
        raise ValueError("no_json_object")
    return json.loads(match.group(0))


def clean_prediction(query: str, payload: dict) -> tuple[StructuredExample | None, list[str]]:
    warnings: list[str] = []

    core_text = normalize_query_text(str(payload.get("core_text", "") or query))
    type_value = str(payload.get("type", "")).strip()
    predicate = str(payload.get("predicate", "")).strip()
    object_value = str(payload.get("object", "")).strip()
    keywords = payload.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
        warnings.append("keywords_not_list")

    if core_text in SPECIAL_WORDS:
        type_value = "词项"
        predicate = ""
        object_value = ""

    if object_value in STOPWORDS:
        object_value = ""
        warnings.append("object_stopword_removed")

    clean_keywords = _clean_keyword_list([str(token) for token in keywords])
    if type_value == "动作短语":
        if predicate and predicate not in STOPWORDS and _is_chinese_word(predicate):
            if predicate not in clean_keywords:
                clean_keywords.insert(0, predicate)
        if object_value and object_value not in STOPWORDS and _is_chinese_word(object_value):
            if object_value not in clean_keywords:
                clean_keywords.append(object_value)
    else:
        predicate = ""
        object_value = ""

    example = StructuredExample(
        query=query,
        core_text=core_text,
        type=type_value,
        predicate=predicate,
        object=object_value,
        keywords=clean_keywords[:5],
    )
    errors = example.validate()

    if errors:
        return None, warnings + errors
    return example, warnings


def render_final_output(example: StructuredExample | dict) -> str:
    if isinstance(example, dict):
        type_value = example["type"]
        keywords = example["keywords"]
    else:
        type_value = example.type
        keywords = example.keywords
    tag = "【单词】" if type_value == "词项" else "【短语】"
    return f"{tag}\n{','.join(keywords)}"


def detect_rule_violations(row: dict) -> list[str]:
    violations: list[str] = []
    query = str(row.get("query", ""))
    type_value = str(row.get("type", ""))
    core_text = str(row.get("core_text", ""))
    predicate = str(row.get("predicate", ""))
    object_value = str(row.get("object", ""))
    keywords = row.get("keywords", [])

    if type_value not in {"词项", "动作短语"}:
        violations.append("invalid_type")

    for shell in ("怎么说", "怎么讲", "怎么表达", "是什么意思", "啥意思"):
        if shell in core_text:
            violations.append("question_shell_retained")
            break

    if object_value in STOPWORDS:
        violations.append("pronoun_or_stopword_object")

    if not isinstance(keywords, list):
        violations.append("keywords_not_list")
        return violations

    seen: set[str] = set()
    for keyword in keywords:
        token = str(keyword)
        if not _is_chinese_word(token):
            violations.append("non_chinese_keyword")
        if token in STOPWORDS:
            violations.append("stopword_keyword")
        if token in seen:
            violations.append("duplicate_keyword")
        seen.add(token)

    rendered = render_final_output(row)
    if "解释" in rendered or "说明" in rendered:
        violations.append("explanatory_text")

    if not query.strip():
        violations.append("empty_query")
    if not core_text.strip():
        violations.append("empty_core_text")
    if type_value == "动作短语" and not predicate:
        violations.append("missing_predicate")

    return sorted(set(violations))
