from __future__ import annotations

from dataclasses import dataclass, field


ALLOWED_TYPES = {"词项", "动作短语"}


@dataclass
class StructuredExample:
    query: str
    core_text: str
    type: str
    predicate: str = ""
    object: str = ""
    keywords: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.query.strip():
            errors.append("empty_query")
        if not self.core_text.strip():
            errors.append("empty_core_text")
        if self.type not in ALLOWED_TYPES:
            errors.append("bad_type")
        if not isinstance(self.keywords, list) or not self.keywords:
            errors.append("empty_keywords")
        return errors

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "core_text": self.core_text,
            "type": self.type,
            "predicate": self.predicate,
            "object": self.object,
            "keywords": self.keywords,
        }
