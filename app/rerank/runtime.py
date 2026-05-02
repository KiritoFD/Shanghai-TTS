from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .features import build_feature_row


class SplitRecallReranker:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir.resolve()
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"reranker checkpoint not found: {self.checkpoint_dir}")
        with (self.checkpoint_dir / "model.pkl").open("rb") as handle:
            self.model = pickle.load(handle)
        self.meta = json.loads((self.checkpoint_dir / "meta.json").read_text(encoding="utf-8"))
        self.feature_names: list[str] = list(self.meta.get("feature_names", []))

    def rerank(self, user_msg: str, parsed: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not results:
            return results
        rows: list[list[float]] = []
        enriched: list[dict[str, Any]] = []
        for rank, candidate in enumerate(results, start=1):
            feature_row = build_feature_row(user_msg, parsed, candidate, rank)
            rows.append(feature_row.values)
            item = dict(candidate)
            item["retrieval_rank"] = rank
            enriched.append(item)
        x = np.asarray(rows, dtype=np.float32)
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(x)[:, 1]
        else:
            scores = self.model.decision_function(x)
        for item, score in zip(enriched, scores.tolist()):
            item["rerank_score"] = round(float(score), 6)
        enriched.sort(
            key=lambda row: (
                float(row.get("rerank_score", 0.0)),
                float(row.get("score", 0.0)),
            ),
            reverse=True,
        )
        return enriched
