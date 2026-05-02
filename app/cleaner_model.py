from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rag_mt import clean_query, extract_keywords, segment_core


PAD = "<pad>"
UNK = "<unk>"


class CoreTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        lengths = (input_ids != 0).sum(dim=1).clamp(min=1).cpu()
        embedded = self.dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=input_ids.size(1))
        return self.classifier(self.dropout(encoded))


class CoreTransformerTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 192,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 6,
        max_length: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.embedding(input_ids) + self.position_embedding(positions)
        padding_mask = input_ids == 0
        encoded = self.encoder(self.dropout(hidden), src_key_padding_mask=padding_mask)
        return self.classifier(self.dropout(encoded))


@dataclass(slots=True)
class CleanerPrediction:
    core_text: str
    keywords: list[str]
    segments: list[str]
    confidence: float
    backend: str

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
            "confidence": round(self.confidence, 4),
        }


class CleanerCoreExtractor:
    def __init__(self, checkpoint_dir: Path, device: str | None = None) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        payload = torch.load(self.checkpoint_dir / "model.pt", map_location="cpu")
        self.vocab: dict[str, int] = payload["vocab"]
        self.max_length = int(payload.get("max_length", 96))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_type = str(payload.get("model_type", "bilstm"))
        if model_type == "transformer":
            self.model = CoreTransformerTagger(
                vocab_size=len(self.vocab),
                embedding_dim=int(payload.get("embedding_dim", 192)),
                hidden_dim=int(payload.get("hidden_dim", 256)),
                num_layers=int(payload.get("num_layers", 4)),
                num_heads=int(payload.get("num_heads", 6)),
                max_length=self.max_length,
            )
        else:
            self.model = CoreTagger(
                vocab_size=len(self.vocab),
                embedding_dim=int(payload.get("embedding_dim", 128)),
                hidden_dim=int(payload.get("hidden_dim", 128)),
                num_layers=int(payload.get("num_layers", 1)),
            )
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, text: str) -> tuple[torch.Tensor, str]:
        compact = re.sub(r"\s+", "", str(text or ""))[: self.max_length]
        ids = [self.vocab.get(ch, self.vocab.get(UNK, 1)) for ch in compact] or [self.vocab.get(UNK, 1)]
        return torch.tensor([ids], dtype=torch.long, device=self.device), compact

    @torch.no_grad()
    def predict(self, query: str) -> CleanerPrediction:
        input_ids, compact = self._encode(query)
        logits = self.model(input_ids)[0, : len(compact)]
        probs = logits.softmax(dim=-1)[:, 1]
        mask = probs >= 0.5
        core = self._best_span(compact, probs, mask)
        if not core:
            fallback = clean_query(query)
            return CleanerPrediction(
                core_text=fallback.core_text,
                keywords=fallback.keywords,
                segments=fallback.segments,
                confidence=0.0,
                backend="core_tagger_fallback_rules",
            )
        segments = segment_core(core)
        keywords = extract_keywords(core, segments)
        confidence = float(probs[mask].mean().detach().cpu()) if mask.any() else float(probs.max().detach().cpu())
        return CleanerPrediction(core_text=core, keywords=keywords, segments=segments, confidence=confidence, backend="core_tagger")

    @staticmethod
    def _best_span(text: str, probs: torch.Tensor, mask: torch.Tensor) -> str:
        best_start = -1
        best_end = -1
        best_score = 0.0
        start = None
        for idx, active in enumerate(mask.tolist() + [False]):
            if active and start is None:
                start = idx
            if not active and start is not None:
                end = idx
                score = float(probs[start:end].mean().detach().cpu()) * (end - start)
                if score > best_score:
                    best_start, best_end, best_score = start, end, score
                start = None
        if best_start < 0:
            return ""
        return text[best_start:best_end].strip("，,。.!！？?；;：:、")


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
