from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from shared.model_defs import PAD, UNK, encode_chars
from shared.model_defs import SEG_TAG_TO_ID as TAG_TO_ID, SEG_ID_TO_TAG as ID_TO_TAG
from shared.text import clean_text, normalize_query as normalize_query_text


class BiLSTMJoint(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        projection_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim * 2, projection_dim)
        self.segment_classifier = nn.Linear(hidden_dim * 2, len(TAG_TO_ID))

    def contextualize(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=ids.size(1))
        return self.dropout(encoded)

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.contextualize(ids, lengths)
        mask = (ids != 0).unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.projection(pooled), p=2, dim=-1)

    def segment_logits(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.segment_classifier(self.contextualize(ids, lengths))


class BilstmJointPreprocessor:
    def __init__(self, checkpoint_dir: Path, max_length: int | None = None):
        self.checkpoint_dir = checkpoint_dir.resolve()
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"BiLSTM checkpoint not found: {self.checkpoint_dir}")
        args = json.loads((self.checkpoint_dir / "training_args.json").read_text(encoding="utf-8"))
        self.vocab: dict[str, int] = json.loads((self.checkpoint_dir / "vocab.json").read_text(encoding="utf-8"))
        self.max_length = int(max_length or args.get("max_length", 96))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMJoint(
            vocab_size=len(self.vocab),
            embedding_dim=int(args["embedding_dim"]),
            hidden_dim=int(args["hidden_dim"]),
            projection_dim=int(args["projection_dim"]),
            num_layers=int(args["num_layers"]),
            dropout=float(args["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.checkpoint_dir / "model.pt", map_location=self.device))
        self.model.eval()
        print(f"[model] bilstm_joint: {self.checkpoint_dir}")

    @torch.no_grad()
    def _predict_tags(self, text: str) -> list[str]:
        ids, lengths = encode_chars(text, self.vocab, self.max_length)
        ids = ids.to(self.device)
        lengths = lengths.to(self.device)
        logits = self.model.segment_logits(ids, lengths)[0, : lengths.item()]
        return [ID_TO_TAG[int(idx)] for idx in logits.argmax(dim=-1).detach().cpu().tolist()]

    def _tags_to_chunks(self, text: str, tags: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""
        for ch, tag in zip(text, tags):
            if tag == "S":
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(ch)
            elif tag == "B":
                if current:
                    chunks.append(current)
                current = ch
            elif tag == "M":
                current += ch
            elif tag == "E":
                current += ch
                if current:
                    chunks.append(current)
                current = ""
            else:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(ch)
        if current:
            chunks.append(current)
        return [chunk for chunk in chunks if chunk]

    def _expand_chunks(self, text: str, chunks: list[str]) -> list[str]:
        expanded = [text]
        expanded.extend(chunks)
        compact = clean_text(text)
        if len(compact) >= 2:
            expanded.extend(compact[i : i + 2] for i in range(len(compact) - 1))
        if len(compact) >= 3:
            expanded.extend(compact[i : i + 3] for i in range(len(compact) - 2))
        seen: set[str] = set()
        output: list[str] = []
        for item in expanded:
            item = item.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            output.append(item)
        return output[:12]

    def preprocess(self, query: str, **_kwargs: Any) -> dict[str, Any]:
        core_text = normalize_query_text(query)
        segment_text = clean_text(core_text)
        tags = self._predict_tags(segment_text)
        chunks = self._tags_to_chunks(segment_text, tags)
        keywords = self._expand_chunks(segment_text, chunks)
        return {
            "core_text": core_text,
            "type": "词项",
            "predicate": "",
            "object": "",
            "keywords": keywords[:8],
            "segments": chunks,
            "segment_tags": tags,
            "backend": "bilstm_joint",
        }

