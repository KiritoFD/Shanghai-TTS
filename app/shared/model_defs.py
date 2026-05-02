"""Shared model definitions for Shanghai-TTS segmentation and POS tagging.

Consolidates duplicate code from train_bilstm_seg_pos_v2.py and v3.py:
  - CRF class (identical in both v2 and v3)
  - BMES / POS tag constants
  - encode_chars utility
"""
from __future__ import annotations

import torch
import torch.nn as nn


PAD = "<pad>"
UNK = "<unk>"

# Segmentation tags (BMES scheme)
SEG_TAG_TO_ID: dict[str, int] = {"B": 0, "M": 1, "E": 2, "S": 3}
SEG_ID_TO_TAG: dict[int, str] = {v: k for k, v in SEG_TAG_TO_ID.items()}

# POS tags (19 categories)
POS_TAGS: list[str] = [
    "N", "V", "A", "M", "Q", "R", "D", "P", "C", "SP",
    "AS", "Y", "FW", "I", "O", "IDM", "vn", "nd", "X",
]
POS_TAG_TO_ID: dict[str, int] = {t: i for i, t in enumerate(POS_TAGS)}
POS_ID_TO_TAG: dict[int, str] = {i: t for t, i in POS_TAG_TO_ID.items()}

# BMES transition constraints: which tags can follow which
BMES_CONSTRAINTS: dict[int, list[int]] = {
    0: [1, 2],   # B → M, E
    1: [1, 2],   # M → M, E
    2: [0, 3],   # E → B, S
    3: [0, 3],   # S → B, S
}
BMES_START_TAGS: list[int] = [0, 3]  # B, S
BMES_END_TAGS: list[int] = [2, 3]    # E, S


def encode_chars(
    text: str,
    vocab: dict[str, int],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode text to character ID tensor and length tensor.

    Returns:
        (ids, lengths) where ids is shape (1, seq_len) and lengths is shape (1,).
    """
    chars = clean_text(text)[:max_length]
    if not chars:
        chars = UNK
    ids = [vocab.get(ch, vocab.get(UNK, 1)) for ch in chars]
    return torch.tensor([ids], dtype=torch.long), torch.tensor([len(ids)], dtype=torch.long)


def clean_text(text: str) -> str:
    """Remove all whitespace from text (local copy to avoid circular import)."""
    return "".join(ch for ch in str(text).strip() if not ch.isspace())


class CRF(nn.Module):
    """Conditional Random Field layer for sequence labeling.

    Supports optional transition constraints (e.g. BMES valid transitions).
    Single canonical implementation (was duplicated in v2 and v3).
    """

    def __init__(
        self,
        num_tags: int,
        constraints: dict[int, list[int]] | None = None,
        start_tags: list[int] | None = None,
        end_tags: list[int] | None = None,
    ):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        if constraints is not None:
            with torch.no_grad():
                mask = torch.full((num_tags, num_tags), -1e4)
                for src, dsts in constraints.items():
                    for dst in dsts:
                        mask[src, dst] = 0.0
                self.transitions.data += mask
        if start_tags is not None:
            with torch.no_grad():
                for i in range(num_tags):
                    if i not in start_tags:
                        self.start_transitions.data[i] = -1e4
        if end_tags is not None:
            with torch.no_grad():
                for i in range(num_tags):
                    if i not in end_tags:
                        self.end_transitions.data[i] = -1e4

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tags.shape
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for i in range(1, seq_len):
            score = score + self.transitions[tags[:, i - 1], tags[:, i]] * mask[:, i]
            score = score + emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1) * mask[:, i]
        lengths = mask.sum(dim=1, dtype=torch.long)
        last_tags = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len = emissions.size(1)
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for i in range(1, seq_len):
            new_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            new_score = torch.logsumexp(new_score, dim=1)
            new_score = new_score + emissions[:, i]
            score = torch.where(mask[:, i].unsqueeze(1).bool(), new_score, score)
        score = score + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(score, dim=1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gold_score = self._compute_score(emissions, tags, mask)
        partition = self._compute_partition(emissions, mask)
        return (partition - gold_score).mean()

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history: list[torch.Tensor] = []
        for i in range(1, seq_len):
            new_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best = new_score.argmax(dim=1)
            new_score = new_score.max(dim=1).values + emissions[:, i]
            score = torch.where(mask[:, i].unsqueeze(1).bool(), new_score, score)
            history.append(best)
        score = score + self.end_transitions.unsqueeze(0)
        best_paths: list[list[int]] = []
        lengths = mask.sum(dim=1, dtype=torch.long)
        for b in range(batch_size):
            L = int(lengths[b].item())
            pos = score[b].argmax().item()
            path = [int(pos)]
            for i in range(len(history) - 1, -1, -1):
                if i + 1 < L:
                    pos = int(history[i][b, pos].item())
                    path.append(pos)
            path.reverse()
            best_paths.append(path)
        return best_paths
