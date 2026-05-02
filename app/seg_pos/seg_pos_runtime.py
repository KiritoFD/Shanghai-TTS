"""Runtime wrapper for the BiLSTM segmentation + POS tagger model."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from shared.model_defs import (
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
from shared.text import clean_text, normalize_query as normalize_query_text

SEG_POS_DIR = Path(__file__).resolve().parent
if str(SEG_POS_DIR) not in sys.path:
    sys.path.insert(0, str(SEG_POS_DIR))

try:
    from train_bilstm_seg_pos import BiLSTMSegPos
except ImportError:
    BiLSTMSegPos = None  # type: ignore[misc,assignment]

_HAS_V2 = False
_HAS_V3 = False
try:
    from train_bilstm_seg_pos_v3 import BiLSTMCRFSegPos  # v3 has layer_norm
    _HAS_V3 = True
    _HAS_V2 = True
except ImportError:
    try:
        from train_bilstm_seg_pos_v2 import BiLSTMCRFSegPos
        _HAS_V2 = True
    except ImportError:
        pass


class SegPosPreprocessor:
    def __init__(self, checkpoint_dir: Path, max_length: int | None = None):
        self.checkpoint_dir = checkpoint_dir.resolve()
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Seg+POS checkpoint not found: {self.checkpoint_dir}")
        args = json.loads((self.checkpoint_dir / "training_args.json").read_text(encoding="utf-8"))
        self.vocab: dict[str, int] = json.loads((self.checkpoint_dir / "vocab.json").read_text(encoding="utf-8"))
        self.max_length = int(max_length or args.get("max_length", 96))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_pos = len(args.get("pos_tags", POS_TAGS))
        model_type = args.get("model_type", "bilstm_seg_pos")
        use_crf = args.get("use_crf", False) and _HAS_V2

        if model_type == "bilstm_crf_seg_pos" and use_crf:
            self.model = BiLSTMCRFSegPos(
                vocab_size=len(self.vocab),
                embedding_dim=int(args["embedding_dim"]),
                hidden_dim=int(args["hidden_dim"]),
                projection_dim=int(args["projection_dim"]),
                num_layers=int(args["num_layers"]),
                dropout=float(args["dropout"]),
                num_pos_tags=num_pos,
                use_crf=True,
            ).to(self.device)
            self._use_crf = True
        else:
            self.model = BiLSTMSegPos(
                vocab_size=len(self.vocab),
                embedding_dim=int(args["embedding_dim"]),
                hidden_dim=int(args["hidden_dim"]),
                projection_dim=int(args["projection_dim"]),
                num_layers=int(args["num_layers"]),
                dropout=float(args["dropout"]),
                num_pos_tags=num_pos,
            ).to(self.device)
            self._use_crf = False

        self.model.load_state_dict(torch.load(self.checkpoint_dir / "model.pt", map_location=self.device, weights_only=True))
        self.model.eval()
        print(f"[model] seg_shanghai: {self.checkpoint_dir} (crf={self._use_crf})")

    @torch.no_grad()
    def _predict_tags(self, text: str) -> tuple[list[str], list[str]]:
        ids, lengths = encode_chars(text, self.vocab, self.max_length)
        ids = ids.to(self.device)
        lengths = lengths.to(self.device)
        if self._use_crf and hasattr(self.model, "seg_emissions"):
            emissions = self.model.seg_emissions(ids, lengths)
            L = int(lengths.item())
            mask = torch.ones(1, L, dtype=torch.bool, device=self.device)
            decoded = self.model.crf.decode(emissions[:, :L], mask)[0]
            seg_tags = [SEG_ID_TO_TAG.get(t, "S") for t in decoded]
        else:
            seg_logits = self.model.segment_logits(ids, lengths)[0, : lengths.item()]
            seg_tags = [SEG_ID_TO_TAG[int(idx)] for idx in seg_logits.argmax(dim=-1).detach().cpu().tolist()]
        pos_logits = self.model.pos_logits(ids, lengths)[0, : lengths.item()]
        pos_tags = [POS_ID_TO_TAG.get(int(idx), "X") for idx in pos_logits.argmax(dim=-1).detach().cpu().tolist()]
        return seg_tags, pos_tags

    def _tags_to_chunks(self, text: str, seg_tags: list[str], pos_tags: list[str]) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        current = ""
        current_pos = ""
        for ch, stag, ptag in zip(text, seg_tags, pos_tags):
            if stag == "S":
                if current:
                    chunks.append({"text": current, "pos": current_pos})
                    current = ""
                chunks.append({"text": ch, "pos": ptag})
            elif stag == "B":
                if current:
                    chunks.append({"text": current, "pos": current_pos})
                current = ch
                current_pos = ptag
            elif stag == "M":
                current += ch
            elif stag == "E":
                current += ch
                if current:
                    chunks.append({"text": current, "pos": current_pos if current_pos else ptag})
                current = ""
                current_pos = ""
            else:
                if current:
                    chunks.append({"text": current, "pos": current_pos})
                    current = ""
                chunks.append({"text": ch, "pos": ptag})
        if current:
            chunks.append({"text": current, "pos": current_pos})
        return chunks

    def _expand_chunks(self, text: str, chunks: list[dict[str, str]]) -> list[str]:
        expanded = [text]
        for chunk in chunks:
            t = chunk["text"].strip()
            if t and t != text:
                expanded.append(t)
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
        seg_tags, pos_tags = self._predict_tags(segment_text)
        chunks = self._tags_to_chunks(segment_text, seg_tags, pos_tags)
        keywords = self._expand_chunks(segment_text, chunks)
        return {
            "core_text": core_text,
            "type": chunks[0]["pos"] if chunks and chunks[0].get("pos") else "词项",
            "predicate": "",
            "object": "",
            "keywords": keywords[:8],
            "segments": [c["text"] for c in chunks if c.get("text")],
            "segment_tags": seg_tags,
            "pos_tags": pos_tags,
            "chunks": chunks,
            "backend": "bilstm_seg_shanghai",
        }


SegShanghaiPreprocessor = SegPosPreprocessor
