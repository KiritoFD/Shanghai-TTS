"""BiLSTM-CRF joint segmentation + POS tagging model for Shanghainese v2.

Improvements over v1:
  - CRF layer for BMES segmentation (enforces valid tag transitions)
  - Cosine LR scheduler with warmup
  - Better loss weighting (higher seg/pos weights, retrieval warmup)
  - Segmentation F1 (word-level) evaluation
  - POS class-weighted loss to handle imbalance
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


PAD = "<pad>"
UNK = "<unk>"
SEG_TAG_TO_ID = {"B": 0, "M": 1, "E": 2, "S": 3}
SEG_ID_TO_TAG = {v: k for k, v in SEG_TAG_TO_ID.items()}

POS_TAGS = ["N", "V", "A", "M", "Q", "R", "D", "P", "C", "SP", "AS", "Y", "FW", "I", "O", "IDM", "vn", "nd", "X"]
POS_TAG_TO_ID = {t: i for i, t in enumerate(POS_TAGS)}
POS_ID_TO_TAG = {i: t for t, i in POS_TAG_TO_ID.items()}

BMES_CONSTRAINTS = {
    0: [1, 2],
    1: [1, 2],
    2: [0, 3],
    3: [0, 3],
}
BMES_START_TAGS = [0, 3]
BMES_END_TAGS = [2, 3]


class CRF(nn.Module):
    def __init__(self, num_tags: int, constraints: dict | None = None,
                 start_tags: list[int] | None = None, end_tags: list[int] | None = None):
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

    def _compute_score(self, emissions, tags, mask):
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

    def _compute_partition(self, emissions, mask):
        seq_len = emissions.size(1)
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for i in range(1, seq_len):
            new_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            new_score = torch.logsumexp(new_score, dim=1)
            new_score = new_score + emissions[:, i]
            score = torch.where(mask[:, i].unsqueeze(1).bool(), new_score, score)
        score = score + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(score, dim=1)

    def forward(self, emissions, tags, mask):
        gold_score = self._compute_score(emissions, tags, mask)
        partition = self._compute_partition(emissions, mask)
        return (partition - gold_score).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history = []
        for i in range(1, seq_len):
            new_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best = new_score.argmax(dim=1)
            new_score = new_score.max(dim=1).values + emissions[:, i]
            score = torch.where(mask[:, i].unsqueeze(1).bool(), new_score, score)
            history.append(best)
        score = score + self.end_transitions.unsqueeze(0)
        best_paths = []
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiLSTM-CRF segmentation + POS tagger v2")
    parser.add_argument("--train_jsonl", default="app/seg_pos/data/train.jsonl", type=str)
    parser.add_argument("--dev_jsonl", default="app/seg_pos/data/dev.jsonl", type=str)
    parser.add_argument("--test_jsonl", default="app/seg_pos/data/test.jsonl", type=str)
    parser.add_argument("--corpus_jsonl", default="app/seg_pos/data/corpus.jsonl", type=str)
    parser.add_argument("--output_dir", default="app/seg_pos/outputs/bilstm_seg_pos_v2", type=str)
    parser.add_argument("--embedding_dim", default=512, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--projection_dim", default=512, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=30, type=float)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--crf_lr_factor", default=0.1, type=float)
    parser.add_argument("--warmup_steps", default=200, type=int)
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--seg_loss_weight", default=1.0, type=float)
    parser.add_argument("--pos_loss_weight", default=1.0, type=float)
    parser.add_argument("--retrieval_loss_weight", default=0.1, type=float)
    parser.add_argument("--use_crf", default=True, type=bool)
    parser.add_argument("--max_length", default=96, type=int)
    parser.add_argument("--save_every_steps", default=500, type=int)
    parser.add_argument("--eval_every_steps", default=300, type=int)
    parser.add_argument("--eval_limit", default=2000, type=int)
    parser.add_argument("--patience", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_text(text: str) -> str:
    return "".join(ch for ch in str(text).strip() if not ch.isspace())


def build_vocab(train_rows: list[dict], corpus_rows: list[dict], min_freq: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in train_rows:
        for field in ("query", "positive_text", "segment_text"):
            for ch in clean_text(row.get(field, "")):
                counts[ch] = counts.get(ch, 0) + 1
    for row in corpus_rows:
        for ch in clean_text(row.get("text", "")):
            counts[ch] = counts.get(ch, 0) + 1
    vocab = {PAD: 0, UNK: 1}
    for ch, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= min_freq and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def compute_pos_weights(train_rows: list[dict], num_pos_tags: int) -> torch.Tensor:
    counts = [1] * num_pos_tags
    for row in train_rows:
        for tag in row.get("pos_tags", []):
            tid = POS_TAG_TO_ID.get(str(tag), -1)
            if 0 <= tid < num_pos_tags:
                counts[tid] += 1
    total = sum(counts)
    weights = [total / (num_pos_tags * c) for c in counts]
    max_w = max(weights)
    weights = [w / max_w for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


def encode_chars(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    text = clean_text(text)[:max_length]
    return [vocab.get(ch, vocab[UNK]) for ch in text] or [vocab[UNK]]


def encode_seg_tags(tags: list[str], max_length: int) -> list[int]:
    values = [SEG_TAG_TO_ID.get(str(tag), 3) for tag in tags[:max_length]]
    return values or [-100]


def encode_pos_tags(tags: list[str], max_length: int) -> list[int]:
    values = [POS_TAG_TO_ID.get(str(tag), POS_TAG_TO_ID["X"]) for tag in tags[:max_length]]
    return values or [-100]


class SegPosRows(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def pad_1d(seqs: list[list[int]], pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out, lengths


def make_collate(vocab: dict[str, int], max_length: int):
    def collate(rows: list[dict]) -> dict:
        query_ids = [encode_chars(row["query"], vocab, max_length) for row in rows]
        doc_ids = [encode_chars(row["positive_text"], vocab, max_length) for row in rows]
        seg_ids = [encode_chars(row.get("segment_text") or row["headword"], vocab, max_length) for row in rows]
        seg_tag_rows: list[list[int]] = []
        pos_tag_rows: list[list[int]] = []
        for row, ids in zip(rows, seg_ids):
            raw_seg = encode_seg_tags(row.get("segment_tags", []), max_length)
            raw_pos = encode_pos_tags(row.get("pos_tags", []), max_length)
            if len(raw_seg) < len(ids):
                raw_seg = raw_seg + [-100] * (len(ids) - len(raw_seg))
            if len(raw_pos) < len(ids):
                raw_pos = raw_pos + [-100] * (len(ids) - len(raw_pos))
            seg_tag_rows.append(raw_seg[: len(ids)])
            pos_tag_rows.append(raw_pos[: len(ids)])

        q, q_len = pad_1d(query_ids, 0)
        d, d_len = pad_1d(doc_ids, 0)
        s, s_len = pad_1d(seg_ids, 0)
        seg_y, _ = pad_1d(seg_tag_rows, -100)
        pos_y, _ = pad_1d(pos_tag_rows, -100)
        return {
            "query_ids": q, "query_len": q_len,
            "doc_ids": d, "doc_len": d_len,
            "seg_ids": s, "seg_len": s_len,
            "seg_labels": seg_y, "pos_labels": pos_y,
        }
    return collate


class BiLSTMCRFSegPos(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        projection_dim: int,
        num_layers: int,
        dropout: float,
        num_pos_tags: int = len(POS_TAGS),
        use_crf: bool = True,
    ):
        super().__init__()
        self.use_crf = use_crf
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
        self.seg_proj = nn.Linear(hidden_dim * 2, len(SEG_TAG_TO_ID))
        if use_crf:
            self.crf = CRF(len(SEG_TAG_TO_ID), BMES_CONSTRAINTS, BMES_START_TAGS, BMES_END_TAGS)
        self.pos_classifier = nn.Linear(hidden_dim * 2, num_pos_tags)

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

    def seg_emissions(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.seg_proj(self.contextualize(ids, lengths))

    def segment_logits(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.seg_emissions(ids, lengths)

    def pos_logits(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.pos_classifier(self.contextualize(ids, lengths))


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_loss(
    model: BiLSTMCRFSegPos, batch: dict, temperature: float,
    seg_weight: float, pos_weight: float, retrieval_weight: float,
    pos_weights: torch.Tensor | None, step: int, warmup_steps: int,
) -> tuple[torch.Tensor, dict]:
    device = next(model.parameters()).device
    rw = retrieval_weight * min(1.0, step / max(1, warmup_steps))

    query_vec = model.encode(batch["query_ids"], batch["query_len"])
    doc_vec = model.encode(batch["doc_ids"], batch["doc_len"])
    logits = query_vec @ doc_vec.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    retrieval_loss = F.cross_entropy(logits, labels)

    seg_emissions = model.seg_emissions(batch["seg_ids"], batch["seg_len"])
    seg_mask = batch["seg_labels"] != -100
    lengths_int = batch["seg_len"]

    if model.use_crf:
        valid = seg_mask.any(dim=1)
        if valid.any():
            em_sub = seg_emissions[valid]
            tag_sub = batch["seg_labels"][valid].clamp(min=0)
            len_sub = lengths_int[valid]
            max_l = int(len_sub.max().item())
            mask_sub = torch.arange(max_l, device=device).unsqueeze(0) < len_sub.unsqueeze(1)
            seg_loss = model.crf(em_sub[:, :max_l], tag_sub[:, :max_l], mask_sub)
        else:
            seg_loss = torch.tensor(0.0, device=device)
    else:
        seg_loss = F.cross_entropy(
            seg_emissions.reshape(-1, seg_emissions.size(-1)),
            batch["seg_labels"].reshape(-1), ignore_index=-100,
        )

    pos_logits = model.pos_logits(batch["seg_ids"], batch["seg_len"])
    pos_loss = F.cross_entropy(
        pos_logits.reshape(-1, pos_logits.size(-1)),
        batch["pos_labels"].reshape(-1), ignore_index=-100,
        weight=pos_weights.to(device) if pos_weights is not None else None,
    )

    loss = rw * retrieval_loss + seg_weight * seg_loss + pos_weight * pos_loss

    with torch.no_grad():
        retrieval_acc = (logits.argmax(dim=1) == labels).float().mean().item()
        if model.use_crf and seg_mask.any():
            seg_preds_all: list[int] = []
            seg_golds_all: list[int] = []
            for b in range(seg_emissions.size(0)):
                L = int(lengths_int[b].item())
                if L == 0:
                    continue
                em_b = seg_emissions[b:b + 1, :L]
                mask_b = torch.ones(1, L, dtype=torch.bool, device=device)
                decoded = model.crf.decode(em_b, mask_b)[0]
                gold = batch["seg_labels"][b, :L].clamp(min=0).tolist()
                seg_preds_all.extend(decoded)
                seg_golds_all.extend(gold)
            if seg_golds_all:
                seg_acc = sum(p == g for p, g in zip(seg_preds_all, seg_golds_all)) / len(seg_golds_all)
            else:
                seg_acc = 0.0
        else:
            seg_preds = seg_emissions.argmax(dim=-1)
            seg_acc = (seg_preds[seg_mask] == batch["seg_labels"][seg_mask]).float().mean().item() if seg_mask.any() else 0.0
        pos_preds = pos_logits.argmax(dim=-1)
        pos_mask = batch["pos_labels"] != -100
        pos_acc = (pos_preds[pos_mask] == batch["pos_labels"][pos_mask]).float().mean().item() if pos_mask.any() else 0.0

    return loss, {
        "loss": float(loss.detach().cpu()),
        "retrieval_loss": float(retrieval_loss.detach().cpu()),
        "seg_loss": float(seg_loss.detach().cpu()),
        "pos_loss": float(pos_loss.detach().cpu()),
        "batch_acc": retrieval_acc,
        "seg_acc": seg_acc,
        "pos_acc": pos_acc,
    }


def tags_to_words(tags: list[str]) -> list[tuple[int, int]]:
    words: list[tuple[int, int]] = []
    start = 0
    for i, t in enumerate(tags):
        if t == "S":
            words.append((start, i + 1))
            start = i + 1
        elif t == "E":
            words.append((start, i + 1))
            start = i + 1
    return words


def compute_seg_f1(pred_tags_list: list[list[str]], gold_tags_list: list[list[str]]) -> dict[str, float]:
    tp, fp, fn = 0, 0, 0
    for pred, gold in zip(pred_tags_list, gold_tags_list):
        pred_words = set(tuple(w) for w in tags_to_words(pred))
        gold_words = set(tuple(w) for w in tags_to_words(gold))
        tp += len(pred_words & gold_words)
        fp += len(pred_words - gold_words)
        fn += len(gold_words - pred_words)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"seg_precision": prec, "seg_recall": rec, "seg_f1": f1}


@torch.no_grad()
def evaluate(
    model: BiLSTMCRFSegPos, loader: DataLoader, device: torch.device,
    temperature: float, seg_weight: float, pos_weight: float, retrieval_weight: float,
    pos_weights: torch.Tensor | None, warmup_steps: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    all_pred_seg: list[list[str]] = []
    all_gold_seg: list[list[str]] = []
    for batch in loader:
        batch = to_device(batch, device)
        _, metrics = train_loss(model, batch, temperature, seg_weight, pos_weight, retrieval_weight, pos_weights, 999999, warmup_steps)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if model.use_crf:
            seg_emissions = model.seg_emissions(batch["seg_ids"], batch["seg_len"])
            lengths_int = batch["seg_len"]
            for b in range(seg_emissions.size(0)):
                L = int(lengths_int[b].item())
                if L == 0:
                    continue
                em_b = seg_emissions[b:b + 1, :L]
                mask_b = torch.ones(1, L, dtype=torch.bool, device=device)
                decoded = model.crf.decode(em_b, mask_b)[0]
                pred_tags = [SEG_ID_TO_TAG.get(t, "S") for t in decoded]
                gold_tags = [SEG_ID_TO_TAG.get(int(t), "S") for t in batch["seg_labels"][b, :L].tolist() if int(t) >= 0]
                if pred_tags and gold_tags:
                    all_pred_seg.append(pred_tags)
                    all_gold_seg.append(gold_tags)
    result = {key: value / count for key, value in totals.items()} if count > 0 else {}
    if all_pred_seg:
        f1_metrics = compute_seg_f1(all_pred_seg, all_gold_seg)
        result.update(f1_metrics)
    return result


def save_checkpoint(
    output_dir: Path, model: BiLSTMCRFSegPos, vocab: dict[str, int], args: argparse.Namespace,
    step: int, metrics: dict, name: str | None = None,
) -> Path:
    target = output_dir / (name or f"checkpoint-{step}")
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "model.pt")
    (target / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "global_step": step,
        "seg_tag_to_id": SEG_TAG_TO_ID,
        "pos_tag_to_id": POS_TAG_TO_ID,
        "pos_tags": POS_TAGS,
        "model_type": "bilstm_crf_seg_pos" if args.use_crf else "bilstm_seg_pos",
        "use_crf": args.use_crf,
        "metrics": metrics or {},
        **{k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
    }
    (target / "training_args.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "metrics.json").write_text(json.dumps({"step": step, **metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(Path(args.train_jsonl))
    dev_rows = read_jsonl(Path(args.dev_jsonl))
    test_rows = read_jsonl(Path(args.test_jsonl))
    corpus = read_jsonl(Path(args.corpus_jsonl))
    vocab = build_vocab(train_rows, corpus)
    pos_weights = compute_pos_weights(train_rows, len(POS_TAGS))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMCRFSegPos(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_crf=args.use_crf,
    ).to(device)
    pos_weights = pos_weights.to(device)

    loader = DataLoader(
        SegPosRows(train_rows), batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collate(vocab, args.max_length),
    )
    crf_params = set()
    if hasattr(model, "crf"):
        crf_params = set(model.crf.parameters())
    param_groups = [
        {"params": [p for p in model.parameters() if p not in crf_params], "lr": args.learning_rate},
        {"params": list(crf_params), "lr": args.learning_rate * args.crf_lr_factor},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    total_steps = max(1, math.ceil(len(loader) * args.epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    step = 0
    best_score = -1.0
    patience_counter = 0
    last_metrics: dict = {}

    print(json.dumps({
        "vocab_size": len(vocab), "train": len(train_rows), "dev": len(dev_rows),
        "test": len(test_rows), "total_steps": total_steps, "device": str(device),
        "use_crf": args.use_crf, "model_params": sum(p.numel() for p in model.parameters()),
    }, ensure_ascii=False), flush=True)

    while step < total_steps:
        for batch in loader:
            model.train()
            batch = to_device(batch, device)
            loss, metrics = train_loss(
                model, batch, args.temperature, args.seg_loss_weight, args.pos_loss_weight,
                args.retrieval_loss_weight, pos_weights, step, args.warmup_steps,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            last_metrics = metrics
            if step == 1 or step % 20 == 0:
                lr = scheduler.get_last_lr()[0]
                print(json.dumps({"step": step, "total_steps": total_steps, "lr": round(lr, 6), **metrics}, ensure_ascii=False), flush=True)
            if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                ckpt = save_checkpoint(output_dir, model, vocab, args, step, {"train": metrics})
                print(json.dumps({"saved_checkpoint": str(ckpt), "step": step}, ensure_ascii=False), flush=True)
            if args.eval_every_steps > 0 and step % args.eval_every_steps == 0:
                dev_loader = DataLoader(
                    SegPosRows(dev_rows[:args.eval_limit]), batch_size=args.batch_size, shuffle=False,
                    collate_fn=make_collate(vocab, args.max_length),
                )
                eval_metrics = evaluate(
                    model, dev_loader, device, args.temperature, args.seg_loss_weight,
                    args.pos_loss_weight, args.retrieval_loss_weight, pos_weights, args.warmup_steps,
                )
                print(json.dumps({"step": step, "eval": eval_metrics}, ensure_ascii=False), flush=True)
                score = eval_metrics.get("seg_f1", 0) + eval_metrics.get("seg_acc", 0) + eval_metrics.get("pos_acc", 0)
                if score > best_score:
                    best_score = score
                    patience_counter = 0
                    save_checkpoint(output_dir, model, vocab, args, step, {"train": metrics, "dev": eval_metrics}, name="best")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(json.dumps({"early_stop": True, "step": step, "best_score": best_score, "patience_counter": patience_counter}, ensure_ascii=False), flush=True)
                        break
            if step >= total_steps:
                break
        if patience_counter >= args.patience:
            break

    dev_loader = DataLoader(
        SegPosRows(dev_rows), batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collate(vocab, args.max_length),
    )
    test_loader = DataLoader(
        SegPosRows(test_rows), batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collate(vocab, args.max_length),
    )
    dev_metrics = evaluate(
        model, dev_loader, device, args.temperature, args.seg_loss_weight,
        args.pos_loss_weight, args.retrieval_loss_weight, pos_weights, args.warmup_steps,
    )
    test_metrics = evaluate(
        model, test_loader, device, args.temperature, args.seg_loss_weight,
        args.pos_loss_weight, args.retrieval_loss_weight, pos_weights, args.warmup_steps,
    )
    final_metrics = {"train_last": last_metrics, "dev": dev_metrics, "test": test_metrics}
    save_checkpoint(output_dir, model, vocab, args, step, final_metrics, name="final")
    (output_dir / "final_metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "step": step, "metrics": final_metrics}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
