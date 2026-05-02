"""BiLSTM joint segmentation + POS tagging model for Shanghainese.

Two heads:
  - segment_classifier: BMES tags (B/M/E/S)
  - pos_classifier: POS tags (N/V/A/M/Q/R/D/P/C/SP/AS/Y/FW/I/O/IDM/vn/nd)

Also produces an embedding for recall via projection layer.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiLSTM segmentation + POS tagger")
    parser.add_argument("--train_jsonl", default="app/seg_pos/data/train.jsonl", type=str)
    parser.add_argument("--dev_jsonl", default="app/seg_pos/data/dev.jsonl", type=str)
    parser.add_argument("--test_jsonl", default="app/seg_pos/data/test.jsonl", type=str)
    parser.add_argument("--corpus_jsonl", default="app/seg_pos/data/corpus.jsonl", type=str)
    parser.add_argument("--output_dir", default="app/seg_pos/outputs/bilstm_seg_pos", type=str)
    parser.add_argument("--embedding_dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--projection_dim", default=256, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=5, type=float)
    parser.add_argument("--learning_rate", default=2e-3, type=float)
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--seg_loss_weight", default=0.3, type=float)
    parser.add_argument("--pos_loss_weight", default=0.3, type=float)
    parser.add_argument("--max_length", default=96, type=int)
    parser.add_argument("--save_every_steps", default=500, type=int)
    parser.add_argument("--eval_every_steps", default=500, type=int)
    parser.add_argument("--eval_limit", default=1000, type=int)
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


class BiLSTMSegPos(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        projection_dim: int,
        num_layers: int,
        dropout: float,
        num_pos_tags: int = len(POS_TAGS),
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
        self.segment_classifier = nn.Linear(hidden_dim * 2, len(SEG_TAG_TO_ID))
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

    def segment_logits(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.segment_classifier(self.contextualize(ids, lengths))

    def pos_logits(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.pos_classifier(self.contextualize(ids, lengths))


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_loss(
    model: BiLSTMSegPos, batch: dict, temperature: float, seg_weight: float, pos_weight: float,
) -> tuple[torch.Tensor, dict]:
    query_vec = model.encode(batch["query_ids"], batch["query_len"])
    doc_vec = model.encode(batch["doc_ids"], batch["doc_len"])
    logits = query_vec @ doc_vec.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    retrieval_loss = F.cross_entropy(logits, labels)

    seg_logits = model.segment_logits(batch["seg_ids"], batch["seg_len"])
    seg_loss = F.cross_entropy(
        seg_logits.reshape(-1, seg_logits.size(-1)), batch["seg_labels"].reshape(-1), ignore_index=-100,
    )

    pos_logits = model.pos_logits(batch["seg_ids"], batch["seg_len"])
    pos_loss = F.cross_entropy(
        pos_logits.reshape(-1, pos_logits.size(-1)), batch["pos_labels"].reshape(-1), ignore_index=-100,
    )

    loss = retrieval_loss + seg_weight * seg_loss + pos_weight * pos_loss
    with torch.no_grad():
        retrieval_acc = (logits.argmax(dim=1) == labels).float().mean().item()
        seg_preds = seg_logits.argmax(dim=-1)
        seg_mask = batch["seg_labels"] != -100
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


@torch.no_grad()
def evaluate(
    model: BiLSTMSegPos, loader: DataLoader, device: torch.device,
    temperature: float, seg_weight: float, pos_weight: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        _, metrics = train_loss(model, batch, temperature, seg_weight, pos_weight)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(
    output_dir: Path, model: BiLSTMSegPos, vocab: dict[str, int], args: argparse.Namespace,
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
        "model_type": "bilstm_seg_pos",
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(Path(args.train_jsonl))
    dev_rows = read_jsonl(Path(args.dev_jsonl))
    test_rows = read_jsonl(Path(args.test_jsonl))
    corpus = read_jsonl(Path(args.corpus_jsonl))
    vocab = build_vocab(train_rows, corpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMSegPos(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    loader = DataLoader(
        SegPosRows(train_rows), batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collate(vocab, args.max_length),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, math.ceil(len(loader) * args.epochs))
    step = 0
    best_seg_acc = -1.0
    last_metrics: dict = {}

    print(json.dumps({
        "vocab_size": len(vocab), "train": len(train_rows), "dev": len(dev_rows),
        "test": len(test_rows), "total_steps": total_steps, "device": str(device),
    }, ensure_ascii=False), flush=True)

    while step < total_steps:
        for batch in loader:
            model.train()
            batch = to_device(batch, device)
            loss, metrics = train_loss(model, batch, args.temperature, args.seg_loss_weight, args.pos_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            last_metrics = metrics
            if step == 1 or step % 20 == 0:
                print(json.dumps({"step": step, "total_steps": total_steps, **metrics}, ensure_ascii=False), flush=True)
            if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                ckpt = save_checkpoint(output_dir, model, vocab, args, step, {"train": metrics})
                print(json.dumps({"saved_checkpoint": str(ckpt), "step": step}, ensure_ascii=False), flush=True)
            if args.eval_every_steps > 0 and step % args.eval_every_steps == 0:
                dev_loader = DataLoader(
                    SegPosRows(dev_rows[:args.eval_limit]), batch_size=args.batch_size, shuffle=False,
                    collate_fn=make_collate(vocab, args.max_length),
                )
                eval_metrics = evaluate(model, dev_loader, device, args.temperature, args.seg_loss_weight, args.pos_loss_weight)
                print(json.dumps({"step": step, "eval": eval_metrics}, ensure_ascii=False), flush=True)
                if eval_metrics.get("seg_acc", 0) + eval_metrics.get("pos_acc", 0) > best_seg_acc:
                    best_seg_acc = eval_metrics.get("seg_acc", 0) + eval_metrics.get("pos_acc", 0)
                    save_checkpoint(output_dir, model, vocab, args, step, {"train": metrics, "dev": eval_metrics}, name="best")
            if step >= total_steps:
                break

    dev_loader = DataLoader(
        SegPosRows(dev_rows), batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collate(vocab, args.max_length),
    )
    test_loader = DataLoader(
        SegPosRows(test_rows), batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collate(vocab, args.max_length),
    )
    dev_metrics = evaluate(model, dev_loader, device, args.temperature, args.seg_loss_weight, args.pos_loss_weight)
    test_metrics = evaluate(model, test_loader, device, args.temperature, args.seg_loss_weight, args.pos_loss_weight)
    final_metrics = {"train_last": last_metrics, "dev": dev_metrics, "test": test_metrics}
    save_checkpoint(output_dir, model, vocab, args, step, final_metrics, name="final")
    (output_dir / "final_metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "step": step, "metrics": final_metrics}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
