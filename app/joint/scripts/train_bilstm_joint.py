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
TAG_TO_ID = {"B": 0, "M": 1, "E": 2, "S": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BiLSTM segmentation + embedding joint baseline")
    parser.add_argument("--train_jsonl", default="app/joint/data/train.jsonl", type=str)
    parser.add_argument("--dev_jsonl", default="app/joint/data/dev.jsonl", type=str)
    parser.add_argument("--test_jsonl", default="app/joint/data/test.jsonl", type=str)
    parser.add_argument("--corpus_jsonl", default="app/joint/data/corpus.jsonl", type=str)
    parser.add_argument("--output_dir", default="app/joint/outputs/bilstm_joint", type=str)
    parser.add_argument("--embedding_dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--projection_dim", default=256, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.15, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=1, type=float)
    parser.add_argument("--learning_rate", default=2e-3, type=float)
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--seg_loss_weight", default=0.2, type=float)
    parser.add_argument("--max_length", default=96, type=int)
    parser.add_argument("--save_every_steps", default=200, type=int)
    parser.add_argument("--eval_every_steps", default=1000, type=int)
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


def encode_tags(tags: list[str], max_length: int) -> list[int]:
    values = [TAG_TO_ID.get(str(tag), -100) for tag in tags[:max_length]]
    return values or [-100]


class JointRows(Dataset):
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
        seg_tags: list[list[int]] = []
        for row, ids in zip(rows, seg_ids):
            tags = encode_tags(row.get("segment_tags", []), max_length)
            if len(tags) < len(ids):
                tags = tags + [-100] * (len(ids) - len(tags))
            seg_tags.append(tags[: len(ids)])

        q, q_len = pad_1d(query_ids, 0)
        d, d_len = pad_1d(doc_ids, 0)
        s, s_len = pad_1d(seg_ids, 0)
        y, _ = pad_1d(seg_tags, -100)
        return {"query_ids": q, "query_len": q_len, "doc_ids": d, "doc_len": d_len, "seg_ids": s, "seg_len": s_len, "seg_labels": y}

    return collate


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
        lengths_cpu = lengths.detach().cpu()
        packed = pack_padded_sequence(embedded, lengths_cpu, batch_first=True, enforce_sorted=False)
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


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_loss(model: BiLSTMJoint, batch: dict, temperature: float, seg_loss_weight: float) -> tuple[torch.Tensor, dict]:
    query_vec = model.encode(batch["query_ids"], batch["query_len"])
    doc_vec = model.encode(batch["doc_ids"], batch["doc_len"])
    logits = query_vec @ doc_vec.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    retrieval_loss = F.cross_entropy(logits, labels)
    seg_logits = model.segment_logits(batch["seg_ids"], batch["seg_len"])
    seg_loss = F.cross_entropy(seg_logits.reshape(-1, seg_logits.size(-1)), batch["seg_labels"].reshape(-1), ignore_index=-100)
    loss = retrieval_loss + seg_loss_weight * seg_loss
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "retrieval_loss": float(retrieval_loss.detach().cpu()),
        "seg_loss": float(seg_loss.detach().cpu()),
        "batch_acc": acc,
    }


@torch.no_grad()
def encode_texts(model: BiLSTMJoint, texts: list[str], vocab: dict[str, int], max_length: int, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    vectors: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        seqs = [encode_chars(text, vocab, max_length) for text in texts[start : start + batch_size]]
        ids, lengths = pad_1d(seqs, 0)
        vectors.append(model.encode(ids.to(device), lengths.to(device)).cpu())
    return torch.cat(vectors, dim=0)


@torch.no_grad()
def evaluate_recall(
    model: BiLSTMJoint,
    rows: list[dict],
    corpus: list[dict],
    vocab: dict[str, int],
    max_length: int,
    batch_size: int,
    device: torch.device,
    limit: int = 0,
) -> dict[str, float]:
    model.eval()
    if limit > 0:
        rows = rows[:limit]
    doc_ids = [row["doc_id"] for row in corpus]
    doc_texts = [row["text"] for row in corpus]
    doc_vecs = encode_texts(model, doc_texts, vocab, max_length, batch_size, device)
    query_vecs = encode_texts(model, [row["query"] for row in rows], vocab, max_length, batch_size, device)
    doc_by_rank = []
    for start in range(0, len(query_vecs), batch_size):
        sims = query_vecs[start : start + batch_size] @ doc_vecs.T
        topk = torch.topk(sims, k=min(10, sims.size(1)), dim=1).indices.tolist()
        doc_by_rank.extend([[doc_ids[i] for i in item] for item in topk])
    hits = {1: 0, 3: 0, 10: 0}
    mrr = 0.0
    for row, ranked in zip(rows, doc_by_rank):
        target = row["positive_doc_id"]
        for k in hits:
            hits[k] += int(target in ranked[:k])
        if target in ranked:
            mrr += 1.0 / (ranked.index(target) + 1)
    n = max(1, len(rows))
    return {
        "count": len(rows),
        "hit@1": hits[1] / n,
        "hit@3": hits[3] / n,
        "hit@10": hits[10] / n,
        "mrr@10": mrr / n,
    }


def save_checkpoint(
    output_dir: Path,
    model: BiLSTMJoint,
    vocab: dict[str, int],
    args: argparse.Namespace,
    step: int,
    metrics: dict,
    name: str | None = None,
) -> Path:
    target = output_dir / (name or f"checkpoint-{step}")
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "model.pt")
    (target / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "training_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
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
    model = BiLSTMJoint(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    loader = DataLoader(JointRows(train_rows), batch_size=args.batch_size, shuffle=True, collate_fn=make_collate(vocab, args.max_length))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, math.ceil(len(loader) * args.epochs))
    step = 0
    best_hit1 = -1.0
    last_metrics: dict = {}

    while step < total_steps:
        for batch in loader:
            model.train()
            batch = to_device(batch, device)
            loss, metrics = train_loss(model, batch, args.temperature, args.seg_loss_weight)
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
                eval_metrics = evaluate_recall(model, dev_rows, corpus, vocab, args.max_length, args.batch_size, device, limit=args.eval_limit)
                print(json.dumps({"step": step, "dev": eval_metrics}, ensure_ascii=False), flush=True)
                if eval_metrics["hit@1"] > best_hit1:
                    best_hit1 = eval_metrics["hit@1"]
                    save_checkpoint(output_dir, model, vocab, args, step, {"train": metrics, "dev": eval_metrics}, name="best")
            if step >= total_steps:
                break

    dev_metrics = evaluate_recall(model, dev_rows, corpus, vocab, args.max_length, args.batch_size, device, limit=0)
    test_metrics = evaluate_recall(model, test_rows, corpus, vocab, args.max_length, args.batch_size, device, limit=0)
    final_metrics = {"train_last": last_metrics, "dev": dev_metrics, "test": test_metrics}
    save_checkpoint(output_dir, model, vocab, args, step, final_metrics, name="final")
    (output_dir / "final_metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "step": step, "metrics": final_metrics}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
