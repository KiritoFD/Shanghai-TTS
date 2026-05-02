from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


TAG_TO_ID = {"B": 0, "M": 1, "E": 2, "S": 3}
ID_TO_TAG = {value: key for key, value in TAG_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint segmentation + embedding fine-tuning for recall encoder")
    parser.add_argument("--train_jsonl", default="app/joint/data/train.jsonl", type=str)
    parser.add_argument("--dev_jsonl", default="app/joint/data/dev.jsonl", type=str)
    parser.add_argument("--model_name_or_path", default="model/bge-m3", type=str)
    parser.add_argument("--output_dir", default="app/joint/outputs/bge_m3_joint", type=str)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--epochs", default=1, type=float)
    parser.add_argument("--max_steps", default=0, type=int, help="0 means derive from epochs")
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--temperature", default=0.05, type=float)
    parser.add_argument("--seg_loss_weight", default=0.15, type=float)
    parser.add_argument("--freeze_encoder", action="store_true", help="Train only the segmentation head; useful for smoke tests")
    parser.add_argument("--eval_steps", default=200, type=int)
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class JointDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class JointEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        hidden_size = int(getattr(self.encoder.config, "hidden_size"))
        self.segment_classifier = nn.Linear(hidden_size, len(TAG_TO_ID))
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        kwargs["token_type_ids"] = token_type_ids
        if self.freeze_encoder:
            with torch.no_grad():
                output = self.encoder(**kwargs)
        else:
            output = self.encoder(**kwargs)
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            pooled = output.pooler_output
        else:
            pooled = mean_pool(output.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def token_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        kwargs["token_type_ids"] = token_type_ids
        if self.freeze_encoder:
            with torch.no_grad():
                output = self.encoder(**kwargs)
        else:
            output = self.encoder(**kwargs)
        return self.segment_classifier(output.last_hidden_state)


def char_tags_to_token_labels(offsets: list[tuple[int, int]], char_tags: list[str], text: str) -> list[int]:
    labels: list[int] = []
    for start, end in offsets:
        if start == end:
            labels.append(-100)
            continue
        if start < 0 or start >= len(char_tags) or end > len(text):
            labels.append(-100)
            continue
        labels.append(TAG_TO_ID.get(char_tags[start], -100))
    return labels


def make_collate(tokenizer, max_length: int):
    def collate(rows: list[dict]) -> dict:
        queries = [str(row["query"]) for row in rows]
        positives = [str(row["positive_text"]) for row in rows]
        segment_texts = [str(row.get("segment_text") or row.get("headword") or row["query"]) for row in rows]
        segment_tags = [list(row.get("segment_tags") or []) for row in rows]

        query_batch = tokenizer(queries, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        doc_batch = tokenizer(positives, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        seg_batch = tokenizer(
            segment_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = seg_batch.pop("offset_mapping").tolist()
        label_rows: list[list[int]] = []
        for text, tags, item_offsets in zip(segment_texts, segment_tags, offsets):
            label_rows.append(char_tags_to_token_labels([tuple(x) for x in item_offsets], tags, text))
        labels = torch.tensor(label_rows, dtype=torch.long)
        return {"query": query_batch, "doc": doc_batch, "seg": seg_batch, "seg_labels": labels}

    return collate


def batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, Mapping):
            moved[key] = {k: v.to(device) for k, v in value.items()}
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def forward_loss(model: JointEncoder, batch: dict, temperature: float, seg_loss_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    query_inputs = batch["query"]
    doc_inputs = batch["doc"]
    seg_inputs = batch["seg"]
    query_vec = model.encode(**query_inputs)
    doc_vec = model.encode(**doc_inputs)

    logits = query_vec @ doc_vec.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    retrieval_loss = F.cross_entropy(logits, labels)

    seg_logits = model.token_logits(**seg_inputs)
    seg_loss = F.cross_entropy(seg_logits.view(-1, seg_logits.size(-1)), batch["seg_labels"].view(-1), ignore_index=-100)
    loss = retrieval_loss + seg_loss_weight * seg_loss
    with torch.no_grad():
        retrieval_acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "retrieval_loss": float(retrieval_loss.detach().cpu()),
        "seg_loss": float(seg_loss.detach().cpu()),
        "batch_acc": retrieval_acc,
    }


@torch.no_grad()
def evaluate(model: JointEncoder, loader: DataLoader, device: torch.device, temperature: float, seg_loss_weight: float) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "retrieval_loss": 0.0, "seg_loss": 0.0, "batch_acc": 0.0}
    count = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        _, metrics = forward_loss(model, batch, temperature, seg_loss_weight)
        for key in totals:
            totals[key] += metrics[key]
        count += 1
    if count == 0:
        return totals
    return {key: value / count for key, value in totals.items()}


def save_model(model: JointEncoder, tokenizer, output_dir: Path, step: int, metrics: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.segment_classifier.state_dict(), output_dir / "segment_classifier.pt")
    meta = {"global_step": step, "tag_to_id": TAG_TO_ID, "metrics": metrics or {}}
    (output_dir / "joint_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = read_jsonl(Path(args.train_jsonl))
    dev_rows = read_jsonl(Path(args.dev_jsonl)) if Path(args.dev_jsonl).exists() else []
    if not train_rows:
        raise RuntimeError("empty training data")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    model = JointEncoder(args.model_name_or_path, freeze_encoder=args.freeze_encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(JointDataset(train_rows), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(JointDataset(dev_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate) if dev_rows else None

    updates_per_epoch = max(1, math.ceil(len(train_rows) / args.batch_size))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, math.ceil(updates_per_epoch * args.epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    output_dir = Path(args.output_dir)
    global_step = 0
    last_metrics: dict[str, float] = {}
    while global_step < total_steps:
        for batch in train_loader:
            model.train()
            batch = batch_to_device(batch, device)
            loss, metrics = forward_loss(model, batch, args.temperature, args.seg_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            last_metrics = metrics
            if global_step == 1 or global_step % 10 == 0:
                print(json.dumps({"step": global_step, **metrics}, ensure_ascii=False))
            if dev_loader is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                eval_metrics = evaluate(model, dev_loader, device, args.temperature, args.seg_loss_weight)
                print(json.dumps({"step": global_step, "eval": eval_metrics}, ensure_ascii=False))
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_model(model, tokenizer, output_dir / f"checkpoint-{global_step}", global_step, last_metrics)
            if global_step >= total_steps:
                break

    final_metrics = last_metrics
    if dev_loader is not None:
        final_metrics = {"train_last": last_metrics, "eval": evaluate(model, dev_loader, device, args.temperature, args.seg_loss_weight)}
    save_model(model, tokenizer, output_dir, global_step, final_metrics)
    print(json.dumps({"saved": str(output_dir), "global_step": global_step, "metrics": final_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
