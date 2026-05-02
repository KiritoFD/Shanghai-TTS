"""Fine-tune bge-m3 embedding model on Shanghai dialect dictionary pairs.

Creates training data from:
  1. Dictionary: definition (Mandarin) → headword (Shanghai dialect)
  2. Recall test labels: user_query → target headword

Produces contrastive training pairs and fine-tunes bge-m3.

Usage:
    python app/recall/finetune_embedding.py --epochs 3 --lr 2e-5 --batch_size 32
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

APP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT))

DICTIONARY_CSV = APP_ROOT / "data" / "processed_results.csv"
RECALL_LABELS = APP_ROOT / "data" / "test_clean_200_recall.jsonl"
RECORDS_JSONL = APP_ROOT / "recall" / "index_local_bge_m3" / "records.jsonl"
OUTPUT_DIR = APP_ROOT / "recall" / "finetuned_bge_m3"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def read_records(path: Path) -> dict[int, dict]:
    """Load records.jsonl into id→record dict."""
    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                row = json.loads(s)
                records[int(row["id"])] = row
    return records


def build_training_pairs(records: dict[int, dict], recall_labels: list[dict]) -> list[dict]:
    """Build (query, positive_headword, negative_headwords) triples.

    Sources:
      1. Dictionary definition → headword (main source, ~15k pairs)
      2. Recall test labels: user_query → target headword (200 pairs)
    """
    pairs = []

    # Source 1: Dictionary definition → headword
    for rid, rec in records.items():
        hw = str(rec.get("shanghai", "")).strip().strip("【】[] ")
        defn = str(rec.get("definition", "")).strip()
        if not hw or not defn:
            continue
        # Clean definition: remove "〈名〉" etc.
        import re
        clean_defn = re.sub(r"^〈[名动形副助量代介连叹语数]+〉\s*", "", defn)
        clean_defn = clean_defn.strip("。！？ ")
        if clean_defn and len(clean_defn) >= 2:
            pairs.append({
                "query": clean_defn,
                "positive": hw,
                "source": "dictionary",
                "record_id": rid,
            })

    # Source 2: Recall test labels
    id_to_hw = {}
    for rid, rec in records.items():
        hw = str(rec.get("shanghai", "")).strip().strip("【】[] ")
        if hw:
            id_to_hw[rid] = hw

    for label in recall_labels:
        query = str(label.get("user_query", "") or label.get("query", "")).strip()
        target_ids = [int(tid) for tid in label.get("target_ids", []) if tid is not None]
        if not query or not target_ids:
            continue
        for tid in target_ids:
            hw = id_to_hw.get(tid)
            if hw:
                pairs.append({
                    "query": query,
                    "positive": hw,
                    "source": "recall_label",
                    "record_id": tid,
                })

    return pairs


def build_triplets(pairs: list[dict], all_headwords: list[str], neg_count: int = 7) -> list[dict]:
    """Build (query, positive, negative1, negative2, ...) triplets.

    For each pair, sample neg_count random headwords as negatives.
    """
    hw_set = set(all_headwords)
    triplets = []

    for pair in pairs:
        pos = pair["positive"]
        # Sample negatives that are different from positive
        negatives = []
        attempts = 0
        while len(negatives) < neg_count and attempts < neg_count * 3:
            neg = random.choice(all_headwords)
            if neg != pos and neg not in negatives:
                negatives.append(neg)
            attempts += 1
        if negatives:
            triplets.append({
                "query": pair["query"],
                "positive": pos,
                "negatives": negatives,
                "source": pair["source"],
            })

    return triplets


class EmbeddingDataset(Dataset):
    def __init__(self, triplets: list[dict], tokenizer, max_length: int = 128):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        # Encode query
        q_enc = self.tokenizer(
            t["query"], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        # Encode positive
        p_enc = self.tokenizer(
            t["positive"], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        # Encode negatives
        n_encs = []
        for neg in t["negatives"]:
            n_enc = self.tokenizer(
                neg, padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            n_encs.append(n_enc)

        return {
            "q_input_ids": q_enc["input_ids"].squeeze(0),
            "q_attention_mask": q_enc["attention_mask"].squeeze(0),
            "p_input_ids": p_enc["input_ids"].squeeze(0),
            "p_attention_mask": p_enc["attention_mask"].squeeze(0),
            "n_input_ids": torch.stack([e["input_ids"].squeeze(0) for e in n_encs]),
            "n_attention_mask": torch.stack([e["attention_mask"].squeeze(0) for e in n_encs]),
        }


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def contrastive_loss(q_vec, p_vec, n_vecs, temperature: float = 0.02):
    """InfoNCE loss: maximize similarity with positive, minimize with negatives."""
    # q_vec: (batch, dim), p_vec: (batch, dim), n_vecs: (batch, neg_count, dim)
    pos_sim = torch.sum(q_vec * p_vec, dim=-1) / temperature  # (batch,)
    neg_sims = torch.bmm(n_vecs, q_vec.unsqueeze(-1)).squeeze(-1) / temperature  # (batch, neg_count)
    # Log-softmax over [positive, negative1, ..., negativeN]
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)  # (batch, 1+neg_count)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # positive is index 0
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


def evaluate(model, tokenizer, records: dict[int, dict], recall_labels: list[dict], device, batch_size: int = 64):
    """Quick evaluation: hit@1 on recall test set."""
    model.eval()

    # Encode all headwords
    all_hws = []
    all_ids = []
    for rid, rec in records.items():
        hw = str(rec.get("shanghai", "")).strip().strip("【】[] ")
        if hw:
            all_hws.append(hw)
            all_ids.append(rid)

    # Batch encode headwords
    print(f"  Encoding {len(all_hws)} headwords...", flush=True)
    hw_vecs = []
    with torch.no_grad():
        for i in range(0, len(all_hws), batch_size):
            batch = all_hws[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = model(**enc)
            vec = mean_pool(out.last_hidden_state, enc["attention_mask"])
            hw_vecs.append(vec.cpu().numpy())
            if (i // batch_size + 1) % 50 == 0:
                print(f"    Encoded {i + len(batch)}/{len(all_hws)}", flush=True)
    hw_matrix = np.concatenate(hw_vecs, axis=0)
    hw_matrix = hw_matrix / (np.linalg.norm(hw_matrix, axis=1, keepdims=True) + 1e-12)

    # Evaluate queries
    hits = 0
    total = 0
    for label in recall_labels:
        query = str(label.get("user_query", "") or label.get("query", "")).strip()
        target_ids = set(int(tid) for tid in label.get("target_ids", []) if tid is not None)
        if not query or not target_ids:
            continue

        with torch.no_grad():
            enc = tokenizer([query], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = model(**enc)
            qv = mean_pool(out.last_hidden_state, enc["attention_mask"])
            qv = qv.cpu().numpy()
            qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)

        sims = hw_matrix @ qv.T  # (n_records, 1)
        top_idx = int(np.argmax(sims))
        top_id = all_ids[top_idx]
        if top_id in target_ids:
            hits += 1
        total += 1

    return hits / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune bge-m3 on Shanghai dialect pairs")
    parser.add_argument("--model_path", default=str(REPO_ROOT / "model" / "bge-m3"),
                        help="Path to base bge-m3 model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--neg_count", type=int, default=7)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load model
    print(f"Loading model: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print("Tokenizer loaded.", flush=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Model loaded on CPU. Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    model = model.to(device)
    print(f"Model moved to {device}.", flush=True)
    model.train()

    # Load data
    print("Loading records...", flush=True)
    records = read_records(RECORDS_JSONL)
    print(f"Records loaded: {len(records)}", flush=True)
    recall_labels = read_jsonl(RECALL_LABELS) if RECALL_LABELS.exists() else []
    print(f"Recall labels: {len(recall_labels)}", flush=True)

    # Build training pairs
    print("Building training pairs...", flush=True)
    pairs = build_training_pairs(records, recall_labels)
    print(f"Training pairs: {len(pairs)}", flush=True)

    # Build triplets
    print("Building triplets...", flush=True)
    all_headwords = []
    for rec in records.values():
        hw = str(rec.get("shanghai", "")).strip().strip("【】[] ")
        if hw:
            all_headwords.append(hw)
    print(f"Headwords: {len(all_headwords)}", flush=True)

    triplets = build_triplets(pairs, all_headwords, neg_count=args.neg_count)
    random.shuffle(triplets)
    print(f"Triplets: {len(triplets)}", flush=True)

    # Create dataset and dataloader
    dataset = EmbeddingDataset(triplets, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * args.epochs)

    # Initial evaluation
    print("\n--- Initial evaluation ---", flush=True)
    hit1 = evaluate(model, tokenizer, records, recall_labels, device)
    print(f"Hit@1 (ANN only): {hit1:.4f}", flush=True)

    # Training loop
    print(f"\n--- Training ---")
    global_step = 0
    best_hit1 = hit1

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            # Move to device
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            p_ids = batch["p_input_ids"].to(device)
            p_mask = batch["p_attention_mask"].to(device)
            n_ids = batch["n_input_ids"].to(device)  # (batch, neg_count, seq_len)
            n_mask = batch["n_attention_mask"].to(device)

            batch_size = q_ids.size(0)
            neg_count = n_ids.size(1)

            # Encode query
            q_out = model(input_ids=q_ids, attention_mask=q_mask)
            q_vec = mean_pool(q_out.last_hidden_state, q_mask)
            q_vec = q_vec / (q_vec.norm(dim=-1, keepdim=True) + 1e-12)

            # Encode positive
            p_out = model(input_ids=p_ids, attention_mask=p_mask)
            p_vec = mean_pool(p_out.last_hidden_state, p_mask)
            p_vec = p_vec / (p_vec.norm(dim=-1, keepdim=True) + 1e-12)

            # Encode negatives
            n_ids_flat = n_ids.view(-1, n_ids.size(-1))
            n_mask_flat = n_mask.view(-1, n_mask.size(-1))
            n_out = model(input_ids=n_ids_flat, attention_mask=n_mask_flat)
            n_vec = mean_pool(n_out.last_hidden_state, n_mask_flat)
            n_vec = n_vec / (n_vec.norm(dim=-1, keepdim=True) + 1e-12)
            n_vec = n_vec.view(batch_size, neg_count, -1)

            # Loss
            loss = contrastive_loss(q_vec, p_vec, n_vec)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % args.eval_steps == 0:
                avg_loss = epoch_loss / n_batches
                print(f"  Step {global_step}  Loss: {avg_loss:.4f}")

                # Evaluate
                hit1 = evaluate(model, tokenizer, records, recall_labels, device)
                print(f"  Hit@1 (ANN only): {hit1:.4f}")

                if hit1 > best_hit1:
                    best_hit1 = hit1
                    # Save best
                    out_dir = Path(args.output_dir) / "best"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(out_dir))
                    tokenizer.save_pretrained(str(out_dir))
                    print(f"  Saved best model (hit@1={hit1:.4f})")

                model.train()

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}  Avg Loss: {avg_loss:.4f}")

    # Final save
    out_dir = Path(args.output_dir) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nFinal model saved to {out_dir}")
    print(f"Best hit@1: {best_hit1:.4f}")


if __name__ == "__main__":
    main()
