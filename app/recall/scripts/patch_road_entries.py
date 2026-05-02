"""Patch road-name entries (CSV col-6 non-empty rows) into an existing vector index.

The road entries are in processed_results.csv rows 21021-21537 (0-indexed after
header). They were omitted from the original index build because col-1 (entry_alt)
is empty, causing build_vector_index.py to drop them.

This script:
  1. Reads the CSV, finds all rows where col-6 (notes) is non-empty.
  2. Skips any entry already present in records.jsonl (by headword match).
  3. Encodes each new entry with the same bge-m3 model used to build the index.
  4. Appends new records to records.jsonl and new vectors to embeddings.npy.
  5. Rebuilds the HNSW index from the combined embeddings.
  6. Updates meta.json row count.

Usage
-----
    uv run python app/recall/scripts/patch_road_entries.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "app" / "recall"))

from model_utils import ensure_model_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch road entries into existing recall index")
    p.add_argument("--csv", default="app/recall/processed_results.csv")
    p.add_argument("--index_dir", default="app/recall/index_local_bge_m3")
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--max_length", default=256, type=int)
    return p.parse_args()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    all_vecs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(device)
            out = model(**enc)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output
            else:
                vec = mean_pool(out.last_hidden_state, enc["attention_mask"])
            all_vecs.append(vec.cpu().numpy().astype(np.float32))
            print(f"  encoded {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.concatenate(all_vecs, axis=0)


def main() -> None:
    args = parse_args()
    csv_path = REPO_ROOT / args.csv
    index_dir = REPO_ROOT / args.index_dir

    # ── load existing index ───────────────────────────────────────────────────
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    existing_vecs = np.load(index_dir / "embeddings.npy").astype(np.float32)
    existing_records: list[dict] = []
    with (index_dir / "records.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                existing_records.append(json.loads(line))

    print(f"Existing index: {len(existing_records)} records, dim={existing_vecs.shape[1]}")

    # Build set of existing headwords (strip 【】) for dedup
    existing_headwords: set[str] = set()
    for rec in existing_records:
        hw = str(rec.get("shanghai", "")).strip().strip("【】[] ").strip()
        if hw:
            existing_headwords.add(hw)

    # ── read CSV, find road entries ───────────────────────────────────────────
    road_entries: list[dict] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if len(row) < 7:
                continue
            notes = row[6].strip()
            if not notes:
                continue
            entry = row[0].strip().strip("\ufeff").strip("【】[] ").strip()
            if not entry or entry in ("entry", "entry_alt"):  # skip header
                continue
            if entry in existing_headwords:
                print(f"  skip (already indexed): {entry}")
                continue
            # Use col-4 as display text (same as col-0 for road names)
            display = row[4].strip() or entry
            # Build index text: entry name + notes (truncated) as context
            index_text = f"{entry}。{notes[:200]}"
            road_entries.append({
                "entry": entry,
                "display": display,
                "notes": notes,
                "index_text": index_text,
            })

    print(f"New road entries to add: {len(road_entries)}")
    if not road_entries:
        print("Nothing to patch — all road entries already in index.")
        return

    # ── load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = str(meta["model_name_or_path"])
    model_path = ensure_model_path(model_name, index_dir)
    print(f"Loading model from: {model_path}  device={device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

    # ── encode new entries ────────────────────────────────────────────────────
    texts = [e["index_text"] for e in road_entries]
    print(f"Encoding {len(texts)} road entries ...")
    new_vecs = encode_texts(texts, tokenizer, model, args.batch_size, args.max_length, device)
    new_vecs = l2_normalize(new_vecs).astype(np.float32)
    print(f"Encoded shape: {new_vecs.shape}")

    # ── build new record dicts ────────────────────────────────────────────────
    next_id = max(int(r["id"]) for r in existing_records) + 1
    new_records: list[dict] = []
    for i, entry in enumerate(road_entries):
        new_records.append({
            "id": next_id + i,
            "shanghai": entry["display"],
            "definition": entry["notes"],
            "index_text": entry["index_text"],
        })

    # ── append to records.jsonl ───────────────────────────────────────────────
    with (index_dir / "records.jsonl").open("a", encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Appended {len(new_records)} records to records.jsonl")

    # ── append to embeddings.npy ──────────────────────────────────────────────
    combined_vecs = np.concatenate([existing_vecs, new_vecs], axis=0)
    np.save(index_dir / "embeddings.npy", combined_vecs)
    print(f"Saved embeddings: {existing_vecs.shape[0]} + {new_vecs.shape[0]} = {combined_vecs.shape[0]} rows")

    # ── rebuild HNSW index ────────────────────────────────────────────────────
    hnsw_path = index_dir / "index_hnsw.bin"
    if hnsw_path.exists():
        try:
            import hnswlib
            dim = combined_vecs.shape[1]
            idx = hnswlib.Index(space="cosine", dim=dim)
            idx.init_index(max_elements=combined_vecs.shape[0] + 2000, ef_construction=200, M=16)
            idx.add_items(combined_vecs, list(range(combined_vecs.shape[0])))
            idx.save_index(str(hnsw_path))
            print(f"Rebuilt HNSW index: {combined_vecs.shape[0]} vectors")
        except ImportError:
            print("hnswlib not available — skipping HNSW rebuild (flat search will be used)")

    # ── update meta.json ─────────────────────────────────────────────────────
    meta["rows"] = int(combined_vecs.shape[0])
    (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated meta.json: rows={meta['rows']}")
    print("Done.")


if __name__ == "__main__":
    main()
