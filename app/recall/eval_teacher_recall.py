from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import hnswlib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate encoder recall against teacher labels")
    parser.add_argument("--labels_jsonl", default=str(base / "data" / "processed" / "teacher_recall_20000.jsonl"), type=str)
    parser.add_argument("--index_dir", default=str(base / "index"), type=str)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--ann", default="hnsw", choices=["hnsw", "flat"], type=str)
    parser.add_argument("--ef_search", default=64, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--output_json", default=str(base / "data" / "processed" / "teacher_eval_metrics.json"), type=str)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    all_vecs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            out = model(**encoded)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output
            else:
                vec = mean_pool(out.last_hidden_state, encoded["attention_mask"])
            all_vecs.append(vec.cpu().numpy().astype(np.float32))
    return l2_normalize(np.concatenate(all_vecs, axis=0))


def dcg_at_k(binary_rel: list[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(binary_rel[:k], start=1):
        if rel > 0:
            score += 1.0 / math.log2(i + 1)
    return score


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    index_dir = Path(args.index_dir)
    labels_path = Path(args.labels_jsonl)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    emb = np.load(index_dir / "embeddings.npy").astype(np.float32)
    emb = l2_normalize(emb)
    records = read_jsonl(index_dir / "records.jsonl")
    id_to_row = {}
    for i, row in enumerate(records):
        try:
            rid = int(row["id"])
        except Exception:
            continue
        id_to_row[rid] = i

    labels = read_jsonl(labels_path)
    queries: list[str] = []
    targets: list[set[int]] = []
    dropped_no_valid_target = 0
    for row in labels:
        q = str(row.get("user_query", "")).strip()
        ids = row.get("target_ids", [])
        if not q or not isinstance(ids, list):
            continue
        valid_rows = set()
        for x in ids:
            try:
                rid = int(x)
            except Exception:
                continue
            if rid in id_to_row:
                valid_rows.add(id_to_row[rid])
        if not valid_rows:
            dropped_no_valid_target += 1
            continue
        queries.append(q)
        targets.append(valid_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(meta["model_name_or_path"], trust_remote_code=True)
    model = AutoModel.from_pretrained(meta["model_name_or_path"], trust_remote_code=True).to(device)
    qvecs = encode_texts(queries, tokenizer, model, device, args.batch_size, args.max_length)

    k = min(args.top_k, emb.shape[0])
    if args.ann == "hnsw" and (index_dir / "index_hnsw.bin").exists():
        ann_index = hnswlib.Index(space="cosine", dim=int(meta["dim"]))
        ann_index.load_index(str(index_dir / "index_hnsw.bin"))
        ann_index.set_ef(args.ef_search)
        labels, _distances = ann_index.knn_query(qvecs, k=k)
        topk_sorted = labels
    else:
        scores = qvecs @ emb.T
        topk_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        topk_scores = np.take_along_axis(scores, topk_idx, axis=1)
        order = np.argsort(-topk_scores, axis=1)
        topk_sorted = np.take_along_axis(topk_idx, order, axis=1)

    total = len(queries)
    hit1 = 0
    hit3 = 0
    hit10 = 0
    mrr10 = 0.0
    ndcg10 = 0.0

    for i in range(total):
        pred = topk_sorted[i].tolist()
        gold = targets[i]
        rel = [1 if p in gold else 0 for p in pred[:10]]
        if any(rel[:1]):
            hit1 += 1
        if any(rel[:3]):
            hit3 += 1
        if any(rel[:10]):
            hit10 += 1

        rr = 0.0
        for rank, p in enumerate(pred[:10], start=1):
            if p in gold:
                rr = 1.0 / rank
                break
        mrr10 += rr

        dcg = dcg_at_k(rel, 10)
        ideal_len = min(10, len(gold))
        idcg = dcg_at_k([1] * ideal_len, 10)
        if idcg > 0:
            ndcg10 += dcg / idcg

    metrics = {
        "labels_total": len(labels),
        "eval_total": total,
        "dropped_no_valid_target": dropped_no_valid_target,
        "top_k": int(k),
        "ann": args.ann,
        "ef_search": int(args.ef_search),
        "hit@1": hit1 / total if total else 0.0,
        "hit@3": hit3 / total if total else 0.0,
        "hit@10": hit10 / total if total else 0.0,
        "mrr@10": mrr10 / total if total else 0.0,
        "ndcg@10": ndcg10 / total if total else 0.0,
        "elapsed_s": time.perf_counter() - t0,
        "device": str(device),
        "encoder_model": meta.get("model_name_or_path", ""),
    }
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
