from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Jina embedding with task-aware retrieval encoding")
    parser.add_argument("--dict_csv", required=True, type=str)
    parser.add_argument("--labels_jsonl", required=True, type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--id_col", default="0", type=str)
    parser.add_argument("--sh_col", default="1", type=str)
    parser.add_argument("--def_col", default="4", type=str)
    parser.add_argument("--header", default="infer", choices=["infer", "none"], type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--output_json", required=True, type=str)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except Exception:
            continue
    return rows


def pick_col(df: pd.DataFrame, key: str):
    if key in df.columns:
        return df[key]
    if key.isdigit():
        idx = int(key)
        if 0 <= idx < len(df.columns):
            return df.iloc[:, idx]
    raise KeyError(f"column not found: {key}")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def encode_task_aware(model, texts: list[str], batch_size: int, task: str, prompt_name: str) -> np.ndarray:
    all_vecs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vec = model.encode(texts=batch, task=task, prompt_name=prompt_name)
        arr = to_numpy(vec).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        all_vecs.append(arr)
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

    header = None if args.header == "none" else "infer"
    raw = pd.read_csv(args.dict_csv, header=header)
    df = pd.DataFrame(
        {
            "id": pick_col(raw, args.id_col),
            "shanghai": pick_col(raw, args.sh_col),
            "definition": pick_col(raw, args.def_col),
        }
    )
    numeric_id = pd.to_numeric(df["id"], errors="coerce")
    if numeric_id.notna().sum() < max(100, int(len(df) * 0.1)):
        df["id"] = list(range(len(df)))
    else:
        df["id"] = numeric_id.fillna(-1).astype(int)
        df = df[df["id"] >= 0].copy()

    df["shanghai"] = df["shanghai"].fillna("").astype(str).str.strip()
    df["definition"] = df["definition"].fillna("").astype(str).str.strip()
    df = df[(df["shanghai"] != "") & (df["definition"] != "") & (df["shanghai"] != "nan") & (df["definition"] != "nan")]
    df = df.reset_index(drop=True)

    labels = read_jsonl(Path(args.labels_jsonl))
    id_to_row = {int(r["id"]): i for i, r in enumerate(df.to_dict("records"))}

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
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = model.to(device)

    documents = (df["shanghai"] + "。释义：" + df["definition"]).tolist()
    dvecs = encode_task_aware(model, documents, args.batch_size, task="retrieval", prompt_name="document")
    qvecs = encode_task_aware(model, queries, args.batch_size, task="retrieval", prompt_name="query")

    k = min(int(args.top_k), dvecs.shape[0])
    scores = qvecs @ dvecs.T
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
        "hit@1": hit1 / total if total else 0.0,
        "hit@3": hit3 / total if total else 0.0,
        "hit@10": hit10 / total if total else 0.0,
        "mrr@10": mrr10 / total if total else 0.0,
        "ndcg@10": ndcg10 / total if total else 0.0,
        "elapsed_s": time.perf_counter() - t0,
        "device": str(device),
        "encoder_model": args.model_name_or_path,
        "task_aware": True,
        "task": "retrieval",
        "doc_prompt_name": "document",
        "query_prompt_name": "query",
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
