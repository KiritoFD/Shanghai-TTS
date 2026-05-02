from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval utility of normalized keywords")
    parser.add_argument("--queries", required=True, type=str, help="JSONL with query and keywords")
    parser.add_argument("--corpus", required=True, type=str, help="JSONL with doc_id and text")
    parser.add_argument("--qrels", required=True, type=str, help="JSONL with query and relevant_doc_ids")
    parser.add_argument("--k", default=10, type=int)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokenize(text: str) -> list[str]:
    text = re.sub(r"[^\u4e00-\u9fff]", "", text)
    if len(text) <= 1:
        return [text] if text else []
    return [text[i : i + 2] for i in range(len(text) - 1)]


def build_bm25(corpus: list[dict]) -> tuple[list[Counter], dict[str, float], float]:
    doc_tfs: list[Counter] = []
    df: Counter = Counter()
    doc_lengths: list[int] = []
    for row in corpus:
        tokens = tokenize(row["text"])
        tf = Counter(tokens)
        doc_tfs.append(tf)
        doc_lengths.append(sum(tf.values()))
        for token in tf:
            df[token] += 1
    n_docs = max(1, len(corpus))
    idf = {token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()}
    avg_dl = sum(doc_lengths) / max(1, len(doc_lengths))
    return doc_tfs, idf, avg_dl


def bm25_rank(query_text: str, corpus: list[dict], doc_tfs: list[Counter], idf: dict[str, float], avg_dl: float, k: int) -> list[str]:
    q_tokens = tokenize(query_text)
    scores: list[tuple[float, str]] = []
    k1 = 1.5
    b = 0.75
    for row, tf in zip(corpus, doc_tfs):
        dl = sum(tf.values())
        score = 0.0
        for token in q_tokens:
            freq = tf.get(token, 0)
            if freq == 0:
                continue
            token_idf = idf.get(token, 0.0)
            denom = freq + k1 * (1 - b + b * dl / max(1e-9, avg_dl))
            score += token_idf * (freq * (k1 + 1) / denom)
        scores.append((score, row["doc_id"]))
    scores.sort(key=lambda item: item[0], reverse=True)
    return [doc_id for _, doc_id in scores[:k]]


def recall_at_k(relevant: set[str], ranked: list[str]) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(ranked)) / len(relevant)


def mrr_at_k(relevant: set[str], ranked: list[str]) -> float:
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevant: set[str], ranked: list[str]) -> float:
    dcg = 0.0
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant), len(ranked))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def main() -> None:
    args = parse_args()
    queries = read_jsonl(args.queries)
    corpus = read_jsonl(args.corpus)
    qrels = {row["query"]: set(row["relevant_doc_ids"]) for row in read_jsonl(args.qrels)}

    doc_tfs, idf, avg_dl = build_bm25(corpus)

    recall_total = 0.0
    ndcg_total = 0.0
    mrr_total = 0.0
    count = 0

    for row in queries:
        query = row["query"]
        if query not in qrels:
            continue
        relevant = qrels[query]
        normalized_query = ",".join(row.get("keywords", [])) or row.get("core_text", query)
        ranked = bm25_rank(normalized_query, corpus, doc_tfs, idf, avg_dl, args.k)
        recall_total += recall_at_k(relevant, ranked)
        ndcg_total += ndcg_at_k(relevant, ranked)
        mrr_total += mrr_at_k(relevant, ranked)
        count += 1

    if count == 0:
        raise RuntimeError("no_overlapping_queries_for_retrieval_eval")

    metrics = {
        "count": count,
        f"recall@{args.k}": recall_total / count,
        f"ndcg@{args.k}": ndcg_total / count,
        f"mrr@{args.k}": mrr_total / count,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
