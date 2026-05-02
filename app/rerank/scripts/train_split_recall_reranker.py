from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "recall"))

import app as app_mod  # noqa: E402
from recall.engine import RecallEngine  # noqa: E402
from rerank.features import build_feature_row, sanitize_headword  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train split-aware reranker on recall top-k candidates")
    parser.add_argument("--train_jsonl", default="app/joint/data_app/train.jsonl", type=str)
    parser.add_argument("--dev_jsonl", default="app/joint/data_app/dev.jsonl", type=str)
    parser.add_argument("--output_dir", default="app/rerank/outputs/hgb_split_recall", type=str)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--limit_train", default=12000, type=int)
    parser.add_argument("--limit_dev", default=2000, type=int)
    parser.add_argument("--backend", default="qwen_lora", choices=["qwen_lora", "transformers", "llama_cpp"], type=str)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def is_positive_candidate(row: dict[str, Any], candidate: dict[str, Any]) -> bool:
    gold_headword = sanitize_headword(row.get("headword", ""))
    candidate_headword = sanitize_headword(candidate.get("shanghai", ""))
    if gold_headword and candidate_headword and gold_headword == candidate_headword:
        return True
    try:
        target_id = int(str(row.get("positive_doc_id", "")).split(":")[-1])
        candidate_id = int(candidate.get("id", -1))
        return target_id == candidate_id
    except Exception:
        return False


def build_matrix(rows: list[dict], preprocessor, engine: RecallEngine, top_k: int) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    feature_names: list[str] | None = None
    kept_queries = 0
    for idx, row in enumerate(rows, start=1):
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        parsed = preprocessor.preprocess(query)
        terms = app_mod.build_search_terms(query, parsed)
        result = engine.search(query, top_k=top_k, top_n=top_k, extra_variants=terms)

        local_x: list[list[float]] = []
        local_y: list[int] = []
        for rank, candidate in enumerate(result.get("results", []), start=1):
            feature_row = build_feature_row(query, parsed, candidate, rank)
            if feature_names is None:
                feature_names = feature_row.names
            local_x.append(feature_row.values)
            local_y.append(1 if is_positive_candidate(row, candidate) else 0)

        if sum(local_y) == 0:
            continue

        x_rows.extend(local_x)
        y_rows.extend(local_y)
        kept_queries += 1
        if idx % 500 == 0:
            print(json.dumps({"processed": idx, "kept_queries": kept_queries, "pairs": len(x_rows)}, ensure_ascii=False), flush=True)

    if not x_rows:
        raise RuntimeError("no training pairs built")
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64), list(feature_names or []), kept_queries


def eval_hitk(rows: list[dict], preprocessor, engine: RecallEngine, model, top_k: int) -> dict[str, float]:
    hit1 = 0
    hit3 = 0
    hit10 = 0
    raw_hit1 = 0
    total = 0
    for row in rows:
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        parsed = preprocessor.preprocess(query)
        terms = app_mod.build_search_terms(query, parsed)
        result = engine.search(query, top_k=top_k, top_n=top_k, extra_variants=terms)
        candidates = result.get("results", [])
        if not candidates:
            continue
        raw_hit1 += int(is_positive_candidate(row, candidates[0]))
        x = np.asarray([build_feature_row(query, parsed, c, rank + 1).values for rank, c in enumerate(candidates)], dtype=np.float32)
        scores = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x)
        ranked = [candidate for _, candidate in sorted(zip(scores.tolist(), candidates), key=lambda pair: pair[0], reverse=True)]
        hit1 += int(any(is_positive_candidate(row, candidate) for candidate in ranked[:1]))
        hit3 += int(any(is_positive_candidate(row, candidate) for candidate in ranked[:3]))
        hit10 += int(any(is_positive_candidate(row, candidate) for candidate in ranked[:10]))
        total += 1
    total = max(total, 1)
    return {
        "count": total,
        "raw_hit@1": raw_hit1 / total,
        "rerank_hit@1": hit1 / total,
        "rerank_hit@3": hit3 / total,
        "rerank_hit@10": hit10 / total,
    }


def main() -> None:
    args = parse_args()
    app_mod.recall_manager.ensure_running = lambda: False
    app_mod.PREPROCESSOR_BACKEND = args.backend
    app_mod.preprocessor = None
    preprocessor = app_mod.ensure_preprocessor()
    engine = RecallEngine(index_dir=app_mod.RECALL_INDEX_DIR, ann="hnsw", ef_search=64)

    train_rows = read_jsonl(Path(args.train_jsonl), args.limit_train)
    dev_rows = read_jsonl(Path(args.dev_jsonl), args.limit_dev)
    x_train, y_train, feature_names, kept_train = build_matrix(train_rows, preprocessor, engine, args.top_k)
    x_dev, y_dev, _, kept_dev = build_matrix(dev_rows, preprocessor, engine, args.top_k)

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=5,
        max_iter=250,
        min_samples_leaf=40,
        random_state=42,
    )
    model.fit(x_train, y_train)

    dev_scores = model.predict_proba(x_dev)[:, 1]
    auc = roc_auc_score(y_dev, dev_scores) if len(np.unique(y_dev)) > 1 else 0.0
    hit_metrics = eval_hitk(dev_rows[:kept_dev], preprocessor, engine, model, args.top_k)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    meta = {
        "backend": args.backend,
        "top_k": args.top_k,
        "feature_names": feature_names,
        "train_pairs": int(len(x_train)),
        "dev_pairs": int(len(x_dev)),
        "train_queries_kept": kept_train,
        "dev_queries_kept": kept_dev,
        "dev_auc": round(float(auc), 6),
        "dev_metrics": {k: round(float(v), 6) for k, v in hit_metrics.items()},
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
