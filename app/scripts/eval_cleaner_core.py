from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cleaner_model import CleanerCoreExtractor
from rag_mt import clean_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG-MT cleaner on split held-out data.")
    parser.add_argument("--input_jsonl", default="app/split/data/processed/testset_clean_360.jsonl", type=str)
    parser.add_argument("--checkpoint_dir", default="app/cleaner_core/outputs/bilstm_core", type=str)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--output_json", default="app/cleaner_core/outputs/cleaner_eval.json", type=str)
    return parser.parse_args()


def read_rows(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def f1(gold: list[str], pred: list[str]) -> float:
    g = {str(x).strip() for x in gold if str(x).strip()}
    p = {str(x).strip() for x in pred if str(x).strip()}
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    inter = len(g & p)
    prec = inter / len(p)
    rec = inter / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input_jsonl), args.limit)
    extractor = CleanerCoreExtractor(Path(args.checkpoint_dir))

    metrics = {
        "n": len(rows),
        "model_core_em": 0,
        "rules_core_em": 0,
        "model_keywords_f1": 0.0,
        "rules_keywords_f1": 0.0,
        "model_avg_ms": 0.0,
        "rules_avg_ms": 0.0,
    }
    bad: list[dict] = []
    for row in rows:
        query = str(row.get("query", "")).strip()
        gold_core = str(row.get("core_text", "")).strip()
        gold_keywords = row.get("keywords", [])
        if not isinstance(gold_keywords, list):
            gold_keywords = []

        ts = time.perf_counter()
        model_pred = extractor.predict(query).as_parsed()
        metrics["model_avg_ms"] += (time.perf_counter() - ts) * 1000

        ts = time.perf_counter()
        rules_pred = clean_query(query).as_parsed()
        metrics["rules_avg_ms"] += (time.perf_counter() - ts) * 1000

        metrics["model_core_em"] += int(model_pred["core_text"] == gold_core)
        metrics["rules_core_em"] += int(rules_pred["core_text"] == gold_core)
        metrics["model_keywords_f1"] += f1(gold_keywords, model_pred.get("keywords", []))
        metrics["rules_keywords_f1"] += f1(gold_keywords, rules_pred.get("keywords", []))

        if model_pred["core_text"] != gold_core and len(bad) < 20:
            bad.append({"query": query, "gold": gold_core, "pred": model_pred["core_text"], "keywords": model_pred.get("keywords", [])})

    n = max(len(rows), 1)
    for key in ("model_core_em", "rules_core_em", "model_keywords_f1", "rules_keywords_f1", "model_avg_ms", "rules_avg_ms"):
        metrics[key] = round(metrics[key] / n, 4)
    report = {"metrics": metrics, "bad_examples": bad}
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
