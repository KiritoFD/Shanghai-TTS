from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import app as app_mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lexicon context recall for RAG-MT.")
    parser.add_argument("--dictionary_csv", default="app/recall/processed_results.csv", type=str)
    parser.add_argument("--limit", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output_json", default="app/rag_mt_data/recall_context_eval.json", type=str)
    return parser.parse_args()


def sanitize_headword(text: str) -> str:
    return str(text or "").strip().strip("[]").strip("【】")


def clean_definition(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"〈[^〉]+〉", "", value)
    value = re.sub(r"﹝[^﹞]+﹞", "", value)
    value = re.split(r"[：:；;。丨◇]", value)[0]
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", value)
    return value


def load_eval_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for idx, row in enumerate(csv.reader(handle)):
            if len(row) < 5:
                continue
            source = clean_definition(row[4])
            target = sanitize_headword(row[0])
            if not source or not target or source == target:
                continue
            if len(source) < 2 or len(source) > 10 or len(target) > 10:
                continue
            rows.append({"id": idx, "source": source, "target": target})
    return rows


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = load_eval_pairs(Path(args.dictionary_csv))
    rng.shuffle(rows)
    rows = rows[: args.limit]

    hits1 = 0
    hits3 = 0
    hits5 = 0
    times: list[float] = []
    bad: list[dict] = []
    for row in rows:
        ts = time.perf_counter()
        exact = app_mod.exact_dictionary_hits(row["source"], top_n=5)
        if exact:
            results = exact
            source = "exact_dictionary"
        else:
            results, source = app_mod.call_recall_service(row["source"], variants=[row["source"]], top_k=20, top_n=5)
        times.append((time.perf_counter() - ts) * 1000)
        preds = [sanitize_headword(item.get("shanghai", "")) for item in results[:5]]
        target = row["target"]
        hits1 += int(target in preds[:1])
        hits3 += int(target in preds[:3])
        hits5 += int(target in preds[:5])
        if target not in preds[:3] and len(bad) < 50:
            bad.append({"source": row["source"], "target": target, "preds": preds, "source_kind": source})

    n = max(len(rows), 1)
    report = {
        "n": len(rows),
        "hit@1": round(hits1 / n, 4),
        "hit@3": round(hits3 / n, 4),
        "hit@5": round(hits5 / n, 4),
        "avg_ms": round(sum(times) / n, 2),
        "bad_examples": bad,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
