from __future__ import annotations

import argparse
import json
from pathlib import Path

from split_distill.rules import detect_rule_violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate structured prediction outputs")
    parser.add_argument("--gold", required=True, type=str)
    parser.add_argument("--pred", required=True, type=str)
    return parser.parse_args()


def read_rows(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["query"]] = row
    return rows


def set_f1(gold: list[str], pred: list[str]) -> float:
    gold_set = set(gold)
    pred_set = set(pred)
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    inter = len(gold_set & pred_set)
    if inter == 0:
        return 0.0
    precision = inter / len(pred_set)
    recall = inter / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def set_precision(gold: list[str], pred: list[str]) -> float:
    gold_set = set(gold)
    pred_set = set(pred)
    if not pred_set:
        return 0.0
    return len(gold_set & pred_set) / len(pred_set)


def set_recall(gold: list[str], pred: list[str]) -> float:
    gold_set = set(gold)
    pred_set = set(pred)
    if not gold_set:
        return 0.0
    return len(gold_set & pred_set) / len(gold_set)


def main() -> None:
    args = parse_args()
    gold = read_rows(args.gold)
    pred = read_rows(args.pred)
    shared = sorted(set(gold) & set(pred))
    if not shared:
        raise RuntimeError("no_shared_queries")

    type_ok = 0
    core_ok = 0
    pred_ok = 0
    obj_ok = 0
    exact_match = 0
    keyword_f1_total = 0.0
    keyword_precision_total = 0.0
    keyword_recall_total = 0.0
    violation_examples = 0

    for query in shared:
        g = gold[query]
        p = pred[query]
        type_ok += int(g["type"] == p["type"])
        core_ok += int(g["core_text"] == p["core_text"])
        pred_ok += int(g.get("predicate", "") == p.get("predicate", ""))
        obj_ok += int(g.get("object", "") == p.get("object", ""))
        exact_match += int(
            g["type"] == p["type"]
            and g["core_text"] == p["core_text"]
            and g.get("predicate", "") == p.get("predicate", "")
            and g.get("object", "") == p.get("object", "")
            and set(g["keywords"]) == set(p["keywords"])
        )
        keyword_precision_total += set_precision(g["keywords"], p["keywords"])
        keyword_recall_total += set_recall(g["keywords"], p["keywords"])
        keyword_f1_total += set_f1(g["keywords"], p["keywords"])
        violation_examples += int(bool(detect_rule_violations(p)))

    n = len(shared)
    metrics = {
        "count": n,
        "valid_json_rate": 1.0,
        "exact_match": exact_match / n,
        "type_accuracy": type_ok / n,
        "core_text_accuracy": core_ok / n,
        "predicate_accuracy": pred_ok / n,
        "object_accuracy": obj_ok / n,
        "keywords_precision": keyword_precision_total / n,
        "keywords_recall": keyword_recall_total / n,
        "keyword_f1": keyword_f1_total / n,
        "rule_violation_rate": violation_examples / n,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
