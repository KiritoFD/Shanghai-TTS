from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate distill predictions")
    parser.add_argument("--gold_jsonl", required=True, type=str)
    parser.add_argument("--pred_jsonl", required=True, type=str)
    parser.add_argument("--report_json", required=True, type=str)
    return parser.parse_args()


def safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {}


def set_f1(gold: list[str], pred: list[str]) -> tuple[float, float, float]:
    g, p = set(gold), set(pred)
    if not g and not p:
        return 1.0, 1.0, 1.0
    if not g:
        return 0.0, 0.0, 0.0
    inter = len(g & p)
    precision = inter / len(p) if p else 0.0
    recall = inter / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def main() -> None:
    args = parse_args()
    gold_rows = [json.loads(x) for x in Path(args.gold_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]
    pred_rows = [json.loads(x) for x in Path(args.pred_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]
    pred_map = {str(row.get("query", "")).strip(): row for row in pred_rows}

    n = 0
    valid_json = 0
    type_correct = 0
    core_em = 0
    pred_em = 0
    obj_em = 0
    kw_p = 0.0
    kw_r = 0.0
    kw_f1 = 0.0

    for g in gold_rows:
        query = str(g.get("query", "")).strip()
        p = pred_map.get(query)
        if not p:
            continue
        n += 1
        parsed = safe_json_loads(str(p.get("parsed_text", "{}")))
        if parsed:
            valid_json += 1

        if str(parsed.get("type", "")).strip() == str(g.get("type", "")).strip():
            type_correct += 1
        if str(parsed.get("core_text", "")).strip() == str(g.get("core_text", "")).strip():
            core_em += 1
        if str(parsed.get("object", "")).strip() == str(g.get("object", "")).strip():
            obj_em += 1

        g_keywords = [str(x).strip() for x in g.get("keywords", []) if str(x).strip()]
        p_keywords = [str(x).strip() for x in parsed.get("keywords", []) if str(x).strip()]
        p_, r_, f_ = set_f1(g_keywords, p_keywords)
        kw_p += p_
        kw_r += r_
        kw_f1 += f_

        gold_struct = {
            "core_text": str(g.get("core_text", "")).strip(),
            "type": str(g.get("type", "")).strip(),
            "predicate": str(g.get("predicate", "")).strip(),
            "object": str(g.get("object", "")).strip(),
            "keywords": sorted(set(g_keywords)),
        }
        pred_struct = {
            "core_text": str(parsed.get("core_text", "")).strip(),
            "type": str(parsed.get("type", "")).strip(),
            "predicate": str(parsed.get("predicate", "")).strip(),
            "object": str(parsed.get("object", "")).strip(),
            "keywords": sorted(set(p_keywords)),
        }
        if gold_struct == pred_struct:
            pred_em += 1

    denom = max(n, 1)
    report = {
        "covered_samples": n,
        "valid_json_rate": valid_json / denom,
        "exact_match": pred_em / denom,
        "type_accuracy": type_correct / denom,
        "core_text_em": core_em / denom,
        "object_em": obj_em / denom,
        "keywords_precision": kw_p / denom,
        "keywords_recall": kw_r / denom,
        "keywords_f1": kw_f1 / denom,
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
