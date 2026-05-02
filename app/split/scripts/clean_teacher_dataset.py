from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from split_distill.rules import clean_prediction, parse_teacher_json, render_final_output

HARDCASE_OVERRIDES = {
    "没用": {"core_text": "没用", "type": "词项", "predicate": "", "object": "", "keywords": ["没用", "无用", "不中用"]},
    "开心": {"core_text": "开心", "type": "词项", "predicate": "", "object": "", "keywords": ["开心", "高兴", "快乐"]},
    "夸人帅怎么说": {"core_text": "夸人帅", "type": "动作短语", "predicate": "夸", "object": "帅", "keywords": ["夸", "帅"]},
    "很累怎么说": {"core_text": "很累", "type": "词项", "predicate": "", "object": "", "keywords": ["累", "疲惫", "疲劳"]},
}


def _apply_hardcase_overrides(row: dict) -> dict:
    query = str(row.get("query", "")).strip()
    if query in HARDCASE_OVERRIDES:
        fixed = HARDCASE_OVERRIDES[query].copy()
        fixed["query"] = query
        return fixed

    if re.search(r"夸.*帅.*怎么(说|讲|表达)", query):
        return {
            "query": query,
            "core_text": "夸人帅",
            "type": "动作短语",
            "predicate": "夸",
            "object": "帅",
            "keywords": ["夸", "帅"],
        }
    if re.search(r"(很)?累.*怎么(说|讲|表达)", query):
        return {
            "query": query,
            "core_text": "很累",
            "type": "词项",
            "predicate": "",
            "object": "",
            "keywords": ["累", "疲惫", "疲劳"],
        }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean teacher outputs into validated structured data")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--reject_output", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    reject_path = Path(args.reject_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.input).open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as ok_handle, reject_path.open("w", encoding="utf-8") as reject_handle:
        for line in src:
            row = json.loads(line)
            try:
                payload = parse_teacher_json(row["raw_response"])
                if isinstance(payload, dict):
                    payload = [payload]
            except Exception as exc:
                payload = None
                batch_issues = [f"parse_error:{exc}"]

            if payload is None or not isinstance(payload, list):
                reject_handle.write(
                    json.dumps(
                        {
                            "batch_queries": row.get("batch_queries", []),
                            "issues": batch_issues if payload is None else ["payload_not_list"],
                            "raw_response": row.get("raw_response", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            source_map = {
                query: source
                for query, source in zip(row.get("batch_queries", []), row.get("batch_sources", []))
            }

            for item in payload:
                query = str(item.get("query", "")).strip()
                if not query:
                    reject_handle.write(
                        json.dumps(
                            {
                                "query": "",
                                "issues": ["missing_query_in_batch_item"],
                                "raw_item": item,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                example, issues = clean_prediction(query, item)
                if example is None:
                    reject_handle.write(
                        json.dumps(
                            {
                                "query": query,
                                "issues": issues,
                                "raw_item": item,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                clean_row = example.to_dict()
                clean_row = _apply_hardcase_overrides(clean_row)
                clean_row["source"] = source_map.get(query, "")
                if issues:
                    clean_row["warnings"] = issues
                clean_row["final_output"] = render_final_output(example)
                ok_handle.write(json.dumps(clean_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
