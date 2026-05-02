from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse teacher raw outputs into clean labeled recall dataset")
    parser.add_argument("--tasks_jsonl", required=True, type=str)
    parser.add_argument("--teacher_jsonl", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--reject_output", required=True, type=str)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows


def extract_json_payload(text: str):
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        return json.loads(m.group(0))
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        return [json.loads(m.group(0))]
    raise ValueError("no_json_payload")


def main() -> None:
    args = parse_args()
    tasks = read_jsonl(args.tasks_jsonl)
    task_map = {str(t["task_id"]): t for t in tasks}
    teacher_rows = read_jsonl(args.teacher_jsonl)

    pred_map: dict[str, list[int]] = {}
    rejects: list[dict] = []

    for row in teacher_rows:
        if "raw_response" not in row:
            rejects.append({"type": "batch_failed", "row": row})
            continue
        try:
            payload = extract_json_payload(str(row["raw_response"]))
            if isinstance(payload, dict):
                payload = [payload]
        except Exception as exc:  # noqa: BLE001
            rejects.append({"type": "parse_error", "error": str(exc), "row": row})
            continue

        for item in payload:
            task_id = str(item.get("task_id", "")).strip()
            if not task_id:
                continue
            ids = item.get("ids", [])
            if not isinstance(ids, list):
                ids = []
            parsed_ids = []
            for x in ids:
                s = str(x).strip()
                if s.isdigit():
                    parsed_ids.append(int(s))
            pred_map[task_id] = parsed_ids

    clean_rows: list[dict] = []
    for task_id, task in task_map.items():
        ids = pred_map.get(task_id, [])
        cand_ids = {int(c["id"]) for c in task.get("candidates", [])}
        top_n = int(task.get("top_n", 3))
        clean_ids = []
        seen = set()
        for x in ids:
            if x in cand_ids and x not in seen:
                seen.add(x)
                clean_ids.append(x)
            if len(clean_ids) >= top_n:
                break
        if not clean_ids:
            rejects.append({"type": "empty_ids", "task_id": task_id})
            continue

        row = dict(task)
        row["target_ids"] = clean_ids
        clean_rows.append(row)

    out = Path(args.output)
    rej = Path(args.reject_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rej.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in clean_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with rej.open("w", encoding="utf-8") as f:
        for row in rejects:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"clean_rows={len(clean_rows)}")
    print(f"reject_rows={len(rejects)}")
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
