from __future__ import annotations

import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "split"
DST = ROOT / "train"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_by_query(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    output: list[dict] = []
    for row in rows:
        query = str(row.get("query", "")).strip()
        if not query or query in seen:
            continue
        seen.add(query)
        output.append(row)
    return output


def main() -> None:
    raw_dst = DST / "data" / "raw"
    processed_dst = DST / "data" / "processed"
    raw_dst.mkdir(parents=True, exist_ok=True)
    processed_dst.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        SRC / "data" / "raw" / "query_train_candidates_merged.jsonl",
        SRC / "data" / "raw" / "query_train_teacher_5000.jsonl",
        SRC / "data" / "processed" / "query_train_teacher_5000_clean_all.jsonl",
        SRC / "data" / "processed" / "query_train_teacher_5000_rejects_all.jsonl",
        SRC / "data" / "processed" / "query_real_test_200.jsonl",
        SRC / "data" / "processed" / "hard_cases_manual_4_clean.jsonl",
        SRC / "query_testset_20000.txt",
    ]
    for src_path in files_to_copy:
        if src_path.exists():
            shutil.copy2(src_path, raw_dst / src_path.name)

    candidate_rows = read_jsonl(SRC / "data" / "raw" / "query_train_candidates_merged.jsonl")
    candidate_queries = {str(row["query"]).strip() for row in candidate_rows if str(row.get("query", "")).strip()}
    clean_rows = read_jsonl(SRC / "data" / "processed" / "query_train_teacher_5000_clean_all.jsonl")
    clean_rows = [row for row in clean_rows if str(row.get("query", "")).strip() in candidate_queries]
    clean_rows = dedupe_by_query(clean_rows)

    test_rows = dedupe_by_query(read_jsonl(SRC / "data" / "processed" / "query_real_test_200.jsonl"))
    test_queries = {str(row["query"]).strip() for row in test_rows}

    train_pool = [row for row in clean_rows if str(row["query"]).strip() not in test_queries]
    hard_rows = dedupe_by_query(read_jsonl(SRC / "data" / "processed" / "hard_cases_manual_4_clean.jsonl"))

    train_map = {str(row["query"]).strip(): row for row in train_pool}
    for row in hard_rows:
        query = str(row.get("query", "")).strip()
        if query and query not in test_queries:
            train_map[query] = row
    train_pool = list(train_map.values())

    random.seed(42)
    random.shuffle(train_pool)
    dev_size = min(200, max(100, int(len(train_pool) * 0.05)))
    dev_rows = train_pool[:dev_size]
    train_rows = train_pool[dev_size:]

    hard_manual = [
        {"query": "没用"},
        {"query": "开心"},
        {"query": "夸人帅怎么说"},
        {"query": "很累怎么说"},
    ]
    write_jsonl(processed_dst / "hard_cases_4_queries.jsonl", hard_manual)
    (processed_dst / "hard_cases_4.txt").write_text("\n".join(x["query"] for x in hard_manual) + "\n", encoding="utf-8")

    write_jsonl(processed_dst / "train_clean.jsonl", train_rows)
    write_jsonl(processed_dst / "dev_clean.jsonl", dev_rows)
    write_jsonl(processed_dst / "test_clean_200.jsonl", test_rows)

    train_q = {str(r["query"]).strip() for r in train_rows}
    missing = sorted(candidate_queries - train_q - test_queries)
    (processed_dst / "missing_queries.txt").write_text("\n".join(missing), encoding="utf-8")

    stats = {
        "candidate_queries": len(candidate_queries),
        "clean_unique_from_teacher": len(clean_rows),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
        "missing_queries_after_clean": len(missing),
    }
    (processed_dst / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
