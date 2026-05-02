from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current Qwen LoRA + recall path on joint xlsx-derived rows")
    parser.add_argument("--input_jsonl", default="app/joint/data/test.jsonl", type=str)
    parser.add_argument("--limit", default=200, type=int)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--top_n", default=10, type=int)
    parser.add_argument("--output_json", default="app/joint/outputs/qwen_lora_recall_eval.json", type=str)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def clean_headword(text: str) -> str:
    value = re.sub(r"[【】\[\]銆愩€慬\s]", "", str(text))
    return value.strip()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    app_dir = repo_root / "app"
    sys.path.insert(0, str(app_dir))
    import app as app_mod  # noqa: WPS433

    rows = read_jsonl(Path(args.input_jsonl), args.limit)
    preprocessor = app_mod.ensure_preprocessor()
    app_mod.recall_manager.ensure_running = lambda: False

    hits = {1: 0, 3: 0, 10: 0}
    records: list[dict] = []
    for index, row in enumerate(rows, start=1):
        query = str(row["query"])
        target = clean_headword(row["headword"])
        parsed = preprocessor.preprocess(query)
        terms = app_mod.build_search_terms(query, parsed)
        results, source = app_mod.call_recall_service(query, variants=terms, top_k=args.top_k, top_n=args.top_n)
        ranked = [clean_headword(item.get("shanghai", "")) for item in results]
        for k in hits:
            hits[k] += int(target in ranked[:k])
        records.append(
            {
                "query": query,
                "target": target,
                "parsed": parsed,
                "terms": terms,
                "ranked": ranked,
                "source": source,
            }
        )
        if index % 20 == 0:
            print(json.dumps({"processed": index, "hit@1_so_far": hits[1] / index}, ensure_ascii=False), flush=True)

    n = max(1, len(rows))
    metrics = {
        "model": "qwen2b_lora_checkpoint_130_plus_existing_recall",
        "count": len(rows),
        "hit@1": hits[1] / n,
        "hit@3": hits[3] / n,
        "hit@10": hits[10] / n,
    }
    output = {"metrics": metrics, "records": records}
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

