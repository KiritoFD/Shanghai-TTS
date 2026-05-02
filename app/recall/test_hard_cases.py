from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from infer import encode_query, l2_normalize, load_records, retrieve


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Batch test hard cases with encoder recall")
    parser.add_argument("--cases_file", default=str(base / "tests" / "hard_cases.txt"), type=str)
    parser.add_argument("--index_dir", default=str(base / "index"), type=str)
    parser.add_argument("--top_n", default=3, type=int)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--output_json", default=str(base / "tests" / "hard_cases_results.json"), type=str)
    return parser.parse_args()


def read_cases(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cleaned = []
    for x in lines:
        s = x.replace("\ufeff", "").strip()
        if s:
            cleaned.append(s)
    return cleaned


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)
    cases_path = Path(args.cases_file)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    records = load_records(index_dir / "records.jsonl")
    matrix = np.load(index_dir / "embeddings.npy").astype(np.float32)
    matrix = l2_normalize(matrix)
    queries = read_cases(cases_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(meta["model_name_or_path"], trust_remote_code=True)
    model = AutoModel.from_pretrained(meta["model_name_or_path"], trust_remote_code=True).to(device).eval()

    rows: list[dict] = []
    total_s = 0.0

    for q in queries:
        t0 = time.perf_counter()
        qv = encode_query(q, tokenizer, model, device)
        result = retrieve(q, qv, matrix, records, args.top_k, args.top_n)
        infer_s = time.perf_counter() - t0
        total_s += infer_s
        result["infer_s"] = round(infer_s, 4)
        rows.append(result)

        print(f"query={q} infer_s={infer_s:.3f}")
        for hit in result["results"]:
            print(f"  output:【{hit['shanghai']}】 score={hit['score']:.4f}")
        print()

    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    avg = total_s / max(1, len(rows))
    print(f"done. queries={len(rows)} total_s={total_s:.3f} avg_s={avg:.3f}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
