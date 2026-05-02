from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recall teacher tasks from dictionary CSV")
    parser.add_argument("--dict_csv", required=True, type=str)
    parser.add_argument("--out_jsonl", required=True, type=str)
    parser.add_argument("--id_col", default="0", type=str)
    parser.add_argument("--sh_col", default="1", type=str)
    parser.add_argument("--def_col", default="4", type=str)
    parser.add_argument("--num_tasks", default=20000, type=int)
    parser.add_argument("--top_n", default=3, type=int)
    parser.add_argument("--max_candidates", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--header", default="infer", choices=["infer", "none"], type=str)
    return parser.parse_args()


def pick_col(df: pd.DataFrame, key: str):
    if key in df.columns:
        return df[key]
    if key.isdigit():
        idx = int(key)
        if 0 <= idx < len(df.columns):
            return df.iloc[:, idx]
    raise KeyError(f"column not found: {key}")


def keyword_split(text: str) -> list[str]:
    parts = re.split(r"[，。、“”‘’；：？！、,\s/()（）]+", str(text).strip())
    parts = [x.strip() for x in parts if len(x.strip()) >= 2]
    return parts[:4] if parts else [str(text).strip()]


def regex_candidates(df: pd.DataFrame, keywords: list[str], max_candidates: int) -> pd.DataFrame:
    escaped = [re.escape(k) for k in keywords if k]
    if not escaped:
        return df.head(max_candidates).copy()
    pattern = "|".join(escaped)
    cands = df[df["definition"].astype(str).str.contains(pattern, na=False)].copy()
    if cands.empty:
        return cands
    if len(cands) > max_candidates:
        cands = cands.head(max_candidates).copy()
    return cands


def render_candidates(cands: list[dict]) -> str:
    return "\n".join([f"ID:{int(row['id'])} | 上海话:{row['shanghai']} | 释义:{row['definition']}" for row in cands])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    header = None if args.header == "none" else "infer"
    raw = pd.read_csv(args.dict_csv, header=header)
    df = pd.DataFrame(
        {
            "id": pick_col(raw, args.id_col),
            "shanghai": pick_col(raw, args.sh_col),
            "definition": pick_col(raw, args.def_col),
        }
    )
    numeric_id = pd.to_numeric(df["id"], errors="coerce")
    if numeric_id.notna().sum() < max(100, int(len(df) * 0.1)):
        df["id"] = list(range(len(df)))
    else:
        df["id"] = numeric_id
        df = df.dropna(subset=["id"]).copy()
        df["id"] = df["id"].astype(int)
    df["shanghai"] = df["shanghai"].fillna("").astype(str).str.strip()
    df["definition"] = df["definition"].fillna("").astype(str).str.strip()
    df = df[(df["shanghai"] != "") & (df["definition"] != "") & (df["shanghai"] != "nan") & (df["definition"] != "nan")]
    rows = df.to_dict("records")
    if len(rows) < 100:
        raise ValueError("dictionary rows too few for teacher task build")

    out_rows: list[dict] = []
    hard_case = {
        "task_id": "hard_nihao",
        "user_query": "你好",
        "top_n": 3,
        "keywords": ["你好", "您好", "哈喽", "嗨"],
        "candidates": [
            {"id": 8439, "shanghai": "侬好", "definition": "互致问候时用，对应“你好”"},
            {"id": 8440, "shanghai": "饭吃过𠲎", "definition": "旧时见面问候语（吃过饭没有）"},
            {"id": 8442, "shanghai": "侬好辣𠲎", "definition": "旧时“你好着吗？”问候语"},
            {"id": 10499, "shanghai": "勿谈了", "definition": "称赞话，不是问候语"},
            {"id": 14001, "shanghai": "熬好戏", "definition": "要你好看，威胁语气"},
            {"id": 14435, "shanghai": "油盐勿进", "definition": "不听劝"},
        ],
        "source": "hard_case",
    }
    hard_case["candidate_text"] = render_candidates(hard_case["candidates"])
    hard_case["query"] = hard_case["user_query"]
    out_rows.append(hard_case)

    for i in range(args.num_tasks - 1):
        anchor = random.choice(rows)
        user_query = random.choice([anchor["definition"], anchor["shanghai"]]).strip()
        keywords = keyword_split(user_query)
        cands_df = regex_candidates(df, keywords, args.max_candidates)
        if cands_df.empty:
            cands = random.sample(rows, k=min(args.max_candidates, len(rows)))
        else:
            cands = cands_df.to_dict("records")
        if int(anchor["id"]) not in {int(x["id"]) for x in cands}:
            cands = [anchor] + cands
        cands = cands[: args.max_candidates]
        random.shuffle(cands)
        row = {
            "task_id": f"t{i+1}",
            "user_query": user_query,
            "top_n": args.top_n,
            "keywords": keywords,
            "candidates": cands,
            "candidate_text": render_candidates(cands),
            "source": "auto",
            "query": user_query,
        }
        out_rows.append(row)

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote={out}")
    print(f"rows={len(out_rows)}")


if __name__ == "__main__":
    main()
