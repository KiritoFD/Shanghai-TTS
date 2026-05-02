from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supervision dataset for top-match ID selection task")
    parser.add_argument("--dict_csv", required=True, type=str)
    parser.add_argument("--out_jsonl", required=True, type=str)
    parser.add_argument("--id_col", default="0", type=str, help="ID column name or index")
    parser.add_argument("--sh_col", default="1", type=str, help="Shanghai phrase column name or index")
    parser.add_argument("--def_col", default="4", type=str, help="Definition column name or index (your snippet uses col 4)")
    parser.add_argument("--num_samples", default=12000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def clean_text(x: object) -> str:
    return str(x).strip()


def render_candidates(cands: list[dict]) -> str:
    return "\n".join([f"ID:{row['id']} | 上海话:{row['shanghai']} | 释义:{row['definition']}" for row in cands])


def pick_col(df: pd.DataFrame, key: str):
    if key in df.columns:
        return df[key]
    if key.isdigit():
        idx = int(key)
        if 0 <= idx < len(df.columns):
            return df.iloc[:, idx]
    raise KeyError(f"column not found: {key}")


def extract_keywords_for_regex(text: str, k: int = 2) -> list[str]:
    tokens = re.split(r"[，。、“”‘’；：？！、,\s/()（）]+", str(text))
    tokens = [t.strip() for t in tokens if len(t.strip()) >= 2]
    if not tokens:
        text = re.sub(r"\s+", "", str(text))
        return [text] if text else []
    random.shuffle(tokens)
    return tokens[:k]


def regex_candidates(df: pd.DataFrame, keywords: list[str], def_col: str = "definition", max_candidates: int = 20) -> pd.DataFrame:
    escaped_keywords = [re.escape(kw) for kw in keywords if kw]
    if not escaped_keywords:
        return df.head(max_candidates).copy()
    pattern = "|".join(escaped_keywords)
    candidates = df[df[def_col].astype(str).str.contains(pattern, na=False)].copy()
    if candidates.empty:
        return candidates
    if len(candidates) > max_candidates:
        candidates = candidates.head(max_candidates).copy()
    return candidates


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    df_raw = pd.read_csv(args.dict_csv)
    df = pd.DataFrame(
        {
            "id": pick_col(df_raw, args.id_col),
            "shanghai": pick_col(df_raw, args.sh_col),
            "definition": pick_col(df_raw, args.def_col),
        }
    )
    df["id"] = df["id"].astype(int)
    df["shanghai"] = df["shanghai"].map(clean_text)
    df["definition"] = df["definition"].map(clean_text)
    df = df[(df["shanghai"] != "") & (df["definition"] != "")]
    records = df.to_dict("records")
    if len(records) < 20:
        raise ValueError("dictionary rows too few")

    out_rows: list[dict] = []
    hard_case = {
        "user_query": "你好",
        "top_n": 3,
        "candidates": [
            {"id": 8439, "shanghai": "侬好", "definition": "互致问候时用，对应“你好”"},
            {"id": 8440, "shanghai": "饭吃过𠲎", "definition": "旧时见面问候语（吃过饭没有）"},
            {"id": 8442, "shanghai": "侬好辣𠲎", "definition": "旧时“你好着吗？”问候语"},
            {"id": 10499, "shanghai": "勿谈了", "definition": "称赞话，不是问候语"},
            {"id": 14001, "shanghai": "熬好戏", "definition": "要你好看，威胁语气"},
            {"id": 14435, "shanghai": "油盐勿进", "definition": "不听劝"},
        ],
        "target_ids": [8439, 8440, 8442],
        "source": "hard_case_hello",
    }
    out_rows.append(hard_case)

    for _ in range(args.num_samples):
        gold = random.choice(records)
        user_query = random.choice([gold["definition"], gold["shanghai"]])

        # 先模拟你线上逻辑：在释义列做关键词正则召回（最多20条）
        keywords = extract_keywords_for_regex(user_query, k=2)
        cands_df = regex_candidates(df, keywords=keywords, def_col="definition", max_candidates=20)

        if cands_df.empty:
            negatives = random.sample(records, k=min(9, len(records)))
            candidates = negatives
        else:
            candidates = cands_df.to_dict("records")

        # 保护：如果 gold 不在召回候选，补进来，确保有监督信号
        if int(gold["id"]) not in {int(x["id"]) for x in candidates}:
            candidates = [gold] + candidates
        candidates = candidates[:20]
        random.shuffle(candidates)

        out_rows.append(
            {
                "user_query": user_query,
                "top_n": 3,
                "candidates": candidates,
                "target_ids": [int(gold["id"])],
                "keywords": keywords,
                "source": "auto_from_dict",
            }
        )

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            row["candidate_text"] = render_candidates(row["candidates"])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote={out_path}")
    print(f"rows={len(out_rows)}")


if __name__ == "__main__":
    main()
