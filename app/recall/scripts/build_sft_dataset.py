from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "你是上海话词典匹配助手。给定用户问题和候选词条后，"
    "选出最匹配的最多3个候选ID。输出必须是一个JSON对象："
    '{"ids":[id1,id2,...]}。'
    "如果没有匹配，输出 {\"ids\":[]}。不要输出解释。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT jsonl for top-match ID selection")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            prompt = (
                f"用户想问：{row['user_query']}\n"
                f"top_n={row.get('top_n', 3)}\n"
                f"候选：\n{row['candidate_text']}\n"
                "请返回最匹配ID。"
            )
            assistant = json.dumps({"ids": row.get("target_ids", [])[:3]}, ensure_ascii=False)
            payload = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant},
                ]
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"wrote={dst}")


if __name__ == "__main__":
    main()
