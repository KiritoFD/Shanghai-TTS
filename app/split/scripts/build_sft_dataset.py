from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "你是中文查询规范化助手。"
    "给定用户输入后，只输出一个JSON对象，字段必须为："
    "core_text,type,predicate,object,keywords。"
    "type只能是“词项”或“动作短语”。"
    "keywords是中文词列表，不要输出任何解释。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT JSONL from cleaned rows")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assistant = json.dumps(
                {
                    "core_text": row["core_text"],
                    "type": row["type"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "keywords": row["keywords"],
                },
                ensure_ascii=False,
            )
            payload = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"用户输入：{row['query']}"},
                    {"role": "assistant", "content": assistant},
                ]
            }
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
