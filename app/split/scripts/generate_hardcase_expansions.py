from __future__ import annotations

import argparse
import json
from pathlib import Path


WORD_HARD = [
    "没用", "开心", "很累", "难受", "有意思", "高兴", "不值当", "不太对劲", "挺不错", "不大好",
    "看得出来", "看得起", "想不开", "搞不清", "说不清", "记不住", "蛮开心", "太离谱", "挺靠谱", "挺自然",
]

PHRASE_HARD = [
    "夸人帅", "夸人美", "夸人聪明", "夸人厉害", "夸人勤快", "夸人礼貌",
    "我喜欢你", "我想帮助你", "给他支持", "跟你道歉", "寻找帮助", "表达感谢",
    "请求支持", "解释门禁", "修改资料", "取消订单", "申请退款", "确认时间",
]

SHELLS = [
    "{x}",
    "{x}怎么说",
    "{x}怎么讲",
    "{x}怎么表达",
    "{x}是什么意思",
    "上海话里{x}怎么说",
    "用上海话{x}怎么讲",
    "帮我问下{x}怎么说",
    "口语里{x}怎么讲",
    "正常说{x}",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hard-case expansions for split distillation")
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen: set[str] = set()
    for token in WORD_HARD + PHRASE_HARD:
        for shell in SHELLS:
            query = shell.format(x=token)
            if query in seen:
                continue
            seen.add(query)
            rows.append({"query": query, "source": "hard_expansion"})

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
