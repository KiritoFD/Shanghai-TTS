from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ADJECTIVES = [
    "开心", "难受", "紧张", "尴尬", "高兴", "伤心", "舒服", "奇怪", "普通", "特殊",
    "方便", "麻烦", "安全", "危险", "重要", "合适", "直接", "稳定", "明显", "认真",
    "随便", "礼貌", "安静", "热闹", "聪明", "勇敢", "冷静", "着急", "顺利", "糟糕",
]

VERBS = [
    "寻找", "学习", "表达", "请求", "安排", "记录", "准备", "比较", "保存", "删除",
    "恢复", "检查", "修复", "更新", "分析", "提供", "查找", "购买", "预约", "发送",
    "回复", "处理", "练习", "说明", "展示", "总结", "上传", "下载", "整理", "提醒",
    "照顾", "支持", "安慰", "感谢", "喜欢", "理解", "确认", "联系", "申请", "取消",
]

OBJECTS = [
    "帮助", "工作", "资料", "规则", "时间", "地址", "结果", "原因", "版本", "建议",
    "商品", "消息", "邮件", "投诉", "语法", "单词", "发音", "想法", "情况", "合作",
    "成果", "重点", "文件", "方向", "朋友", "老师", "电影", "英语", "意思", "需求",
    "订单", "申请", "支持", "客户", "内容", "价格", "数据", "房间", "记录", "信息",
]

PRONOUN_OBJECTS = ["我", "你", "他", "她", "它"]

WORD_SHELLS = [
    "{x}",
    "{x}怎么说",
    "{x}怎么讲",
    "{x}怎么表达",
    "{x}是什么意思",
    "上海话里{x}怎么说",
    "用上海话{x}怎么讲",
    "{x}这个词怎么讲",
]

PHRASE_SHELLS = [
    "{v}{o}",
    "{v}{o}怎么说",
    "{v}{o}怎么讲",
    "{v}{o}怎么表达",
    "上海话里{v}{o}怎么说",
    "用上海话{v}{o}怎么讲",
    "我想{v}{o}",
    "我要{v}{o}",
    "怎么{v}{o}",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate larger compositional query pool")
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--count", default=6000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    rows: list[dict] = []

    while len(rows) < args.count:
        if random.random() < 0.42:
            x = random.choice(ADJECTIVES)
            query = random.choice(WORD_SHELLS).format(x=x)
            source = "compositional_word"
        else:
            v = random.choice(VERBS)
            if random.random() < 0.25:
                o = random.choice(PRONOUN_OBJECTS)
            else:
                o = random.choice(OBJECTS)
            query = random.choice(PHRASE_SHELLS).format(v=v, o=o)
            source = "compositional_phrase"
        if query in seen:
            continue
        seen.add(query)
        rows.append({"query": query, "source": source})

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
