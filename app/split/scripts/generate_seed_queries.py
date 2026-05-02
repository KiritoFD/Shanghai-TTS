from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


WORD_ITEMS = [
    "开心", "难受", "没用", "有意思", "高兴", "伤心", "尴尬", "紧张",
    "聪明", "勇敢", "麻烦", "方便", "厉害", "生气", "安静", "热闹",
    "舒服", "难看", "漂亮", "便宜", "昂贵", "简单", "复杂", "危险",
    "安全", "努力", "懒惰", "可靠", "真实", "虚假", "及时", "突然",
    "稳定", "明显", "模糊", "合适", "奇怪", "正常", "特殊", "重要",
    "普通", "直接", "间接", "温柔", "冷静", "着急", "慌张", "耐心",
    "礼貌", "粗心", "细心", "大方", "小气", "随便", "认真", "优秀",
    "一般", "顺利", "倒霉", "幸运", "失败", "成功", "糟糕", "不错",
]

PHRASE_ITEMS = [
    "寻找帮助", "学习编程", "喜欢音乐", "表达感谢", "请求支持", "寻找工作",
    "学习上海话", "理解规则", "整理资料", "分享经验", "解决问题", "表达喜欢",
    "申请退款", "取消订单", "提交申请", "修改资料", "确认时间", "联系客户",
    "安排会议", "记录内容", "准备材料", "阅读说明", "比较价格", "保存文件",
    "删除记录", "恢复数据", "检查结果", "修复错误", "更新版本", "分析原因",
    "表达歉意", "提供建议", "查找信息", "确认地址", "购买商品", "预约时间",
    "发送消息", "回复邮件", "处理投诉", "学习语法", "记住单词", "练习发音",
    "表达想法", "说明情况", "寻求合作", "展示成果", "总结重点", "上传文件",
    "下载资料", "整理房间", "寻找方向", "提醒别人", "照顾孩子", "支持朋友",
    "安慰别人", "感谢老师", "喜欢电影", "学习英语", "理解意思", "表达需求",
]

SHELLS = [
    "{x}",
    "{x}怎么说",
    "{x}怎么讲",
    "{x}怎么表达",
    "{x}是什么意思",
    "上海话里{x}怎么说",
    "用上海话{x}怎么讲",
    "{x}怎么表达更自然",
    "{x}用中文怎么说",
    "平时怎么说{x}",
    "{x}这个词怎么讲",
    "{x}怎么理解",
    "{x}怎么讲更合适",
]

HARD_CASES = [
    "我喜欢你",
    "帮我一下怎么讲",
    "没用怎么说",
    "有意思怎么表达",
    "我想帮助你",
    "寻找帮助怎么讲",
    "给他支持怎么说",
    "跟你道歉怎么讲",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate seed queries for split distillation")
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--count", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = [{"query": value, "source": "hard_case"} for value in HARD_CASES]

    candidates = WORD_ITEMS + PHRASE_ITEMS
    while len(rows) < args.count:
        item = random.choice(candidates)
        query = random.choice(SHELLS).format(x=item)
        rows.append(
            {
                "query": query,
                "source": "template_word" if item in WORD_ITEMS else "template_phrase",
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows[: args.count]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
