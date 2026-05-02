"""Shanghai dialect character-level mappings for query expansion.

These mappings help bridge the gap between Mandarin user queries
and Shanghai dialect dictionary headwords.

Categories:
  1. Character substitutions (不→勿, 什→啥, etc.)
  2. Common word-level equivalences
"""
from __future__ import annotations

import re

# Character-level substitutions common in Shanghai dialect writing
CHAR_SUBS: list[tuple[str, str]] = [
    ("不", "勿"),
    ("什么", "啥"),
    ("怎么", "哪能"),
    ("这里", "迭搭"),
    ("那里", "伊面"),
    ("这个", "迭个"),
    ("那个", "伊个"),
    ("这样", "迭能"),
    ("那样", "伊能"),
    ("什么", "啥"),
    ("没有", "呒没"),
    ("不要", "覅"),
    ("不用", "覅"),
    ("没有", "呒"),
    ("很", "邪气"),
    ("非常", "邪气"),
    ("太", "忒"),
    ("漂亮", "趣"),
    ("好看", "趣"),
    ("舒服", "适意"),
    ("累", "衰"),
    ("便宜", "便宜"),
    ("贵", "贵"),
    ("吃", "吃"),
    ("喝", "吃"),
    ("说话", "讲闲话"),
    ("聊天", "讲张"),
    ("吵架", "相骂"),
    ("打架", "相打"),
    ("睡觉", "困觉"),
    ("起床", "起来"),
    ("回家", "转去"),
    ("回去", "转去"),
    ("过来", "过来"),
    ("过去", "过去"),
    ("上午", "上半日"),
    ("下午", "下半日"),
    ("今天", "今朝"),
    ("明天", "明朝"),
    ("后天", "后日"),
    ("昨天", "昨日"),
    ("前天", "前日"),
    ("早上", "早浪向"),
    ("晚上", "夜头"),
    ("中午", "中浪向"),
    ("时候", "辰光"),
    ("现在", "现在"),
    ("刚才", "刚刚"),
    ("一直", "一径"),
    ("经常", "常桩"),
    ("偶尔", "难得"),
    ("全部", "侪"),
    ("都", "侪"),
    ("也", "也"),
    ("还", "还"),
    ("就", "就"),
    ("只", "只"),
    ("先", "先"),
    ("再", "再"),
    ("已经", "已经"),
    ("快", "快"),
    ("慢", "慢"),
    ("大", "大"),
    ("小", "小"),
    ("多", "多"),
    ("少", "少"),
    ("好", "好"),
    ("坏", "坏"),
    ("对", "对"),
    ("错", "错"),
    ("是", "是"),
    ("不是", "勿是"),
    ("可以", "可以"),
    ("不行", "勿来"),
    ("不会", "勿会"),
    ("不知道", "勿晓得"),
    ("明白", "晓得"),
    ("知道", "晓得"),
    ("告诉", "讲拨"),
    ("给", "拨"),
    ("拿", "拿"),
    ("放", "放"),
    ("走", "走"),
    ("跑", "跑"),
    ("来", "来"),
    ("去", "去"),
    ("看", "看"),
    ("听", "听"),
    ("说", "讲"),
    ("想", "想"),
    ("做", "做"),
    ("要", "要"),
    ("有", "有"),
    ("没有", "呒没"),
    ("能", "能"),
    ("会", "会"),
]


def expand_query_variants(query: str) -> list[str]:
    """Generate dialect-expanded variants of a Mandarin query.

    For each character substitution, try replacing in the query.
    Returns list of unique variants (including original).
    """
    variants = [query]
    seen = {query}

    for mandarin, shanghai in CHAR_SUBS:
        if mandarin in query:
            expanded = query.replace(mandarin, shanghai)
            if expanded not in seen and expanded != query:
                seen.add(expanded)
                variants.append(expanded)

    return variants


def get_dialect_keyword_variants(keywords: list[str]) -> list[str]:
    """Expand a list of keywords with dialect variants."""
    all_variants = []
    seen = set()
    for kw in keywords:
        for variant in expand_query_variants(kw):
            if variant not in seen:
                seen.add(variant)
                all_variants.append(variant)
    return all_variants
