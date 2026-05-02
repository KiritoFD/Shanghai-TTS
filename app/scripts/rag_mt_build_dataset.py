from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path


SYSTEM_PROMPT = "你是一个精确的上海话翻译器。请参考给定的词汇对照表，将普通话翻译为上海话。不解释，只输出结果。"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dictionary-grounded RAG-MT SFT data.")
    parser.add_argument("--dictionary_csv", default="app/recall/processed_results.csv", type=str)
    parser.add_argument("--split_jsonl", default="app/split/data/processed/query_train_teacher_5000_clean_all.jsonl", type=str)
    parser.add_argument("--output_dir", default="app/rag_mt_data", type=str)
    parser.add_argument("--train_size", default=1200, type=int)
    parser.add_argument("--eval_size", default=120, type=int)
    parser.add_argument("--max_split_rows", default=2500, type=int)
    parser.add_argument("--include_hand_seeds", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def sanitize_headword(text: str) -> str:
    return str(text or "").strip().strip("[]").strip("【】")


def clean_definition(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"〈[^〉]+〉", "", value)
    value = re.sub(r"《[^》]+》", "", value)
    value = re.sub(r"～", "", value)
    value = re.split(r"[：:；;。]", value)[0]
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9，,、]", "", value)
    value = value.strip("，,、 ")
    return value


def is_good_pair(source: str, target: str) -> bool:
    if not source or not target:
        return False
    if len(source) < 2 or len(source) > 18:
        return False
    if len(target) > 12:
        return False
    if source == target:
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", source)) and bool(re.search(r"[\u4e00-\u9fff]", target))


def make_payload(
    core_text: str,
    context: str,
    target: str,
    source: str,
    raw_query: str | None = None,
    keywords: list[str] | None = None,
) -> dict:
    clean_keywords = [str(item).strip() for item in (keywords or []) if str(item).strip()]
    if core_text and core_text not in clean_keywords:
        clean_keywords.insert(0, core_text)
    return {
        "raw_query": raw_query or core_text,
        "core_text": core_text,
        "keywords": clean_keywords,
        "lexicon_context": context,
        "target": target,
        "source": source,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"【原始输入】：{raw_query or core_text}\n"
                    f"【清洗后普通话】：{core_text}\n"
                    f"【关键词】：{'、'.join(clean_keywords) or '无'}\n"
                    f"【参考词汇】：{context or '无'}"
                ),
            },
            {"role": "assistant", "content": target},
        ],
    }


def read_dictionary(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 5:
                continue
            target = sanitize_headword(row[0])
            source = clean_definition(row[4])
            if is_good_pair(source, target):
                pairs.append((source, target))
    return pairs


def find_pair_for_keyword(keyword: str, pairs: list[tuple[str, str]]) -> tuple[str, str] | None:
    key = clean_definition(keyword)
    if not key:
        return None
    for source, target in pairs:
        if source == key:
            return source, target
    for source, target in pairs:
        if key in source or source in key:
            return source, target
    return None


def read_split_examples(path: Path, pairs: list[tuple[str, str]], max_rows: int) -> list[dict]:
    if not path.exists():
        return []

    examples: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            core_text = clean_definition(row.get("core_text", ""))
            if not core_text:
                continue
            raw_keywords = row.get("keywords", [])
            if not isinstance(raw_keywords, list):
                raw_keywords = []
            keywords = [clean_definition(str(item)) for item in raw_keywords]
            keywords = [item for item in keywords if item]
            if core_text not in keywords:
                keywords.insert(0, core_text)

            context_pairs: list[tuple[str, str]] = []
            translated = core_text
            for keyword in sorted(set(keywords), key=len, reverse=True):
                match = find_pair_for_keyword(keyword, pairs)
                if match is None:
                    continue
                source, target = match
                context_pairs.append((keyword, target))
                if keyword in translated:
                    translated = translated.replace(keyword, target)

            if not context_pairs:
                continue
            if translated == core_text:
                translated = context_pairs[0][1]
            context = "; ".join(f"{source}={target}" for source, target in context_pairs[:8])
            examples.append(
                make_payload(
                    core_text,
                    context,
                    translated,
                    "split_synthetic",
                    raw_query=str(row.get("query", "")).strip() or core_text,
                    keywords=keywords,
                )
            )
    return examples


def build_examples(pairs: list[tuple[str, str]], split_examples: list[dict]) -> list[dict]:
    examples: list[dict] = []
    for core_text, target in pairs:
        context = f"{core_text}={target}"
        examples.append(make_payload(core_text, context, target, "dictionary"))

    examples.extend(split_examples)
    return examples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    pairs = read_dictionary(Path(args.dictionary_csv))
    split_examples = read_split_examples(Path(args.split_jsonl), pairs, max_rows=args.max_split_rows)
    rng.shuffle(pairs)
    examples = build_examples(pairs, split_examples)
    rng.shuffle(examples)

    total = min(len(examples), args.train_size + args.eval_size)
    selected = examples[:total]
    eval_rows = selected[: args.eval_size]
    train_rows = selected[args.eval_size :]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)

    stats = {
        "dictionary_pairs": len(pairs),
        "split_synthetic": len(split_examples),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "output_dir": str(output_dir),
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
