from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge xlsx joint data with old app labels for BiLSTM app training")
    parser.add_argument("--joint_data_dir", default="app/joint/data", type=str)
    parser.add_argument("--recall_records", default="app/recall/index_local_bge_m3/records.jsonl", type=str)
    parser.add_argument("--old_train", default="app/data/train_clean.jsonl", type=str)
    parser.add_argument("--old_dev", default="app/data/dev_clean.jsonl", type=str)
    parser.add_argument("--old_test", default="app/data/test_clean_200.jsonl", type=str)
    parser.add_argument("--out_dir", default="app/joint/data_app", type=str)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_headword(text: str) -> str:
    return re.sub(r"[【】\[\]銆愩€戠\s]", "", str(text)).strip()


def clean_query(text: str) -> str:
    return re.sub(r"\s+", "", str(text).strip())


def segment_tags(text: str, spans: list[tuple[int, int]]) -> list[str]:
    tags = ["S"] * len(text)
    for start, end in spans:
        if start < 0 or end <= start or end > len(text):
            continue
        length = end - start
        if length == 1:
            tags[start] = "S"
        elif length == 2:
            tags[start] = "B"
            tags[start + 1] = "E"
        else:
            tags[start] = "B"
            for i in range(start + 1, end - 1):
                tags[i] = "M"
            tags[end - 1] = "E"
    return tags


def keyword_spans(text: str, keywords: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    occupied = [False] * len(text)
    for keyword in sorted({str(x).strip() for x in keywords if str(x).strip()}, key=len, reverse=True):
        start = text.find(keyword)
        if start < 0:
            continue
        end = start + len(keyword)
        if any(occupied[start:end]):
            continue
        for i in range(start, end):
            occupied[i] = True
        spans.append((start, end))
    return sorted(spans)


def split_name(key: str) -> str:
    value = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "dev"
    return "test"


def build_recall_corpus(records: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    corpus: list[dict] = []
    index: dict[str, dict] = {}
    for row in records:
        doc_id = f"recall:{row.get('id')}"
        headword = clean_headword(row.get("shanghai", ""))
        definition = str(row.get("definition", "")).strip()
        if not headword or not definition:
            continue
        item = {
            "doc_id": doc_id,
            "headword": headword,
            "segments": [headword],
            "pos": "",
            "definition": definition,
            "text": str(row.get("index_text") or f"{headword}。释义：{definition}"),
        }
        corpus.append(item)
        index[doc_id] = item
    return corpus, index


def ngrams(text: str) -> set[str]:
    value = clean_query(text)
    grams: set[str] = set()
    for n in (1, 2, 3):
        if len(value) >= n:
            grams.update(value[i : i + n] for i in range(len(value) - n + 1))
    return grams


def build_match_index(corpus: list[dict]) -> dict[str, object]:
    headword_exact: dict[str, list[dict]] = {}
    definition_exact: dict[str, list[dict]] = {}
    gram_index: dict[str, list[dict]] = {}
    for item in corpus:
        headword = item["headword"]
        definition = item["definition"]
        headword_exact.setdefault(headword, []).append(item)
        definition_exact.setdefault(definition, []).append(item)
        for gram in ngrams(headword) | ngrams(definition):
            gram_index.setdefault(gram, []).append(item)
    return {
        "headword_exact": headword_exact,
        "definition_exact": definition_exact,
        "gram_index": gram_index,
    }


def match_old_row(row: dict, match_index: dict[str, object]) -> dict | None:
    keywords = [str(x).strip() for x in row.get("keywords", []) if str(x).strip()]
    core = str(row.get("core_text", "")).strip()
    candidates = keywords + ([core] if core else [])
    headword_exact = match_index["headword_exact"]
    definition_exact = match_index["definition_exact"]
    gram_index = match_index["gram_index"]

    exact_pool: list[dict] = []
    for term in candidates:
        exact_pool.extend(headword_exact.get(term, []))
        exact_pool.extend(definition_exact.get(term, []))
    if exact_pool:
        return exact_pool[0]

    candidate_docs: dict[str, dict] = {}
    for term in candidates:
        grams = sorted(ngrams(term), key=len, reverse=True)
        for gram in grams[:8]:
            for item in gram_index.get(gram, [])[:200]:
                candidate_docs[item["doc_id"]] = item

    best: tuple[int, dict] | None = None
    for item in candidate_docs.values():
        headword = item["headword"]
        definition = item["definition"]
        score = 0
        for term in candidates:
            if not term:
                continue
            if term == headword:
                score += 20
            elif term in headword:
                score += 8
            if term == definition:
                score += 12
            elif term in definition:
                score += 4
        if score and (best is None or score > best[0]):
            best = (score, item)
    return best[1] if best else None


def old_row_to_sample(row: dict, positive: dict, source: str) -> dict:
    query = str(row.get("query", "")).strip()
    core = clean_query(row.get("core_text") or query)
    keywords = [str(x).strip() for x in row.get("keywords", []) if str(x).strip()]
    spans = keyword_spans(core, keywords)
    return {
        "query": query,
        "positive_doc_id": positive["doc_id"],
        "positive_text": positive["text"],
        "headword": positive["headword"],
        "segments": [core[start:end] for start, end in spans] or [core],
        "segment_text": core,
        "segment_tags": segment_tags(core, spans),
        "pos": positive.get("pos", ""),
        "keywords": keywords[:8],
        "source": source,
        "old_type": row.get("type", ""),
        "old_predicate": row.get("predicate", ""),
        "old_object": row.get("object", ""),
    }


def normalize_joint_row(row: dict) -> dict:
    text = clean_query(row.get("segment_text") or row.get("headword") or row.get("query", ""))
    tags = row.get("segment_tags") or segment_tags(text, [(0, len(text))])
    if len(tags) != len(text):
        tags = tags[: len(text)] + ["S"] * max(0, len(text) - len(tags))
    row = dict(row)
    row["segment_text"] = text
    row["segment_tags"] = tags
    row.setdefault("source", "xlsx_joint")
    return row


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    joint_dir = Path(args.joint_data_dir)

    recall_corpus, _ = build_recall_corpus(read_jsonl(Path(args.recall_records)))
    match_index = build_match_index(recall_corpus)
    xlsx_corpus = read_jsonl(joint_dir / "corpus.jsonl")
    corpus = recall_corpus + xlsx_corpus

    splits: dict[str, list[dict]] = {
        "train": [normalize_joint_row(row) for row in read_jsonl(joint_dir / "train.jsonl")],
        "dev": [normalize_joint_row(row) for row in read_jsonl(joint_dir / "dev.jsonl")],
        "test": [normalize_joint_row(row) for row in read_jsonl(joint_dir / "test.jsonl")],
    }

    old_specs = [
        ("train", Path(args.old_train), "old_train_clean"),
        ("dev", Path(args.old_dev), "old_dev_clean"),
        ("test", Path(args.old_test), "old_test_clean"),
    ]
    old_added = {"train": 0, "dev": 0, "test": 0}
    old_unmatched = 0
    for split, path, source in old_specs:
        for row in read_jsonl(path):
            positive = match_old_row(row, match_index)
            if positive is None:
                old_unmatched += 1
                continue
            sample = old_row_to_sample(row, positive, source)
            target_split = split if split in {"dev", "test"} else split_name(sample["headword"])
            splits[target_split].append(sample)
            old_added[target_split] += 1

    for split, rows in splits.items():
        random.shuffle(rows)
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    write_jsonl(out_dir / "corpus.jsonl", corpus)

    stats = {
        "corpus": len(corpus),
        "recall_corpus": len(recall_corpus),
        "xlsx_corpus": len(xlsx_corpus),
        "train": len(splits["train"]),
        "dev": len(splits["dev"]),
        "test": len(splits["test"]),
        "old_added": old_added,
        "old_unmatched": old_unmatched,
        "out_dir": str(out_dir),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
