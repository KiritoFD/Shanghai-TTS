from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build joint segmentation + embedding data from xlsx files")
    parser.add_argument("--rare_xlsx", default="app/data/rare2oo.xlsx", type=str)
    parser.add_argument("--sentences_xlsx", default="app/data/output.xlsx", type=str)
    parser.add_argument("--out_dir", default="app/joint/data", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_examples_per_entry", default=2, type=int)
    return parser.parse_args()


def column_number(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref or "")
    if not match:
        return 0
    value = 0
    for ch in match.group(1):
        value = value * 26 + ord(ch) - 64
    return value


def cell_text(cell: ET.Element, shared: list[str]) -> str:
    kind = cell.get("t")
    if kind == "s":
        value = cell.find("a:v", NS)
        if value is None or value.text is None:
            return ""
        return shared[int(value.text)]
    if kind == "inlineStr":
        inline = cell.find("a:is", NS)
        if inline is None:
            return ""
        return "".join(t.text or "" for t in inline.findall(".//a:t", NS))
    value = cell.find("a:v", NS)
    return value.text if value is not None and value.text is not None else ""


def read_first_sheet(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", NS):
                shared.append("".join(t.text or "" for t in item.findall(".//a:t", NS)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.get("Id"): rel.get("Target") for rel in rels}
        sheet = workbook.find("a:sheets/a:sheet", NS)
        if sheet is None:
            return []
        target = str(relmap[sheet.get(f"{{{NS['r']}}}id")]).lstrip("/")
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        root = ET.fromstring(zf.read(sheet_path))

        rows: list[list[str]] = []
        for row in root.findall("a:sheetData/a:row", NS):
            values: list[str] = []
            last_col = 0
            for cell in row.findall("a:c", NS):
                col = column_number(cell.get("r", ""))
                values.extend([""] * max(0, col - last_col - 1))
                values.append(cell_text(cell, shared).strip())
                last_col = col
            rows.append(values)
        return rows


def clean_headword(raw: str) -> tuple[str, list[str]]:
    parts = [part.strip() for part in raw.split("|") if part.strip()]
    if not parts:
        value = raw.strip()
        return value, [value] if value else []
    return "".join(parts), parts


def strip_markup(text: str) -> str:
    text = re.sub(r"〈[^〉]*〉", "", text)
    text = re.sub(r"《[^》]*》[^：“]*：“[^”]*”", "", text)
    text = re.sub(r"◇.*$", "", text)
    return text.strip()


def definition_keywords(definition: str, headword: str) -> list[str]:
    cleaned = strip_markup(definition)
    cleaned = cleaned.replace("～", headword)
    chunks = re.split(r"[：:；;，,。丨|（）()、\s]+", cleaned)
    output: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        token = re.sub(r"[^\u4e00-\u9fff]", "", chunk)
        if not token or token == headword or len(token) > 8:
            continue
        if token not in seen:
            seen.add(token)
            output.append(token)
        if len(output) >= 3:
            break
    return output


def segment_tags(text: str, segments: list[str]) -> list[str]:
    if not text:
        return []
    if not segments or "".join(segments) != text:
        segments = [text]
    tags: list[str] = []
    for segment in segments:
        if len(segment) == 1:
            tags.append("S")
        elif len(segment) == 2:
            tags.extend(["B", "E"])
        else:
            tags.extend(["B"] + ["M"] * (len(segment) - 2) + ["E"])
    return tags[: len(text)]


def stable_bucket(key: str) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def split_name(headword: str) -> str:
    bucket = stable_bucket(headword)
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def entry_doc_text(headword: str, pos: str, definition: str) -> str:
    pos_part = f"词性：{pos}。" if pos else ""
    return f"{headword}。{pos_part}释义：{definition}".strip()


def make_queries(headword: str, definition: str, examples: list[str]) -> Iterable[tuple[str, str]]:
    keys = definition_keywords(definition, headword)
    yield f"{headword}是什么意思", "headword_meaning"
    yield f"{headword}用上海话怎么说", "headword_surface"
    for key in keys:
        yield f"{key}用上海话怎么说", "definition_to_word"
        yield f"上海话里{key}怎么讲", "definition_to_word"
    for sentence in examples[:2]:
        yield f"{sentence} 这句话里的{headword}是什么意思", "example_context"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rare_rows = read_first_sheet(Path(args.rare_xlsx))
    sentence_rows = read_first_sheet(Path(args.sentences_xlsx))
    sentences = [row[0].strip() for row in sentence_rows[1:] if row and row[0].strip()]

    corpus: list[dict] = []
    samples: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    seen_headwords: set[str] = set()

    for row_idx, row in enumerate(rare_rows[1:], start=1):
        if len(row) < 3:
            continue
        raw_word, pos, definition = row[0].strip(), row[1].strip(), row[2].strip()
        headword, segments = clean_headword(raw_word)
        if not headword or not definition or headword in seen_headwords:
            continue
        seen_headwords.add(headword)

        examples = [s for s in sentences if headword in s][: args.max_examples_per_entry]
        doc_id = f"rare:{row_idx}"
        doc_text = entry_doc_text(headword, pos, definition)
        corpus.append(
            {
                "doc_id": doc_id,
                "headword": headword,
                "segments": segments,
                "pos": pos,
                "definition": definition,
                "text": doc_text,
            }
        )

        current_split = split_name(headword)
        tags = segment_tags(headword, segments)
        for query, source in make_queries(headword, definition, examples):
            keywords = [headword] + definition_keywords(definition, headword)
            samples[current_split].append(
                {
                    "query": query,
                    "positive_doc_id": doc_id,
                    "positive_text": doc_text,
                    "headword": headword,
                    "segments": segments,
                    "segment_text": headword,
                    "segment_tags": tags,
                    "pos": pos,
                    "keywords": keywords[:5],
                    "source": source,
                }
            )

    for split, rows in samples.items():
        random.shuffle(rows)
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for item in corpus:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "entries": len(corpus),
        "sentences": len(sentences),
        "train": len(samples["train"]),
        "dev": len(samples["dev"]),
        "test": len(samples["test"]),
        "segmented_entries": sum(1 for item in corpus if len(item["segments"]) > 1),
        "out_dir": str(out_dir),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

