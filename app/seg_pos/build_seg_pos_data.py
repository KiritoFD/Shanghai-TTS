"""Build segmentation + POS tagging training data from rare2oo.xlsx and output.xlsx.

POS tag set:
  N  = noun      V  = verb      A  = adjective   M = numeral
  Q  = measure   R  = pronoun   D  = adverb       P = preposition
  C  = conjunction SP = structural particle  AS = aspect particle
  Y  = modal     FW = frame word  I = interjection  O = onomatopoeia
  IDM = idiom    vn = verb-object  nd = xx头里
"""
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

POS_MAP = {
    "n": "N", "名": "N", "名词": "N",
    "v": "V", "动": "V", "动词": "V",
    "a": "A", "形": "A", "形容词": "A",
    "m": "M", "数": "M", "数词": "M",
    "q": "Q", "量": "Q", "量词": "Q",
    "r": "R", "代": "R", "代词": "R",
    "d": "D", "副": "D", "副词": "D",
    "p": "P", "介": "P", "介词": "P",
    "c": "C", "连": "C", "连词": "C",
    "sp": "SP", "结构助词": "SP", "助": "SP",
    "as": "AS", "体貌助词": "AS", "体": "AS",
    "y": "Y", "语气助词": "Y", "语气": "Y",
    "fw": "FW", "框式虚词": "FW",
    "i": "I", "叹": "I", "叹词": "I",
    "o": "O", "拟": "O", "拟声词": "O",
    "idm": "IDM", "习语": "IDM", "成语": "IDM",
    "vn": "vn",
    "nd": "nd",
}

VALID_POS = set(POS_MAP.values())


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


def normalize_pos(raw_pos: str) -> str:
    p = raw_pos.strip().lower()
    if p in POS_MAP:
        return POS_MAP[p]
    for key, val in POS_MAP.items():
        if key in p:
            return val
    return "N"


def extract_pos_from_definition(definition: str) -> str | None:
    m = re.match(r"[〈<](.{1,4})[〉>]", definition)
    if m:
        raw = m.group(1).strip()
        return normalize_pos(raw)
    return None


def clean_headword(raw: str) -> tuple[str, list[str]]:
    parts = [part.strip() for part in raw.split("|") if part.strip()]
    if not parts:
        value = raw.strip()
        return value, [value] if value else []
    return "".join(parts), parts


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


def pos_tags_for_word(headword: str, segments: list[str], pos: str) -> list[str]:
    if pos in ("vn",):
        if len(segments) >= 2:
            tags: list[str] = []
            for i, seg in enumerate(segments):
                seg_pos = "V" if i == 0 else "N"
                for _ in seg:
                    tags.append(seg_pos)
            return tags[: len(headword)]
        return [pos] * len(headword)
    if pos == "nd":
        return [pos] * len(headword)
    mapped = normalize_pos(pos) if pos else "N"
    return [mapped] * len(headword)


def stable_bucket(key: str) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def strip_markup(text: str) -> str:
    text = re.sub(r"〈[^〉]*〉", "", text)
    text = re.sub(r"《[^》]*》[^：]*：\"[^\"]*\"", "", text)
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


def parse_examples_from_definition(definition: str, headword: str) -> list[str]:
    examples: list[str] = []
    text = definition.replace("～", headword)
    for m in re.finditer(r"[：:]" + r"([^\u3002\uff1f\uff01!?;；，,～◇〈》]+[\u4e00-\u9fff]{2,}[。！？!?\u4e00-\u9fff]*)", text):
        s = m.group(1).strip().strip("。！？!?，,")
        if len(s) >= 3 and headword in s:
            examples.append(s)
    return examples[:3]


def make_queries_with_pos(headword: str, pos: str, definition: str, examples: list[str]) -> Iterable[tuple[str, str]]:
    keys = definition_keywords(definition, headword)
    yield f"{headword}是什么意思", "headword_meaning"
    yield f"{headword}用上海话怎么说", "headword_surface"
    for key in keys:
        yield f"{key}用上海话怎么说", "definition_to_word"
        yield f"上海话里{key}怎么讲", "definition_to_word"
    for sentence in examples[:2]:
        yield f"{sentence} 这句话里的{headword}是什么意思", "example_context"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build segmentation + POS training data")
    parser.add_argument("--rare_xlsx", default="app/data/rare2oo.xlsx", type=str)
    parser.add_argument("--sentences_xlsx", default="app/data/output.xlsx", type=str)
    parser.add_argument("--out_dir", default="app/seg_pos/data", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_examples_per_entry", default=2, type=int)
    return parser.parse_args()


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
        raw_word, raw_pos, definition = row[0].strip(), row[1].strip() if row[1] else "", row[2].strip() if row[2] else ""
        headword, segments = clean_headword(raw_word)
        if not headword or not definition or headword in seen_headwords:
            continue
        seen_headwords.add(headword)

        pos_from_def = extract_pos_from_definition(definition)
        pos = normalize_pos(raw_pos) if raw_pos else (pos_from_def or "N")
        examples = [s for s in sentences if headword in s][: args.max_examples_per_entry]
        doc_id = f"rare:{row_idx}"
        doc_text = f"{headword}。释义：{definition}"

        seg_tags = segment_tags(headword, segments)
        pos_tags_list = pos_tags_for_word(headword, segments, raw_pos if raw_pos else (pos_from_def or "N"))

        corpus.append({
            "doc_id": doc_id,
            "headword": headword,
            "segments": segments,
            "pos": pos,
            "definition": definition,
            "text": doc_text,
        })

        current_split = "train" if stable_bucket(headword) < 80 else ("dev" if stable_bucket(headword) < 90 else "test")

        for query, source in make_queries_with_pos(headword, pos, definition, examples):
            keywords = [headword] + definition_keywords(definition, headword)
            samples[current_split].append({
                "query": query,
                "positive_doc_id": doc_id,
                "positive_text": doc_text,
                "headword": headword,
                "segments": segments,
                "segment_text": headword,
                "segment_tags": seg_tags,
                "pos_tags": pos_tags_list,
                "pos": pos,
                "keywords": keywords[:5],
                "source": source,
            })

        for sentence in examples:
            sent_segs: list[str] = []
            sent_seg_tags: list[str] = []
            sent_pos_tags: list[str] = []
            remaining = sentence
            for seg in segments:
                seg_clean = seg.strip()
                idx = remaining.find(seg_clean)
                if idx >= 0:
                    if idx > 0:
                        for ch in remaining[:idx]:
                            sent_segs.append(ch)
                            sent_seg_tags.append("S")
                            sent_pos_tags.append("X")
                    for ch_idx, ch in enumerate(seg_clean):
                        sent_segs.append(seg_clean if ch_idx == 0 else "")
                        if len(seg_clean) == 1:
                            sent_seg_tags.append("S")
                        elif ch_idx == 0:
                            sent_seg_tags.append("B")
                        elif ch_idx == len(seg_clean) - 1:
                            sent_seg_tags.append("E")
                        else:
                            sent_seg_tags.append("M")
                        sent_pos_tags.append(pos_tags_list[min(ch_idx, len(pos_tags_list) - 1)])
                    remaining = remaining[idx + len(seg_clean):]
                else:
                    break
            if remaining:
                for ch in remaining:
                    sent_segs.append(ch)
                    sent_seg_tags.append("S")
                    sent_pos_tags.append("X")

            if len(sentence) == len(sent_seg_tags) and len(sentence) == len(sent_pos_tags):
                s_split = "train" if stable_bucket(sentence) < 80 else ("dev" if stable_bucket(sentence) < 90 else "test")
                samples[s_split].append({
                    "query": sentence,
                    "positive_doc_id": doc_id,
                    "positive_text": doc_text,
                    "headword": headword,
                    "segments": segments,
                    "segment_text": sentence,
                    "segment_tags": sent_seg_tags,
                    "pos_tags": sent_pos_tags,
                    "pos": pos,
                    "keywords": [headword],
                    "source": "sentence",
                })

    pos_stat: dict[str, int] = {}
    for split_name, rows in samples.items():
        random.shuffle(rows)
        with (out_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                p = item.get("pos", "N")
                pos_stat[p] = pos_stat.get(p, 0) + 1

    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for item in corpus:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "entries": len(corpus),
        "sentences": len(sentences),
        "train": len(samples["train"]),
        "dev": len(samples["dev"]),
        "test": len(samples["test"]),
        "pos_distribution": pos_stat,
        "segmented_entries": sum(1 for item in corpus if len(item["segments"]) > 1),
        "out_dir": str(out_dir),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
