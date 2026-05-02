"""Diagnose seg+POS training data quality."""
import json
from pathlib import Path

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def check_bmes(tags):
    for i, t in enumerate(tags):
        if t == "B" and i + 1 < len(tags) and tags[i + 1] not in ("M", "E"):
            return False
        if t == "M" and (i == 0 or tags[i - 1] not in ("B", "M")):
            return False
        if t == "E" and (i == 0 or tags[i - 1] not in ("B", "M")):
            return False
    return True

def extract_words(seg_text, seg_tags, pos_tags):
    words = []
    cur_text = ""
    cur_pos = []
    for i, (st, pt) in enumerate(zip(seg_tags, pos_tags)):
        ch = seg_text[i] if i < len(seg_text) else "?"
        if st == "S":
            if cur_text:
                words.append((cur_text, cur_pos))
                cur_text = ""
                cur_pos = []
            words.append((ch, [pt]))
        elif st == "B":
            if cur_text:
                words.append((cur_text, cur_pos))
            cur_text = ch
            cur_pos = [pt]
        elif st == "M":
            cur_text += ch
            cur_pos.append(pt)
        elif st == "E":
            cur_text += ch
            cur_pos.append(pt)
            words.append((cur_text, cur_pos))
            cur_text = ""
            cur_pos = []
    if cur_text:
        words.append((cur_text, cur_pos))
    return words

for split in ("train", "dev", "test"):
    rows = load_jsonl(Path(f"app/seg_pos/data/{split}.jsonl"))
    n = len(rows)
    has_seg = sum(1 for r in rows if r.get("segment_tags"))
    has_pos = sum(1 for r in rows if r.get("pos_tags"))
    seg_len_ok = sum(1 for r in rows if r.get("segment_tags") and len(r["segment_tags"]) == len(r.get("segment_text", "").replace(" ", "")))
    bmes_valid = sum(1 for r in rows if r.get("segment_tags") and check_bmes(r["segment_tags"]))

    pos_consistent = 0
    pos_inconsistent = 0
    inconsistent_examples = []
    for r in rows:
        seg_tags = r.get("segment_tags", [])
        pos_tags = r.get("pos_tags", [])
        seg_text = r.get("segment_text", "").replace(" ", "")
        if not seg_tags or not pos_tags or len(seg_tags) != len(pos_tags):
            continue
        words = extract_words(seg_text, seg_tags, pos_tags)
        all_ok = all(len(set(p)) == 1 for _, p in words if p)
        if all_ok:
            pos_consistent += 1
        else:
            pos_inconsistent += 1
            bad = [(w, p) for w, p in words if len(set(p)) > 1]
            if len(inconsistent_examples) < 8:
                inconsistent_examples.append((r["query"][:40], r["headword"], bad[:2]))

    print(f"\n{split}: n={n}  has_seg={has_seg}  has_pos={has_pos}  seg_len_ok={seg_len_ok}  bmes_valid={bmes_valid}  pos_consistent={pos_consistent}  pos_inconsistent={pos_inconsistent}")
    for q, hw, bad in inconsistent_examples[:5]:
        print(f"  Q={q}  hw={hw}  inconsistent={bad}")

# POS tag distribution per position in word
from collections import Counter
pos_at_b = Counter()
pos_at_m = Counter()
pos_at_e = Counter()
pos_at_s = Counter()
rows = load_jsonl(Path("app/seg_pos/data/train.jsonl"))
for r in rows:
    seg_tags = r.get("segment_tags", [])
    pos_tags = r.get("pos_tags", [])
    if not seg_tags or not pos_tags:
        continue
    for st, pt in zip(seg_tags, pos_tags):
        if st == "B":
            pos_at_b[pt] += 1
        elif st == "M":
            pos_at_m[pt] += 1
        elif st == "E":
            pos_at_e[pt] += 1
        elif st == "S":
            pos_at_s[pt] += 1

print("\nPOS distribution at B positions:", dict(pos_at_b.most_common(10)))
print("POS distribution at M positions:", dict(pos_at_m.most_common(10)))
print("POS distribution at E positions:", dict(pos_at_e.most_common(10)))
print("POS distribution at S positions:", dict(pos_at_s.most_common(10)))
