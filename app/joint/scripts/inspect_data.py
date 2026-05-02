import sys
import json
from pathlib import Path

app_root = Path(__file__).resolve().parents[2] / "app"
sys.path.insert(0, str(app_root))
sys.path.insert(0, str(app_root / "joint" / "scripts"))

from build_joint_data import read_first_sheet

rare_rows = read_first_sheet(Path("app/data/rare2oo.xlsx"))
sent_rows = read_first_sheet(Path("app/data/output.xlsx"))

print("=== rare2oo.xlsx ===")
print(f"Total rows: {len(rare_rows)}")
for r in rare_rows[:5]:
    print([str(c)[:60] if c else "" for c in r[:8]])

print("\n=== output.xlsx ===")
print(f"Total rows: {len(sent_rows)}")
for r in sent_rows[:10]:
    print([str(c)[:80] if c else "" for c in r[:5]])

# Check how many rare2oo rows have | separator (segmented headwords)
segmented = 0
pos_info = set()
for r in rare_rows[1:]:
    if len(r) < 3:
        continue
    raw_word = str(r[0]).strip()
    pos = str(r[1]).strip() if len(r) > 1 and r[1] else ""
    if pos:
        pos_info.add(pos)
    if "|" in raw_word:
        segmented += 1
print(f"\nSegmented headwords (with |): {segmented}/{len(rare_rows)-1}")
print(f"POS tags seen: {sorted(pos_info)[:30]}")

# Count definition patterns
def_patterns = {}
for r in rare_rows[1:]:
    if len(r) < 3:
        continue
    definition = str(r[2]).strip() if r[2] else ""
    if definition.startswith("\u3008"):
        tag = definition[:5]
        def_patterns[tag] = def_patterns.get(tag, 0) + 1
print(f"\nDefinition POS prefix patterns: {json.dumps(def_patterns, ensure_ascii=False, indent=2)}")
