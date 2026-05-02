import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "joint" / "scripts"))
from build_joint_data import read_first_sheet

sent_rows = read_first_sheet(Path("app/data/output.xlsx"))
print(f"Total sentence rows: {len(sent_rows)}")
for r in sent_rows[:15]:
    print([str(c)[:80] if c else "" for c in r[:4]])

rare_rows = read_first_sheet(Path("app/data/rare2oo.xlsx"))
print(f"\nRare rows sample (with POS):")
for r in rare_rows[1:20]:
    if len(r) >= 3:
        raw_word = str(r[0]).strip()[:20]
        pos = str(r[1]).strip()[:10] if r[1] else ""
        definition = str(r[2]).strip()[:60] if r[2] else ""
        print(f"  word={raw_word:20s}  pos={pos:10s}  def={definition}")
