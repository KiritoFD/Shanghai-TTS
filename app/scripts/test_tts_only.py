"""Test TTS engine independently."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os_path = Path(r"G:\GitHub\Shanghai-TTS\app\static")
os_path.mkdir(exist_ok=True)

from tts_engine import synthesize

test_cases = [
    ("shi33 yan55 kua33 chi21", "上海话：洋夸气"),
    ("ngu23 teq5 maon22", "上海话：我特忙"),
    ("baq5 xiang33", "上海话：白相"),
    ("gha23 se55 hu23", "上海话：嘎讪胡"),
]

print("Testing TTS engine...")
print("=" * 60)

for pinyin, desc in test_cases:
    out_path = str(os_path / f"test_{hash(pinyin) % 10000}.wav")
    t0 = time.perf_counter()
    try:
        synthesize(pinyin, out_path)
        elapsed = time.perf_counter() - t0
        file_size = Path(out_path).stat().st_size
        print(f"  [OK] {desc:20s} pinyin={pinyin:30s} -> {file_size:,d} bytes ({elapsed*1000:.0f}ms)")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  [FAIL] {desc:20s} pinyin={pinyin:30s} -> ERROR: {exc}")

print("=" * 60)
print("TTS test complete.")
