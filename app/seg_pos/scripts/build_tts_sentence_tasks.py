from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.seg_pos.build_seg_pos_data import read_first_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TTS-oriented sentence annotation tasks from output.xlsx")
    parser.add_argument("--input_xlsx", default="app/seg_pos/data/output.xlsx", type=str)
    parser.add_argument("--output_jsonl", default="app/seg_pos/data/tts_sentence_tasks.jsonl", type=str)
    parser.add_argument("--limit", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_first_sheet(Path(args.input_xlsx))
    if not rows:
        raise RuntimeError(f"no rows found in {args.input_xlsx}")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentences = [row[0].strip() for row in rows[1:] if row and row[0].strip()]
    if args.limit > 0:
        sentences = sentences[: args.limit]

    with output_path.open("w", encoding="utf-8") as handle:
        for index, sentence in enumerate(sentences, start=1):
            row = {
                "sentence_id": f"output-{index:05d}",
                "sentence": sentence,
                "source": "output.xlsx",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input_xlsx": args.input_xlsx,
        "output_jsonl": str(output_path),
        "count": len(sentences),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
