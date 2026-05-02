from __future__ import annotations

import argparse
import json
from pathlib import Path

from split_distill.rules import render_final_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render structured rows into production output format")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.input).open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            payload = {
                "query": row["query"],
                "rendered": render_final_output(row),
            }
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
