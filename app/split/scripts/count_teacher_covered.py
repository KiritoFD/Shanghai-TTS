from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print(0)
        return
    path = Path(sys.argv[1])
    if not path.exists():
        print(0)
        return

    covered = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row.get("batch_queries"), list):
                covered += len(row["batch_queries"])
    print(covered)


if __name__ == "__main__":
    main()
