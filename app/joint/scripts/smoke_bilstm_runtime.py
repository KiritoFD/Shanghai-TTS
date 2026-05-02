from __future__ import annotations

import os
import sys

sys.path.insert(0, "app")
os.environ.setdefault("WUU_PREPROCESSOR_BACKEND", "bilstm_joint")
os.environ.setdefault("WUU_TEXT_ONLY", "1")

import app as app_mod  # noqa: E402


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "揵是什么意思"
    preprocessor = app_mod.ensure_preprocessor()
    parsed = preprocessor.preprocess(query)
    print(parsed)
    print(app_mod.build_search_terms(query, parsed))


if __name__ == "__main__":
    main()
