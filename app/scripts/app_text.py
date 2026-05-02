from __future__ import annotations

import os

os.environ.setdefault("WUU_TEXT_ONLY", "1")

from app import app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8082, debug=False)
