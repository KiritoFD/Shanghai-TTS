from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(
        [
            "pkill",
            "-f",
            "train/recall/recall_service.py --mode api --host 127.0.0.1 --port 8088",
        ],
        cwd=ROOT,
        check=False,
    )

    proc = subprocess.Popen(
        [sys.executable, "train/app.py"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 90
        health_ok = False
        conf_ok = False
        chat_result = None
        while time.time() < deadline:
            try:
                health_ok = requests.get("http://127.0.0.1:8088/health", timeout=2).status_code == 200
            except Exception:
                health_ok = False
            try:
                conf_ok = requests.get("http://127.0.0.1:8081/api/user/confinfo", timeout=2).status_code == 200
            except Exception:
                conf_ok = False
            if conf_ok:
                chat_result = requests.post(
                    "http://127.0.0.1:8081/api/chat",
                    json={"message": "你好怎么说"},
                    timeout=90,
                ).json()
                break
            time.sleep(1)

        print(
            json.dumps(
                {
                    "recall_health_ok": health_ok,
                    "app_conf_ok": conf_ok,
                    "chat_result": chat_result,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        print("\n--- app log tail ---")
        print(out[-5000:])


if __name__ == "__main__":
    main()
