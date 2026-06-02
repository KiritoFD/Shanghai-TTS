from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def post_json(base_url: str, path: str, payload: dict) -> dict:
    response = requests.post(f"{base_url}{path}", json=payload, timeout=120)
    data = response.json()
    expect(response.ok, f"{path} failed: {response.status_code} {data}")
    return data


def get_json(base_url: str, path: str) -> dict:
    response = requests.get(f"{base_url}{path}", timeout=120)
    data = response.json()
    expect(response.ok, f"{path} failed: {response.status_code} {data}")
    return data


def model_status(payload: dict, model_name: str) -> dict:
    return payload["tts"]["models"][model_name]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test dual TTS runtime APIs.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081", help="Flask base url")
    parser.add_argument("--shaoxing-headword", default="唔有", help="Known Shaoxing headword for audio check")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    print("[1] Initial status")
    status = get_json(base, "/api/tts/status")
    expect(status["ok"], "status not ok")
    expect("shanghai" in status["tts"]["models"], "missing shanghai model")
    expect("shaoxing" in status["tts"]["models"], "missing shaoxing model")

    print("[2] Shanghai -> CPU")
    payload = post_json(base, "/api/tts/load", {"model": "shanghai", "device": "cpu"})
    expect(model_status(payload, "shanghai")["device"] == "cpu", "shanghai not on cpu")

    print("[3] Shanghai -> UNLOADED")
    payload = post_json(base, "/api/tts/load", {"model": "shanghai", "device": "unloaded"})
    expect(model_status(payload, "shanghai")["device"] == "unloaded", "shanghai not unloaded")
    expect(model_status(payload, "shanghai")["loaded"] is False, "shanghai still marked loaded")

    print("[4] Shaoxing -> GPU")
    payload = post_json(base, "/api/tts/load", {"model": "shaoxing", "device": "cuda"})
    expect(model_status(payload, "shaoxing")["device"] == "cuda", "shaoxing not on cuda")

    print("[5] Set pinyin direct model -> shaoxing")
    payload = post_json(base, "/api/tts/direct_model", {"model": "shaoxing"})
    expect(payload["active_direct_model"] == "shaoxing", "direct model did not switch")

    print("[6] Direct pinyin read")
    payload = post_json(base, "/api/chat", {"message": "shi33 sa55 tha55 yan52"})
    expect("/static/" in payload.get("audio", ""), "direct pinyin did not return audio")
    expect("shaoxing" in payload.get("text", ""), "direct pinyin did not report shaoxing model")

    print("[7] Shanghai headword read")
    payload = post_json(base, "/api/chat", {"message": "[太阳]"})
    expect("/static/" in payload.get("audio", ""), "shanghai headword did not return audio")
    expect("shanghai" in payload.get("text", ""), "shanghai headword did not report shanghai model")

    print("[8] Shaoxing headword read")
    payload = post_json(base, "/api/tts/read_headword", {"headword": args.shaoxing_headword, "source": "shaoxing_xlsx"})
    expect("/static/" in payload.get("audio", ""), "shaoxing headword did not return audio")
    expect(payload.get("model") == "shaoxing", "shaoxing headword did not use shaoxing model")

    print("[9] Cleanup -> unload both")
    payload = post_json(base, "/api/tts/load", {"model": "shanghai", "device": "unloaded"})
    expect(model_status(payload, "shanghai")["device"] == "unloaded", "cleanup failed for shanghai")
    payload = post_json(base, "/api/tts/load", {"model": "shaoxing", "device": "unloaded"})
    expect(model_status(payload, "shaoxing")["device"] == "unloaded", "cleanup failed for shaoxing")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
