from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

try:
    import aiohttp
except Exception:  # noqa: BLE001
    aiohttp = None
try:
    import requests
except Exception:  # noqa: BLE001
    requests = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from annotate_tts_seg_async import (
    VALID_TAGS,
    build_user_text,
    extract_text_from_response,
    parse_json_array,
    read_jsonl,
    validate_item,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark candidate NVIDIA teacher models for TTS sentence annotation")
    parser.add_argument("--input_jsonl", required=True, type=str)
    parser.add_argument("--config_json", default="", type=str)
    parser.add_argument("--models", nargs="+", required=True, type=str)
    parser.add_argument("--sample_size", default=64, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--api_base", default="https://integrate.api.nvidia.com/v1", type=str)
    parser.add_argument("--system_prompt", default="app/seg_pos/prompts/tts_seg_batch_system_prompt.txt", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_tokens", default=3200, type=int)
    parser.add_argument("--timeout_s", default=180, type=int)
    parser.add_argument("--output_json", default="app/seg_pos/outputs/tts_teacher_model_benchmark.json", type=str)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


async def run_one_batch(
    api_base: str,
    headers: dict[str, str],
    session: Any,
    model: str,
    system_prompt: str,
    batch: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_text(batch)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    try:
        if session is not None and aiohttp is not None:
            async with session.post(
                f"{api_base.rstrip('/')}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as response:
                response.raise_for_status()
                data = await response.json()
        else:
            if requests is not None:
                data = await asyncio.to_thread(
                    lambda: _requests_post_json(
                        api_base=api_base,
                        payload=payload,
                        headers=headers,
                        timeout_s=timeout_s,
                    )
                )
            else:
                body = json.dumps(payload).encode("utf-8")
                req = urllib_request.Request(
                    url=f"{api_base.rstrip('/')}/chat/completions",
                    data=body,
                    headers=headers,
                    method="POST",
                )
                data = await asyncio.to_thread(
                    lambda: json.loads(urllib_request.urlopen(req, timeout=timeout_s).read().decode("utf-8"))
                )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "latency_s": round(time.perf_counter() - t0, 3), "error": f"{type(exc).__name__}: {exc}"}

    latency_s = round(time.perf_counter() - t0, 3)
    try:
        text = extract_text_from_response(data)
        if not text:
            raise RuntimeError("empty_model_text")
        parsed = parse_json_array(text)
        if len(parsed) != len(batch):
            raise RuntimeError(f"batch_size_mismatch expected={len(batch)} got={len(parsed)}")
        expected_map = {str(row['sentence_id']): str(row['sentence']) for row in batch}
        for item in parsed:
            sentence_id = str(item.get("sentence_id", "")).strip()
            expected_sentence = expected_map.get(sentence_id, "")
            if not expected_sentence:
                raise RuntimeError(f"unexpected_sentence_id={sentence_id}")
            ok, reason = validate_item(item, expected_sentence)
            if not ok:
                raise RuntimeError(reason)
        uncertain_count = sum(len(item.get("uncertain", [])) for item in parsed)
        token_count = sum(len(item.get("tokens", [])) for item in parsed)
        return {
            "ok": True,
            "latency_s": latency_s,
            "uncertain_count": uncertain_count,
            "token_count": token_count,
            "sample_output": parsed[0] if parsed else {},
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "latency_s": latency_s, "error": f"{type(exc).__name__}: {exc}"}


async def main_async(args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = load_config(args.config_json)
    api_key = cfg.get("api_key") or os.environ.get("NVAPI_KEY", "")
    if not api_key:
        raise RuntimeError("NVAPI_KEY is not set and config_json.api_key is missing")
    api_base = cfg.get("api_base", args.api_base)
    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")

    rows = read_jsonl(Path(args.input_jsonl))
    sample_rows = rows[: args.sample_size]
    batches = chunked(sample_rows, args.batch_size)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    results: list[dict[str, Any]] = []
    connector = aiohttp.TCPConnector(limit=8, ssl=False) if aiohttp is not None else None

    async def run_all(session: Any) -> None:
        for model in args.models:
            batch_results: list[dict[str, Any]] = []
            for batch in batches:
                batch_results.append(
                    await run_one_batch(
                        api_base=api_base,
                        headers=headers,
                        session=session,
                        model=model,
                        system_prompt=system_prompt,
                        batch=batch,
                        temperature=float(args.temperature),
                        max_tokens=int(args.max_tokens),
                        timeout_s=int(args.timeout_s),
                    )
                )
            ok_batches = [result for result in batch_results if result.get("ok")]
            results.append({
                "model": model,
                "batches": len(batch_results),
                "ok_batches": len(ok_batches),
                "parse_pass_rate": round(len(ok_batches) / max(len(batch_results), 1), 4),
                "avg_latency_s": round(sum(result["latency_s"] for result in batch_results) / max(len(batch_results), 1), 3),
                "avg_uncertain_per_ok_batch": round(sum(result.get("uncertain_count", 0) for result in ok_batches) / max(len(ok_batches), 1), 3),
                "sample_output": ok_batches[0].get("sample_output", {}) if ok_batches else {},
                "errors": [result.get("error", "") for result in batch_results if not result.get("ok")][:5],
            })

    if aiohttp is not None:
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            await run_all(session)
    else:
        await run_all(None)
    return results


def _requests_post_json(api_base: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: int) -> dict[str, Any]:
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse_args()
    results = asyncio.run(main_async(args))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, ensure_ascii=False, indent=2)
    output_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
