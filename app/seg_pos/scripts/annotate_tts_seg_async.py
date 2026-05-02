from __future__ import annotations

import argparse
import asyncio
import json
import os
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


VALID_TAGS = {
    "N", "V", "A", "M", "Q", "R", "D", "P", "C",
    "SP", "AS", "Y", "FW", "I", "O", "IDM", "vn", "nd", "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asynchronous batched TTS-oriented seg/POS annotation")
    parser.add_argument("--input_jsonl", required=True, type=str)
    parser.add_argument("--output_jsonl", required=True, type=str)
    parser.add_argument("--config_json", default="", type=str)
    parser.add_argument("--system_prompt", default="app/seg_pos/prompts/tts_seg_batch_system_prompt.txt", type=str)
    parser.add_argument("--api_base", default="https://integrate.api.nvidia.com/v1", type=str)
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_tokens", default=3200, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--timeout_s", default=180, type=int)
    parser.add_argument("--max_retries", default=3, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--launch_interval_s", default=2.0, type=float)
    parser.add_argument("--max_in_flight", default=10, type=int)
    parser.add_argument("--failure_log_jsonl", default="app/seg_pos/outputs/tts_seg_failures.jsonl", type=str)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid jsonl: {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def extract_text_from_response(data: dict[str, Any]) -> str | None:
    message = data["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip().startswith(("[", "{")):
            return value
    return None


def build_user_text(batch: list[dict[str, Any]]) -> str:
    lines = [
        "请按顺序标注以下句子，并返回 JSON 数组。",
        "每句必须包含 sentence_id, sentence, tokens, token_tags, tone_domain_ids, uncertain, notes。",
        "",
    ]
    for item in batch:
        lines.append(f"sentence_id={item['sentence_id']}")
        lines.append(f"sentence={item['sentence']}")
        lines.append("---")
    return "\n".join(lines)


def parse_json_array(text: str) -> list[dict[str, Any]]:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("no_json_array_found")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, list):
        raise ValueError("top_level_is_not_array")
    return payload


def validate_item(item: dict[str, Any], expected_sentence: str) -> tuple[bool, str]:
    sentence = str(item.get("sentence", ""))
    tokens = item.get("tokens", [])
    token_tags = item.get("token_tags", [])
    tone_domain_ids = item.get("tone_domain_ids", [])
    uncertain = item.get("uncertain", [])

    if sentence != expected_sentence:
        return False, "sentence_mismatch"
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        return False, "tokens_invalid"
    if "".join(tokens) != expected_sentence:
        return False, "tokens_do_not_reconstruct_sentence"
    if not isinstance(token_tags, list) or len(token_tags) != len(tokens):
        return False, "token_tags_length_mismatch"
    if not isinstance(tone_domain_ids, list) or len(tone_domain_ids) != len(tokens):
        return False, "tone_domain_ids_length_mismatch"
    if any(str(tag) not in VALID_TAGS for tag in token_tags):
        return False, "invalid_tag_present"
    if any(not isinstance(domain_id, int) for domain_id in tone_domain_ids):
        return False, "tone_domain_ids_not_int"
    if not isinstance(uncertain, list):
        return False, "uncertain_not_list"
    return True, ""


def post_chat_blocking(
    api_base: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    if requests is not None:
        response = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout_s,
        )
        response.raise_for_status()
        return response.json()

    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url=f"{api_base.rstrip('/')}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


async def post_chat_async(
    session: "aiohttp.ClientSession",
    api_base: str,
    payload: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    async with session.post(
        f"{api_base.rstrip('/')}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as response:
        response.raise_for_status()
        return await response.json()


async def request_with_retry(
    api_base: str,
    headers: dict[str, str],
    session: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
    verbose: bool,
    batch_tag: str,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            if session is not None and aiohttp is not None:
                data = await post_chat_async(session, api_base, payload, timeout_s)
            else:
                data = await asyncio.to_thread(post_chat_blocking, api_base, headers, payload, timeout_s)
            text = extract_text_from_response(data)
            if not text:
                raise RuntimeError("empty_model_text")
            return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if verbose:
                print(f"[retry] {batch_tag} attempt={attempt + 1}/{max_retries} error={type(exc).__name__}: {exc}")
            if attempt + 1 < max_retries:
                await asyncio.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(str(last_error) if last_error else "request_failed")


async def main_async(args: argparse.Namespace) -> None:
    cfg = load_config(args.config_json)
    api_key = cfg.get("api_key") or os.environ.get("NVAPI_KEY", "")
    if not api_key:
        raise RuntimeError("NVAPI_KEY is not set and config_json.api_key is missing")

    model = args.model or cfg.get("model") or "moonshotai/kimi-k2.5"
    api_base = cfg.get("api_base", args.api_base)
    temperature = float(cfg.get("temperature", args.temperature))
    max_tokens = int(cfg.get("max_tokens", args.max_tokens))
    batch_size = int(cfg.get("batch_size", args.batch_size))
    timeout_s = int(cfg.get("timeout_s", args.timeout_s))
    max_retries = int(cfg.get("max_retries", args.max_retries))
    launch_interval_s = float(cfg.get("launch_interval_s", args.launch_interval_s))
    max_in_flight = int(cfg.get("max_in_flight", args.max_in_flight))

    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    input_rows = read_jsonl(Path(args.input_jsonl))
    output_path = Path(args.output_jsonl)
    failure_log_path = Path(args.failure_log_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and output_path.exists():
        for row in read_jsonl(output_path):
            sentence_id = str(row.get("sentence_id", "")).strip()
            if sentence_id:
                done_ids.add(sentence_id)

    pending_rows = [row for row in input_rows if str(row.get("sentence_id", "")).strip() not in done_ids]
    batches = chunked(pending_rows, batch_size)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_in_flight)
    start_time = time.time()
    connector = aiohttp.TCPConnector(limit=max_in_flight * 2, ssl=False) if aiohttp is not None else None

    async def run_with_session(session: Any) -> None:
        out_mode = "a" if args.resume else "w"
        fail_mode = "a" if args.resume else "w"
        with output_path.open(out_mode, encoding="utf-8") as out_handle, failure_log_path.open(fail_mode, encoding="utf-8") as fail_handle:

            async def write_failure(payload: dict[str, Any]) -> None:
                async with write_lock:
                    fail_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    fail_handle.flush()

            async def run_batch(batch_index: int, batch: list[dict[str, Any]]) -> None:
                async with semaphore:
                    batch_tag = f"batch={batch_index} size={len(batch)}"
                    launched_at = time.time()
                    if args.verbose:
                        print(f"[launch] {batch_tag}")
                    user_text = build_user_text(batch)
                    try:
                        raw_text = await request_with_retry(
                            api_base=api_base,
                            headers=headers,
                            session=session,
                            model=model,
                            system_prompt=system_prompt,
                            user_text=user_text,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout_s=timeout_s,
                            max_retries=max_retries,
                            verbose=args.verbose,
                            batch_tag=batch_tag,
                        )
                        parsed_items = parse_json_array(raw_text)
                    except Exception as exc:  # noqa: BLE001
                        await write_failure({
                            "batch_index": batch_index,
                            "batch_sentence_ids": [str(row["sentence_id"]) for row in batch],
                            "teacher_model": model,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        return

                    expected_map = {str(row["sentence_id"]): str(row["sentence"]) for row in batch}
                    if len(parsed_items) != len(batch):
                        await write_failure({
                            "batch_index": batch_index,
                            "batch_sentence_ids": list(expected_map.keys()),
                            "teacher_model": model,
                            "error": f"batch_size_mismatch expected={len(batch)} got={len(parsed_items)}",
                            "raw_response": raw_text,
                        })
                        return

                    normalized_rows: list[dict[str, Any]] = []
                    for item in parsed_items:
                        sentence_id = str(item.get("sentence_id", "")).strip()
                        expected_sentence = expected_map.get(sentence_id, "")
                        if not expected_sentence:
                            await write_failure({
                                "batch_index": batch_index,
                                "batch_sentence_ids": list(expected_map.keys()),
                                "teacher_model": model,
                                "error": f"unexpected_sentence_id={sentence_id}",
                                "raw_item": item,
                            })
                            return
                        ok, reason = validate_item(item, expected_sentence)
                        if not ok:
                            await write_failure({
                                "batch_index": batch_index,
                                "batch_sentence_ids": list(expected_map.keys()),
                                "teacher_model": model,
                                "error": reason,
                                "raw_item": item,
                            })
                            return
                        normalized_rows.append({
                            "sentence_id": sentence_id,
                            "sentence": expected_sentence,
                            "tokens": item["tokens"],
                            "token_tags": item["token_tags"],
                            "tone_domain_ids": item["tone_domain_ids"],
                            "uncertain": item.get("uncertain", []),
                            "notes": item.get("notes", ""),
                            "teacher_model": model,
                        })

                    async with write_lock:
                        for row in normalized_rows:
                            out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_handle.flush()

                    if args.verbose:
                        print(f"[done] {batch_tag} elapsed_s={time.time() - launched_at:.1f}")

            tasks: list[asyncio.Task] = []
            for batch_index, batch in enumerate(batches):
                tasks.append(asyncio.create_task(run_batch(batch_index, batch)))
                await asyncio.sleep(launch_interval_s)

            completed = 0
            for task in asyncio.as_completed(tasks):
                await task
                completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    elapsed = time.time() - start_time
                    print(f"completed_batches={completed}/{len(tasks)} elapsed_s={elapsed:.1f}")

    if aiohttp is not None:
        async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
            await run_with_session(session)
    else:
        await run_with_session(None)


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
