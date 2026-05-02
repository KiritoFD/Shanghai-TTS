from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asynchronous batched teacher generation with resume support")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--system_prompt", default="split/prompts/teacher_batch_system_prompt.txt", type=str)
    parser.add_argument("--api_base", default="https://integrate.api.nvidia.com/v1", type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_tokens", default=2200, type=int)
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--timeout_s", default=300, type=int)
    parser.add_argument("--max_retries", default=3, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--launch_interval_s", default=2.0, type=float)
    parser.add_argument("--max_in_flight", default=24, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str, *, allow_bad_lines: bool = False) -> list[dict]:
    rows: list[dict] = []
    bad_lines = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if not allow_bad_lines:
                        raise RuntimeError(f"invalid jsonl: {path}:{line_no}")
                    bad_lines += 1
    if bad_lines:
        print(f"warning: skipped_bad_jsonl_lines={bad_lines} file={path}")
    return rows


def chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_text_from_response(data: dict) -> str | None:
    message = data["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip().startswith(("[", "{")):
        return reasoning_content
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip().startswith(("[", "{")):
        return reasoning
    return None


async def post_chat(
    session: aiohttp.ClientSession,
    api_base: str,
    payload: dict,
    timeout_s: int,
) -> dict:
    async with session.post(
        f"{api_base}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as response:
        response.raise_for_status()
        return await response.json()


async def request_with_retry(
    session: aiohttp.ClientSession,
    api_base: str,
    model: str,
    system_prompt: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
    verbose: bool = False,
    batch_tag: str = "",
) -> tuple[str | None, dict | None]:
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
            data = await post_chat(session, api_base, payload, timeout_s)
            return extract_text_from_response(data), data
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if verbose:
                print(f"[retry] {batch_tag} attempt={attempt + 1}/{max_retries} error={type(exc).__name__}: {exc}")
            if attempt + 1 < max_retries:
                await asyncio.sleep(min(8.0, 1.5 * (attempt + 1)))
    if last_error:
        raise last_error
    raise RuntimeError("request_failed_without_error")


async def main_async(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    api_key = config.get("api_key") or os.environ.get("NVAPI_KEY", "")
    if not api_key:
        raise RuntimeError("NVAPI_KEY is not set")

    model = args.model or config.get("model") or os.environ.get("SPLIT_TEACHER_MODEL", "moonshotai/kimi-k2.5")
    fallback_model = config.get("fallback_model")
    api_base = config.get("api_base", args.api_base)
    temperature = float(config.get("temperature", args.temperature))
    max_tokens = int(config.get("max_tokens", args.max_tokens))
    batch_size = int(config.get("batch_size", args.batch_size))
    timeout_s = int(config.get("timeout_s", args.timeout_s))
    max_retries = int(config.get("max_retries", args.max_retries))
    launch_interval_s = float(config.get("launch_interval_s", args.launch_interval_s))
    max_in_flight = int(config.get("max_in_flight", args.max_in_flight))

    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    input_rows = read_jsonl(args.input, allow_bad_lines=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_queries: set[str] = set()
    if args.resume and output_path.exists():
        for row in read_jsonl(str(output_path), allow_bad_lines=True):
            if isinstance(row.get("batch_queries"), list):
                for query in row["batch_queries"]:
                    value = str(query).strip()
                    if value:
                        completed_queries.add(value)
            else:
                value = str(row.get("query", "")).strip()
                if value:
                    completed_queries.add(value)

    pending_rows = [row for row in input_rows if str(row["query"]) not in completed_queries]
    batches = chunked(pending_rows, batch_size)

    connector = aiohttp.TCPConnector(limit=max_in_flight * 2, ssl=False)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_in_flight)
    start_time = time.time()

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        out_mode = "a" if args.resume else "w"
        with output_path.open(out_mode, encoding="utf-8") as handle:

            async def run_batch(batch_index: int, batch: list[dict]) -> None:
                async with semaphore:
                    launched_at = time.time()
                    batch_tag = f"batch={batch_index} size={len(batch)}"
                    if args.verbose:
                        print(f"[launch] {batch_tag}")
                    user_text = "请逐条标注以下查询，并按原顺序返回 JSON 数组：\n" + "\n".join(
                        f"{index + 1}. {row['query']}" for index, row in enumerate(batch)
                    )

                    text: str | None = None
                    used_model = model
                    last_error = ""
                    try:
                        text, _ = await request_with_retry(
                            session=session,
                            api_base=api_base,
                            model=model,
                            system_prompt=system_prompt,
                            user_text=user_text,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout_s=timeout_s,
                            max_retries=max_retries,
                            verbose=args.verbose,
                            batch_tag=f"{batch_tag} model={model}",
                        )
                    except Exception as exc:  # noqa: BLE001
                        last_error = f"{type(exc).__name__}: {exc}"
                        if args.verbose:
                            print(f"[error] {batch_tag} model={model} error={last_error}")

                    if (text is None or not text.strip()) and fallback_model:
                        try:
                            text, _ = await request_with_retry(
                                session=session,
                                api_base=api_base,
                                model=fallback_model,
                                system_prompt=system_prompt,
                                user_text=user_text,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                timeout_s=timeout_s,
                                max_retries=max_retries,
                                verbose=args.verbose,
                                batch_tag=f"{batch_tag} model={fallback_model}",
                            )
                            used_model = fallback_model
                        except Exception as exc:  # noqa: BLE001
                            last_error = f"{type(exc).__name__}: {exc}"
                            if args.verbose:
                                print(f"[error] {batch_tag} model={fallback_model} error={last_error}")

                    if text is None or not text.strip():
                        output_row = {
                            "batch_index": batch_index,
                            "failed_batch_queries": [str(row["query"]) for row in batch],
                            "batch_sources": [row.get("source", "") for row in batch],
                            "teacher_model": used_model,
                            "error": last_error or "teacher_response_has_no_text",
                        }
                        if args.verbose:
                            print(f"[failed] {batch_tag} elapsed_s={time.time() - launched_at:.1f} error={output_row['error']}")
                    else:
                        output_row = {
                            "batch_index": batch_index,
                            "batch_queries": [str(row["query"]) for row in batch],
                            "batch_sources": [row.get("source", "") for row in batch],
                            "teacher_model": used_model,
                            "raw_response": text,
                        }
                        if args.verbose:
                            print(f"[done] {batch_tag} model={used_model} elapsed_s={time.time() - launched_at:.1f}")

                    async with write_lock:
                        handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                        handle.flush()

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


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
