from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp


SYSTEM_PROMPT = (
    "你是上海话词典匹配助手。"
    "你将收到多个任务。每个任务包含 task_id、user_query、top_n、候选词条。"
    "请为每个任务选择最匹配的最多top_n个ID。"
    "输出必须是JSON数组，每项格式：{\"task_id\":\"...\",\"ids\":[8439,8440]}。"
    "如没有匹配，返回空数组ids。不要输出任何解释。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async teacher generation for recall top-match labels")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--api_base", default="https://integrate.api.nvidia.com/v1", type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_tokens", default=2200, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--timeout_s", default=300, type=int)
    parser.add_argument("--max_retries", default=3, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--launch_interval_s", default=2.0, type=float)
    parser.add_argument("--max_in_flight", default=10, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows


def chunked(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def extract_text(data: dict) -> str | None:
    msg = data["choices"][0]["message"]
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content
    for key in ("reasoning_content", "reasoning"):
        v = msg.get(key)
        if isinstance(v, str) and v.strip().startswith(("[", "{")):
            return v
    return None


async def post_chat(session: aiohttp.ClientSession, api_base: str, payload: dict, timeout_s: int) -> dict:
    async with session.post(
        f"{api_base}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def request_with_retry(
    session: aiohttp.ClientSession,
    api_base: str,
    model: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
    verbose: bool = False,
    tag: str = "",
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            data = await post_chat(session, api_base, payload, timeout_s)
            text = extract_text(data)
            if text is None:
                raise RuntimeError("empty_teacher_text")
            return text
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if verbose:
                print(f"[retry] {tag} attempt={attempt+1}/{max_retries} err={type(exc).__name__}:{exc}")
            if attempt + 1 < max_retries:
                await asyncio.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(str(last_err) if last_err else "teacher_request_failed")


def build_user_text(batch: list[dict]) -> str:
    blocks = []
    for row in batch:
        blocks.append(
            f"task_id={row['task_id']}\n"
            f"user_query={row['user_query']}\n"
            f"top_n={row.get('top_n',3)}\n"
            f"candidates:\n{row['candidate_text']}\n"
        )
    return "请按顺序处理以下任务并返回JSON数组：\n\n" + "\n---\n".join(blocks)


async def main_async(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    api_key = cfg.get("api_key") or os.environ.get("NVAPI_KEY", "")
    if not api_key:
        raise RuntimeError("NVAPI_KEY is not set")
    model = args.model or cfg.get("model") or "moonshotai/kimi-k2-instruct-0905"
    api_base = cfg.get("api_base", args.api_base)
    temperature = float(cfg.get("temperature", args.temperature))
    max_tokens = int(cfg.get("max_tokens", args.max_tokens))

    input_rows = read_jsonl(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and output_path.exists():
        for row in read_jsonl(str(output_path)):
            for tid in row.get("batch_task_ids", []):
                done_ids.add(str(tid))

    pending = [r for r in input_rows if str(r.get("task_id", "")) not in done_ids]
    batches = chunked(pending, args.batch_size)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    sem = asyncio.Semaphore(args.max_in_flight)
    write_lock = asyncio.Lock()
    start = time.time()

    async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(limit=args.max_in_flight * 2, ssl=False)) as session:
        mode = "a" if args.resume else "w"
        with output_path.open(mode, encoding="utf-8") as fout:

            async def run_batch(i: int, batch: list[dict]) -> None:
                async with sem:
                    tag = f"batch={i} size={len(batch)}"
                    if args.verbose:
                        print(f"[launch] {tag}")
                    t0 = time.time()
                    user_text = build_user_text(batch)
                    try:
                        text = await request_with_retry(
                            session=session,
                            api_base=api_base,
                            model=model,
                            user_text=user_text,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout_s=args.timeout_s,
                            max_retries=args.max_retries,
                            verbose=args.verbose,
                            tag=tag,
                        )
                        row = {
                            "batch_index": i,
                            "batch_task_ids": [str(x["task_id"]) for x in batch],
                            "teacher_model": model,
                            "raw_response": text,
                        }
                        if args.verbose:
                            print(f"[done] {tag} elapsed_s={time.time()-t0:.1f}")
                    except Exception as exc:  # noqa: BLE001
                        row = {
                            "batch_index": i,
                            "failed_batch_task_ids": [str(x["task_id"]) for x in batch],
                            "teacher_model": model,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        if args.verbose:
                            print(f"[failed] {tag} err={row['error']}")
                    async with write_lock:
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fout.flush()

            tasks: list[asyncio.Task] = []
            for i, batch in enumerate(batches):
                tasks.append(asyncio.create_task(run_batch(i, batch)))
                await asyncio.sleep(args.launch_interval_s)

            done = 0
            for t in asyncio.as_completed(tasks):
                await t
                done += 1
                if done % 20 == 0 or done == len(tasks):
                    print(f"completed_batches={done}/{len(tasks)} elapsed_s={time.time()-start:.1f}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
