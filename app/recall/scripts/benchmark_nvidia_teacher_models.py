from __future__ import annotations

import argparse
import asyncio
import json
import re
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
    parser = argparse.ArgumentParser(description="Benchmark NVIDIA chat models for teacher labeling quality")
    parser.add_argument("--tasks_jsonl", required=True, type=str)
    parser.add_argument("--config_json", required=True, type=str)
    parser.add_argument("--models", required=True, nargs="+", type=str)
    parser.add_argument("--sample_size", default=16, type=int)
    parser.add_argument("--api_base", default="https://integrate.api.nvidia.com/v1", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_tokens", default=2200, type=int)
    parser.add_argument("--timeout_s", default=180, type=int)
    parser.add_argument("--output_json", default="", type=str)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except Exception:
            continue
    return rows


def build_user_text(batch: list[dict]) -> str:
    blocks: list[str] = []
    for row in batch:
        blocks.append(
            f"task_id={row['task_id']}\n"
            f"user_query={row['user_query']}\n"
            f"top_n={row.get('top_n', 3)}\n"
            f"candidates:\n{row['candidate_text']}\n"
        )
    return "请按顺序处理以下任务并返回JSON数组：\n\n" + "\n---\n".join(blocks)


def extract_payload(content: str) -> list[dict]:
    m = re.search(r"\[.*\]", content, re.S)
    if m:
        data = json.loads(m.group(0))
        if isinstance(data, list):
            return data
    raise ValueError("no_json_array")


async def run_one(
    session: aiohttp.ClientSession,
    model: str,
    api_base: str,
    user_text: str,
    cand_map: dict[str, set[int]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{api_base.rstrip('/')}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            body = await resp.text()
            dt = time.perf_counter() - t0
            if resp.status != 200:
                return {
                    "model": model,
                    "ok": False,
                    "status": resp.status,
                    "latency_s": round(dt, 3),
                    "error": body[:300],
                }

            data = json.loads(body)
            content = str(data["choices"][0]["message"].get("content", ""))
            items = extract_payload(content)

            valid_task = 0
            total_ids = 0
            valid_id = 0
            for item in items:
                task_id = str(item.get("task_id", "")).strip()
                ids = item.get("ids", [])
                if task_id in cand_map:
                    valid_task += 1
                if not isinstance(ids, list):
                    continue
                for x in ids:
                    sx = str(x).strip()
                    if not sx.isdigit():
                        continue
                    total_ids += 1
                    if task_id in cand_map and int(sx) in cand_map[task_id]:
                        valid_id += 1

            return {
                "model": model,
                "ok": True,
                "status": 200,
                "latency_s": round(dt, 3),
                "tasks_returned": len(items),
                "valid_task": valid_task,
                "total_ids": total_ids,
                "valid_id": valid_id,
                "valid_id_ratio": round(valid_id / max(total_ids, 1), 4),
            }
    except Exception as exc:  # noqa: BLE001
        return {"model": model, "ok": False, "status": "exc", "error": f"{type(exc).__name__}: {exc}"}


async def main_async(args: argparse.Namespace) -> list[dict]:
    cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    api_key = cfg.get("api_key", "")
    if not api_key:
        raise RuntimeError("api_key missing in config_json")
    api_base = cfg.get("api_base") or args.api_base

    tasks = read_jsonl(Path(args.tasks_jsonl))
    if not tasks:
        raise RuntimeError("tasks_jsonl is empty")
    batch = tasks[: max(1, int(args.sample_size))]
    user_text = build_user_text(batch)
    cand_map: dict[str, set[int]] = {}
    for row in batch:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        cand_map[task_id] = set()
        for c in row.get("candidates", []):
            try:
                cand_map[task_id].add(int(c.get("id")))
            except Exception:
                continue

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(ssl=False)) as session:
        out: list[dict] = []
        for model in args.models:
            out.append(
                await run_one(
                    session=session,
                    model=model,
                    api_base=api_base,
                    user_text=user_text,
                    cand_map=cand_map,
                    temperature=float(args.temperature),
                    max_tokens=int(args.max_tokens),
                    timeout_s=int(args.timeout_s),
                )
            )
    return out


def main() -> None:
    args = parse_args()
    result = asyncio.run(main_async(args))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"saved={out_path}")


if __name__ == "__main__":
    main()
