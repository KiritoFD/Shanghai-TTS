"""Generate benchmark labels for 200 test queries using NVIDIA API.

Produces:
  1. Recall labels (target_ids) — for recall engine evaluation
  2. Split labels (core_text, type, keywords) — for query normalization evaluation

Usage:
    python app/scripts/generate_bench_labels.py --nvapi_key nvapi-xxx
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://integrate.api.nvidia.com/v1"
RECALL_MODEL = "moonshotai/kimi-k2-instruct-0905"
SPLIT_MODEL = "moonshotai/kimi-k2.5"

RECALL_SYSTEM_PROMPT = (
    "你是上海话词典匹配助手。"
    "你将收到多个任务。每个任务包含 task_id、user_query、top_n、候选词条。"
    "请为每个任务选择最匹配的最多top_n个ID。"
    "输出必须是JSON数组，每项格式：{\"task_id\":\"...\",\"ids\":[8439,8440]}。"
    "如没有匹配，返回空数组ids。不要输出任何解释。"
)

SPLIT_SYSTEM_PROMPT = (
    "你是一个中文查询规范化标注器。你会一次看到多条用户输入，"
    "你的任务是为每条输入输出一个结构化 JSON 对象，并最终返回一个 JSON 数组。\n\n"
    "严格规则：\n"
    "1. 只输出 JSON 数组，不要输出解释、Markdown、代码块。\n"
    "2. 数组中每个元素必须包含字段：query, core_text, type, predicate, object, keywords\n"
    "3. type 只能是：词项 或 动作短语\n"
    "4. 先去掉问句壳和表达壳，只保留核心语义\n"
    '5. 如果是词项：predicate和object置空，keywords输出1到5个中文近义词\n'
    "6. 如果是动作短语：提取predicate，必要时保留object\n"
    "7. keywords必须是中文、去重、不含代词虚词\n"
    "8. 结果数组长度必须与输入条数一致，保留原query字段\n\n"
    "输出示例：\n"
    '[{"query":"开心怎么说","core_text":"开心","type":"词项","predicate":"","object":"","keywords":["快乐","高兴"]}]'
)

INPUT_JSONL = APP_ROOT / "data" / "test_clean_200.jsonl"
RECALL_OUTPUT = APP_ROOT / "data" / "test_clean_200_recall.jsonl"
SPLIT_OUTPUT = APP_ROOT / "data" / "test_clean_200_split.jsonl"

LAUNCH_INTERVAL = 3.0  # seconds between requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def chunked(items: list, size: int) -> list[list]:
    return [items[i: i + size] for i in range(0, len(items), size)]


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


async def post_chat(session: aiohttp.ClientSession, payload: dict, timeout_s: int = 60) -> dict:
    async with session.post(
        f"{API_BASE}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def request_with_retry(
    session: aiohttp.ClientSession,
    model: str,
    system_prompt: str,
    user_text: str,
    max_retries: int = 3,
    timeout_s: int = 60,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.0,
        "max_tokens": 2200,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            data = await post_chat(session, payload, timeout_s)
            text = extract_text(data)
            if text:
                return text
            raise RuntimeError("empty response")
        except Exception as exc:
            last_err = exc
            if attempt + 1 < max_retries:
                await asyncio.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(str(last_err) if last_err else "request failed")


# ---------------------------------------------------------------------------
# Recall task building
# ---------------------------------------------------------------------------


def build_recall_tasks(queries: list[dict]) -> tuple[list[dict], object]:
    """Build recall tasks using RecallEngine to find candidates."""
    from recall.engine import RecallEngine, encode_queries_batch, merge_query_variants

    print("[recall] loading RecallEngine for candidate retrieval...", flush=True)
    engine = RecallEngine(index_dir=APP_ROOT / "recall" / "index_local_bge_m3", ann="hnsw", ef_search=64)

    # Collect all query texts
    query_texts = []
    valid_indices = []
    for i, row in enumerate(queries):
        q = str(row.get("query", "")).strip()
        if q:
            query_texts.append(q)
            valid_indices.append(i)

    # Batch-encode all query variants
    print(f"[recall] batch-encoding {len(query_texts)} queries...")
    all_variants = []
    variant_map = []
    for q in query_texts:
        variants = merge_query_variants(q)
        variant_map.append(variants)
        all_variants.extend(variants)
    unique_variants = list(dict.fromkeys(all_variants))
    vecs = encode_queries_batch(unique_variants, engine.tokenizer, engine.model, engine.device, batch_size=64)
    vec_map = {v: vecs[i] for i, v in enumerate(unique_variants)}

    # Search with pre-computed vectors
    print(f"[recall] searching {len(query_texts)} queries...", flush=True)
    tasks = []
    for idx, (q, variants) in enumerate(zip(query_texts, variant_map)):
        result = engine.search(q, top_k=20, top_n=20, precomputed_vecs=vec_map)
        candidates = result.get("results", [])
        if (idx + 1) % 50 == 0:
            print(f"[recall] searched {idx + 1}/{len(query_texts)}", flush=True)

        candidate_text = "\n".join(
            f"ID:{int(c['id'])} | 上海话:{c.get('shanghai','')} | 释义:{c.get('definition','')}"
            for c in candidates
        )

        tasks.append({
            "task_id": f"q{valid_indices[idx]}",
            "user_query": q,
            "top_n": 3,
            "candidate_text": candidate_text,
            "candidate_ids": [int(c["id"]) for c in candidates],
        })

    return tasks, engine


def build_recall_user_text(batch: list[dict]) -> str:
    blocks = []
    for row in batch:
        blocks.append(
            f"task_id={row['task_id']}\n"
            f"user_query={row['user_query']}\n"
            f"top_n={row['top_n']}\n"
            f"candidates:\n{row['candidate_text']}"
        )
    return "请按顺序处理以下任务并返回JSON数组：\n\n" + "\n---\n".join(blocks)


def parse_recall_response(text: str, batch: list[dict]) -> dict[str, list[int]]:
    """Parse teacher response into task_id -> ids mapping."""
    result = {}
    try:
        # Try to find JSON array in the response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            items = json.loads(text[start:end])
            for item in items:
                tid = str(item.get("task_id", ""))
                ids = item.get("ids", [])
                if tid:
                    result[tid] = [int(x) for x in ids if x is not None]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return result


# ---------------------------------------------------------------------------
# Split task building
# ---------------------------------------------------------------------------


def build_split_user_text(queries: list[str]) -> str:
    items = [{"query": q} for q in queries]
    return json.dumps(items, ensure_ascii=False)


def parse_split_response(text: str) -> list[dict]:
    """Parse teacher response into list of dicts."""
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return []


# ---------------------------------------------------------------------------
# Main async
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    api_key = args.nvapi_key or os.environ.get("NVAPI_KEY", "")
    if not api_key:
        raise RuntimeError("NVAPI_KEY is not set")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    queries = read_jsonl(INPUT_JSONL)
    print(f"[load] {len(queries)} queries from {INPUT_JSONL}")

    # ===================================================================
    # Phase 1: Recall labels
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  Phase 1: Generating recall target_ids ({len(queries)} queries)")
    print(f"{'='*60}")

    recall_tasks, engine = build_recall_tasks(queries)
    recall_batches = chunked(recall_tasks, args.recall_batch_size)
    print(f"[recall] {len(recall_tasks)} tasks, {len(recall_batches)} batches")

    recall_results: dict[str, list[int]] = {}  # task_id -> target_ids

    # Load existing results for resume
    if args.resume and RECALL_OUTPUT.exists():
        for row in read_jsonl(RECALL_OUTPUT):
            tid = row.get("task_id", "")
            tids = row.get("target_ids", [])
            if tid and tids:
                recall_results[tid] = tids
        print(f"[recall] resume: {len(recall_results)} already done")

    recall_pending = [b for b in recall_batches if b[0]["task_id"] not in recall_results]
    sem_recall = asyncio.Semaphore(args.max_in_flight)

    async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(limit=20, ssl=False)) as session:
        async def run_recall_batch(batch: list[dict]) -> None:
            async with sem_recall:
                user_text = build_recall_user_text(batch)
                text = await request_with_retry(session, RECALL_MODEL, RECALL_SYSTEM_PROMPT, user_text)
                parsed = parse_recall_response(text, batch)
                for task in batch:
                    tid = task["task_id"]
                    ids = parsed.get(tid, [])
                    recall_results[tid] = ids

        t0 = time.time()
        tasks = []
        for batch in recall_pending:
            tasks.append(asyncio.create_task(run_recall_batch(batch)))
            await asyncio.sleep(LAUNCH_INTERVAL)

        done = 0
        for t in asyncio.as_completed(tasks):
            try:
                await t
            except Exception as exc:
                print(f"[recall] batch failed: {exc}")
            done += 1
            if done % 10 == 0:
                print(f"[recall] progress: {done}/{len(recall_pending)} batches")

    elapsed = time.time() - t0
    print(f"[recall] done in {elapsed:.0f}s, {len(recall_results)} labeled")

    # Write recall output
    with RECALL_OUTPUT.open("w", encoding="utf-8") as f:
        for task in recall_tasks:
            tid = task["task_id"]
            row = {
                "task_id": tid,
                "user_query": task["user_query"],
                "query": task["user_query"],
                "target_ids": recall_results.get(tid, []),
                "candidate_ids": task["candidate_ids"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[recall] saved to {RECALL_OUTPUT}")

    # ===================================================================
    # Phase 2: Split labels
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  Phase 2: Generating split labels ({len(queries)} queries)")
    print(f"{'='*60}")

    all_split_results: dict[str, dict] = {}  # query -> result

    # Load existing for resume
    if args.resume and SPLIT_OUTPUT.exists():
        for row in read_jsonl(SPLIT_OUTPUT):
            q = row.get("query", "")
            if q and row.get("core_text"):
                all_split_results[q] = row
        print(f"[split] resume: {len(all_split_results)} already done")

    query_texts = [str(r.get("query", "")).strip() for r in queries if r.get("query")]
    split_batches = chunked(query_texts, args.split_batch_size)
    split_pending = [b for b in split_batches if b[0] not in all_split_results]
    print(f"[split] {len(query_texts)} queries, {len(split_pending)} pending batches")

    sem_split = asyncio.Semaphore(args.max_in_flight)

    async with aiohttp.ClientSession(headers=headers, connector=aiohttp.TCPConnector(limit=20, ssl=False)) as session:
        async def run_split_batch(batch: list[str]) -> None:
            async with sem_split:
                user_text = build_split_user_text(batch)
                text = await request_with_retry(session, SPLIT_MODEL, SPLIT_SYSTEM_PROMPT, user_text)
                parsed = parse_split_response(text)
                for item in parsed:
                    q = str(item.get("query", "")).strip()
                    if q:
                        all_split_results[q] = item

        t0 = time.time()
        tasks = []
        for batch in split_pending:
            tasks.append(asyncio.create_task(run_split_batch(batch)))
            await asyncio.sleep(LAUNCH_INTERVAL)

        done = 0
        for t in asyncio.as_completed(tasks):
            try:
                await t
            except Exception as exc:
                print(f"[split] batch failed: {exc}")
            done += 1
            if done % 10 == 0:
                print(f"[split] progress: {done}/{len(split_pending)} batches")

    elapsed = time.time() - t0
    print(f"[split] done in {elapsed:.0f}s, {len(all_split_results)} labeled")

    # Write split output
    with SPLIT_OUTPUT.open("w", encoding="utf-8") as f:
        for qtext in query_texts:
            row = all_split_results.get(qtext, {})
            row.setdefault("query", qtext)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[split] saved to {SPLIT_OUTPUT}")

    print(f"\nDone! Results:")
    print(f"  Recall: {RECALL_OUTPUT}")
    print(f"  Split:  {SPLIT_OUTPUT}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate benchmark labels via NVIDIA API")
    p.add_argument("--nvapi_key", default="", help="NVIDIA API key (or set NVAPI_KEY env)")
    p.add_argument("--recall_batch_size", default=4, type=int, help="queries per recall API call")
    p.add_argument("--split_batch_size", default=10, type=int, help="queries per split API call")
    p.add_argument("--max_in_flight", default=5, type=int, help="max concurrent API calls")
    p.add_argument("--resume", action="store_true", help="resume from existing output files")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
