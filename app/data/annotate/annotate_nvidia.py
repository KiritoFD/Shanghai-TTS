"""NVIDIA API-based annotation script for Shanghai dialect TTS segmentation.

Reads pending sentences from output.xlsx, batches them into JSONL tasks,
calls the NVIDIA-hosted LLM API via ThreadPoolExecutor + requests (no aiohttp),
validates responses, and writes results back to output.xlsx and to JSONL files.

Usage
-----
    python app/data/annotate/annotate_nvidia.py \\
        --write_back_xlsx \\
        --resume

    # use compact system prompt:
    python app/data/annotate/annotate_nvidia.py --use_compact_prompt --write_back_xlsx --resume

Environment
-----------
    NVAPI_KEY   — NVIDIA API bearer token (or pass via --config_json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    import requests
except Exception:  # noqa: BLE001
    requests = None

try:
    import pandas as pd
except Exception as _pd_exc:  # noqa: BLE001
    pd = None  # type: ignore[assignment]
    _PD_IMPORT_ERROR = _pd_exc
else:
    _PD_IMPORT_ERROR = None

from urllib import request as urllib_request

# ── constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]

VALID_TAGS: frozenset[str] = frozenset({
    "N", "V", "A", "VN", "M", "Q", "R", "D", "P", "C",
    "SP", "AS", "Y", "FW", "ND", "I", "O", "IDM", "PU", "X",
})

# SP must never appear as a standalone token
_SP_STANDALONE_WARNING = (
    "SP tag found as standalone token — SP must be merged with preceding token"
)

# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NVIDIA API annotation for Shanghai TTS segmentation/POS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_xlsx",
        default="app/data/annotate/output.xlsx",
        help="Path to output.xlsx with sentence/annotation columns",
    )
    p.add_argument(
        "--output_jsonl",
        default="app/data/annotate/results/annotated.jsonl",
        help="Destination JSONL for all successful LLM annotations",
    )
    p.add_argument(
        "--failures_jsonl",
        default="app/data/annotate/failures/failures.jsonl",
        help="Destination JSONL for failed batches",
    )
    p.add_argument(
        "--tasks_jsonl",
        default="app/data/annotate/tasks/pending_tasks.jsonl",
        help="Intermediate JSONL of pending tasks (auto-built from xlsx)",
    )
    p.add_argument(
        "--system_prompt",
        default="app/seg_pos/prompts/tts_seg_annotation_v2_system_prompt.txt",
        help="Path to full system prompt text file",
    )
    p.add_argument(
        "--system_prompt_compact",
        default="app/seg_pos/prompts/tts_seg_annotation_compact_system_prompt.txt",
        help="Path to compact system prompt text file (use with --use_compact_prompt)",
    )
    p.add_argument(
        "--use_compact_prompt",
        action="store_true",
        help="Use compact system prompt instead of full prompt",
    )
    p.add_argument(
        "--api_base",
        default="https://integrate.api.nvidia.com/v1",
        help="OpenAI-compatible API base URL",
    )
    p.add_argument(
        "--model",
        default="qwen/qwen3.5-397b-a17b",
        help="Model identifier",
    )
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--max_in_flight", default=10, type=int)
    p.add_argument("--temperature", default=0.0, type=float)
    p.add_argument("--max_tokens", default=4096, type=int)
    p.add_argument("--timeout_s", default=180, type=int)
    p.add_argument("--max_retries", default=3, type=int)
    p.add_argument("--launch_interval_s", default=2.0, type=float)
    p.add_argument(
        "--config_json",
        default="",
        help="Optional JSON file with api_key and overrides",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip sentence_ids already present in output_jsonl",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Build tasks only; do not send any API requests",
    )
    p.add_argument(
        "--write_back_xlsx",
        action="store_true",
        help="Write annotation column back to input_xlsx after completion",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-batch debug info to stderr",
    )
    return p.parse_args()


# ── config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── xlsx reading (zero-dependency, via build_seg_pos_data helpers) ────────────

def _require_pandas(operation: str) -> None:
    if pd is None:
        raise RuntimeError(
            f"pandas is required for {operation}. "
            f"Install it with: pip install pandas openpyxl. "
            f"Original error: {_PD_IMPORT_ERROR}"
        )


def read_xlsx_sentences(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (rows, column_names) where rows is a list of dicts from the xlsx.

    Each row dict has at minimum 'sentence' and 'annotation' keys.
    Rows whose 'sentence' is empty are preserved (with empty sentence) so
    that write-back can round-trip to the exact same row indices.
    """
    _require_pandas("reading xlsx")
    df = pd.read_excel(path, dtype=str, keep_default_na=False)
    # Normalise column names: ensure 'sentence' and 'annotation' exist
    if "sentence" not in df.columns:
        df.columns = ["sentence"] + list(df.columns[1:])
    if "annotation" not in df.columns:
        df["annotation"] = ""
    rows: list[dict[str, Any]] = df.to_dict(orient="records")
    return rows, list(df.columns)


def write_xlsx_annotation(
    path: Path,
    rows: list[dict[str, Any]],
    annotation_map: dict[str, str],
) -> int:
    """Update the annotation column using annotation_map (sentence -> annotation).

    Returns the number of cells updated.
    """
    _require_pandas("writing xlsx")
    updated = 0
    for row in rows:
        sentence = str(row.get("sentence", "")).strip()
        if sentence in annotation_map:
            existing = str(row.get("annotation", "")).strip()
            if not existing:
                row["annotation"] = annotation_map[sentence]
                updated += 1
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    return updated


# ── task building ─────────────────────────────────────────────────────────────

def build_pending_tasks(
    rows: list[dict[str, Any]],
    done_sentence_ids: set[str],
) -> list[dict[str, Any]]:
    """Build task list from xlsx rows, assigning output-XXXXX IDs.

    Skips rows that are already annotated OR already in done_sentence_ids.
    Uses 1-based index over all non-empty sentences (matching build_tts_sentence_tasks.py).
    """
    tasks: list[dict[str, Any]] = []
    index = 0  # will become 1-based sentence counter
    for row in rows:
        sentence = str(row.get("sentence", "")).strip()
        if not sentence or sentence == "nan":
            continue
        index += 1
        sentence_id = f"output-{index:05d}"
        annotation = str(row.get("annotation", "")).strip()
        if annotation and annotation != "nan":
            continue  # already annotated
        if sentence_id in done_sentence_ids:
            continue  # already processed in a previous run
        tasks.append({
            "sentence_id": sentence_id,
            "sentence": sentence,
            "source": "output.xlsx",
        })
    return tasks


def save_tasks_jsonl(tasks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")


# ── JSONL helpers ─────────────────────────────────────────────────────────────

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
                raise RuntimeError(
                    f"invalid jsonl at {path}:{line_no}: {exc}"
                ) from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def chunked(
    items: list[dict[str, Any]], size: int
) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ── validation ────────────────────────────────────────────────────────────────

def validate_item(
    item: dict[str, Any],
    expected_sentence: str,
) -> tuple[bool, str, list[str]]:
    """Validate a single LLM-returned annotation item.

    Returns (ok, error_reason, warnings).
    Warnings are non-fatal issues recorded into the uncertain list.
    """
    warnings: list[str] = []

    sentence = str(item.get("sentence", ""))
    tokens: Any = item.get("tokens", [])
    token_tags: Any = item.get("token_tags", [])
    tone_domain_ids: Any = item.get("tone_domain_ids", [])
    uncertain: Any = item.get("uncertain", [])

    if sentence != expected_sentence:
        return False, "sentence_mismatch", warnings

    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        return False, "tokens_invalid_type", warnings

    if any(t == "" for t in tokens):
        return False, "tokens_contain_empty_string", warnings

    if "".join(tokens) != expected_sentence:
        return False, "tokens_do_not_reconstruct_sentence", warnings

    if not isinstance(token_tags, list) or len(token_tags) != len(tokens):
        return False, "token_tags_length_mismatch", warnings

    if not isinstance(tone_domain_ids, list) or len(tone_domain_ids) != len(tokens):
        return False, "tone_domain_ids_length_mismatch", warnings

    invalid_tags = [str(tag) for tag in token_tags if str(tag) not in VALID_TAGS]
    if invalid_tags:
        return False, f"invalid_tags={invalid_tags}", warnings

    if not all(isinstance(d, int) and d >= 0 for d in tone_domain_ids):
        return False, "tone_domain_ids_not_nonneg_int", warnings

    if not isinstance(uncertain, list):
        return False, "uncertain_not_list", warnings

    # Non-fatal: SP as standalone token
    for idx, (token, tag) in enumerate(zip(tokens, token_tags)):
        if str(tag) == "SP":
            msg = f"SP_standalone_token idx={idx} span='{token}'"
            warnings.append(msg)

    return True, "", warnings


# ── LLM prompt construction ───────────────────────────────────────────────────

def build_user_text(batch: list[dict[str, Any]]) -> str:
    lines = [
        "请按顺序标注以下句子，并返回 JSON 数组。",
        "每句必须包含 sentence_id, sentence, tokens, token_tags, "
        "tone_domain_ids, uncertain, notes, teacher_model 字段。",
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


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def post_chat_blocking(
    api_base: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    if requests is not None:
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[return-value]

    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url=f"{api_base.rstrip('/')}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def request_with_retry_sync(
    *,
    api_base: str,
    headers: dict[str, str],
    model: str,
    system_prompt: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
    verbose: bool,
    batch_tag: str,
    session: Any = None,  # requests.Session if available
) -> str:
    """Synchronous retry loop using requests.Session (or urllib fallback)."""
    payload: dict[str, Any] = {
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
            if session is not None:
                # Use the provided requests.Session for connection reuse
                resp = session.post(
                    f"{api_base.rstrip('/')}/chat/completions",
                    json=payload,
                    timeout=timeout_s,
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
            else:
                data = post_chat_blocking(api_base, headers, payload, timeout_s)
            text = extract_text_from_response(data)
            if not text:
                raise RuntimeError("empty_model_text")
            return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if verbose:
                print(
                    f"[retry] {batch_tag} attempt={attempt + 1}/{max_retries} "
                    f"error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            if attempt + 1 < max_retries:
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(str(last_error) if last_error else "request_failed")


# ── annotation format conversion ─────────────────────────────────────────────

def annotation_from_llm_item(item: dict[str, Any]) -> str:
    """Convert LLM output item to annotation string: 词/TAG#词/TAG#...

    Tags are kept in their original case (already uppercase from LLM).
    """
    tokens: list[str] = item["tokens"]
    token_tags: list[str] = [str(t) for t in item["token_tags"]]
    return "#".join(f"{w}/{t}" for w, t in zip(tokens, token_tags))


# ── main synchronous orchestration (ThreadPoolExecutor) ──────────────────────────

def main_thread_pool(args: argparse.Namespace) -> None:  # noqa: C901 — intentionally long
    cfg = load_config(args.config_json)
    api_key: str = cfg.get("api_key") or os.environ.get("NVAPI_KEY", "")
    if not api_key and not args.dry_run:
        raise RuntimeError(
            "NVAPI_KEY environment variable is not set and "
            "config_json.api_key is missing. "
            "Pass the key via: export NVAPI_KEY=nvapi-..."
        )

    model: str = args.model or cfg.get("model") or "qwen/qwen3.5-397b-a17b"
    api_base: str = str(cfg.get("api_base", args.api_base))
    temperature: float = float(cfg.get("temperature", args.temperature))
    max_tokens: int = int(cfg.get("max_tokens", args.max_tokens))
    batch_size: int = int(cfg.get("batch_size", args.batch_size))
    timeout_s: int = int(cfg.get("timeout_s", args.timeout_s))
    max_retries: int = int(cfg.get("max_retries", args.max_retries))
    launch_interval_s: float = float(cfg.get("launch_interval_s", args.launch_interval_s))
    max_in_flight: int = int(cfg.get("max_in_flight", args.max_in_flight))

    # config_json can also set use_compact_prompt
    use_compact_prompt: bool = bool(
        cfg.get("use_compact_prompt", False)
    ) or args.use_compact_prompt

    # ── resolve paths relative to repo root if not absolute ──────────────────────
    def _resolve(raw: str) -> Path:
        p = Path(raw)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p

    input_xlsx_path = _resolve(args.input_xlsx)
    output_jsonl_path = _resolve(args.output_jsonl)
    failures_jsonl_path = _resolve(args.failures_jsonl)
    tasks_jsonl_path = _resolve(args.tasks_jsonl)

    # Select system prompt based on flag
    if use_compact_prompt:
        system_prompt_path = _resolve(args.system_prompt_compact)
    else:
        system_prompt_path = _resolve(args.system_prompt)

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    failures_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    print(f"[config] model={model}", file=sys.stderr)
    print(
        f"[config] system_prompt={'compact' if use_compact_prompt else 'full'}: "
        f"{system_prompt_path}",
        file=sys.stderr,
    )
    print(f"[config] input_xlsx={input_xlsx_path}", file=sys.stderr)

    # ── read xlsx ──────────────────────────────────────────────────────────────
    print("[info] reading input xlsx...", file=sys.stderr)
    xlsx_rows, _columns = read_xlsx_sentences(input_xlsx_path)
    total_sentences = sum(
        1 for r in xlsx_rows
        if str(r.get("sentence", "")).strip() not in ("", "nan")
    )
    already_annotated = sum(
        1 for r in xlsx_rows
        if str(r.get("sentence", "")).strip() not in ("", "nan")
        and str(r.get("annotation", "")).strip() not in ("", "nan")
    )
    print(
        f"[info] xlsx: total={total_sentences} already_annotated={already_annotated} "
        f"pending={total_sentences - already_annotated}",
        file=sys.stderr,
    )

    # ── determine already-done sentence IDs from output jsonl ─────────────────────
    done_sentence_ids: set[str] = set()
    if args.resume and output_jsonl_path.exists():
        for row in read_jsonl(output_jsonl_path):
            sid = str(row.get("sentence_id", "")).strip()
            if sid:
                done_sentence_ids.add(sid)
        print(
            f"[resume] found {len(done_sentence_ids)} completed sentence_ids in output jsonl",
            file=sys.stderr,
        )

    # ── build pending tasks ─────────────────────────────────────────────────
    pending_tasks = build_pending_tasks(xlsx_rows, done_sentence_ids)
    print(f"[info] pending tasks to annotate: {len(pending_tasks)}", file=sys.stderr)
    save_tasks_jsonl(pending_tasks, tasks_jsonl_path)
    print(f"[info] tasks written to {tasks_jsonl_path}", file=sys.stderr)

    if args.dry_run:
        print("[dry_run] tasks built; no API requests will be sent.", file=sys.stderr)
        print(f"dry_run_task_count={len(pending_tasks)}")
        return

    if not pending_tasks:
        print("[info] nothing to annotate — all sentences are already done.", file=sys.stderr)
        return

    # ── batch the tasks ───────────────────────────────────────────────────────────
    batches = chunked(pending_tasks, batch_size)
    print(
        f"[info] batches={len(batches)} batch_size={batch_size} max_in_flight={max_in_flight}",
        file=sys.stderr,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    write_lock = threading.Lock()
    start_time = time.time()

    # annotation_map accumulates sentence -> annotation string for xlsx write-back
    annotation_map: dict[str, str] = {}

    out_mode = "a" if args.resume else "w"
    fail_mode = "a" if args.resume else "w"

    # Each worker thread gets its own requests.Session for connection reuse
    _thread_local = threading.local()

    def _get_session() -> Any:
        """Return a per-thread requests.Session, creating one if needed."""
        if requests is None:
            return None
        if not hasattr(_thread_local, "session"):
            sess = requests.Session()
            sess.headers.update(headers)
            _thread_local.session = sess
        return _thread_local.session

    def write_failure(
        fail_handle: Any,
        payload: dict[str, Any],
    ) -> None:
        with write_lock:
            fail_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fail_handle.flush()
        if args.verbose:
            print(
                f"[failure] batch_index={payload.get('batch_index')} "
                f"error={payload.get('error', '')[:120]}",
                file=sys.stderr,
            )

    def write_results(
        out_handle: Any,
        normalized_rows: list[dict[str, Any]],
    ) -> None:
        with write_lock:
            for row in normalized_rows:
                out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                annotation_map[row["sentence"]] = annotation_from_llm_item(row)
            out_handle.flush()

    def run_batch(
        batch_index: int,
        batch: list[dict[str, Any]],
        out_handle: Any,
        fail_handle: Any,
    ) -> None:
        """Process one batch synchronously (runs in a thread-pool worker)."""
        batch_tag = f"batch={batch_index}/{len(batches)} size={len(batch)}"
        launched_at = time.time()
        if args.verbose:
            print(f"[launch] {batch_tag}", file=sys.stderr)

        sess = _get_session()
        user_text = build_user_text(batch)
        raw_text: str = ""
        try:
            raw_text = request_with_retry_sync(
                api_base=api_base,
                headers=headers,
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                max_retries=max_retries,
                verbose=args.verbose,
                batch_tag=batch_tag,
                session=sess,
            )
            parsed_items = parse_json_array(raw_text)
        except json.JSONDecodeError as exc:
            write_failure(fail_handle, {
                "batch_index": batch_index,
                "batch_sentence_ids": [str(r["sentence_id"]) for r in batch],
                "teacher_model": model,
                "error": f"json_decode_error: {exc}",
                "raw_response": raw_text,
            })
            return
        except ValueError as exc:
            write_failure(fail_handle, {
                "batch_index": batch_index,
                "batch_sentence_ids": [str(r["sentence_id"]) for r in batch],
                "teacher_model": model,
                "error": f"parse_json_array: {exc}",
            })
            return
        except Exception as exc:  # noqa: BLE001
            write_failure(fail_handle, {
                "batch_index": batch_index,
                "batch_sentence_ids": [str(r["sentence_id"]) for r in batch],
                "teacher_model": model,
                "error": f"{type(exc).__name__}: {exc}",
            })
            return

        expected_map = {
            str(r["sentence_id"]): str(r["sentence"]) for r in batch
        }

        if len(parsed_items) != len(batch):
            write_failure(fail_handle, {
                "batch_index": batch_index,
                "batch_sentence_ids": list(expected_map.keys()),
                "teacher_model": model,
                "error": (
                    f"batch_size_mismatch expected={len(batch)} "
                    f"got={len(parsed_items)}"
                ),
                "raw_response": raw_text,
            })
            return

        normalized_rows: list[dict[str, Any]] = []
        batch_ok = True
        for item in parsed_items:
            sentence_id = str(item.get("sentence_id", "")).strip()
            expected_sentence = expected_map.get(sentence_id, "")
            if not expected_sentence:
                write_failure(fail_handle, {
                    "batch_index": batch_index,
                    "batch_sentence_ids": list(expected_map.keys()),
                    "teacher_model": model,
                    "error": f"unexpected_sentence_id={sentence_id!r}",
                    "raw_item": item,
                })
                batch_ok = False
                break

            ok, reason, item_warnings = validate_item(item, expected_sentence)
            if not ok:
                write_failure(fail_handle, {
                    "batch_index": batch_index,
                    "batch_sentence_ids": list(expected_map.keys()),
                    "teacher_model": model,
                    "error": reason,
                    "raw_item": item,
                })
                batch_ok = False
                break

            # Merge SP-standalone warnings into the item's uncertain list
            existing_uncertain: list[Any] = list(item.get("uncertain") or [])
            for warn_msg in item_warnings:
                existing_uncertain.append({
                    "span": warn_msg,
                    "token_index": -1,
                    "candidates": [],
                    "preferred": "",
                    "reason": _SP_STANDALONE_WARNING,
                })
                print(
                    f"[warn] sentence_id={sentence_id} {warn_msg}",
                    file=sys.stderr,
                )

            normalized_rows.append({
                "sentence_id": sentence_id,
                "sentence": expected_sentence,
                "tokens": item["tokens"],
                "token_tags": [str(t) for t in item["token_tags"]],
                "tone_domain_ids": item["tone_domain_ids"],
                "uncertain": existing_uncertain,
                "notes": item.get("notes", ""),
                "teacher_model": model,
            })

        if batch_ok and normalized_rows:
            write_results(out_handle, normalized_rows)
            if args.verbose:
                print(
                    f"[done] {batch_tag} "
                    f"elapsed_s={time.time() - launched_at:.1f}",
                    file=sys.stderr,
                )

    # ── submit all batches to the thread pool ───────────────────────────────────
    with (
        output_jsonl_path.open(out_mode, encoding="utf-8") as out_handle,
        failures_jsonl_path.open(fail_mode, encoding="utf-8") as fail_handle,
        ThreadPoolExecutor(max_workers=max_in_flight) as executor,
    ):
        futures = {}
        for batch_index, batch in enumerate(batches):
            fut = executor.submit(run_batch, batch_index, batch, out_handle, fail_handle)
            futures[fut] = batch_index
            if launch_interval_s > 0:
                time.sleep(launch_interval_s)

        completed = 0
        for fut in as_completed(futures):
            # Propagate unexpected exceptions from worker threads
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[error] unhandled exception in batch worker: {exc}",
                    file=sys.stderr,
                )
            completed += 1
            if completed % 5 == 0 or completed == len(futures):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0.0
                approx_done_sentences = completed * batch_size
                print(
                    f"completed_batches={completed}/{len(futures)} "
                    f"~sentences={approx_done_sentences} "
                    f"elapsed_s={elapsed:.1f} "
                    f"rate_batches/s={rate:.2f}"
                )

    # ── optional xlsx write-back ──────────────────────────────────────────────────
    if args.write_back_xlsx and annotation_map:
        print(
            f"[write_back] writing {len(annotation_map)} annotations to {input_xlsx_path}...",
            file=sys.stderr,
        )
        # Re-read xlsx rows (they may have been updated on disk by another process)
        xlsx_rows, _ = read_xlsx_sentences(input_xlsx_path)
        updated = write_xlsx_annotation(input_xlsx_path, xlsx_rows, annotation_map)
        print(
            f"[write_back] done — updated {updated} cells in {input_xlsx_path}",
            file=sys.stderr,
        )
    elif args.write_back_xlsx and not annotation_map:
        print(
            "[write_back] no new annotations to write back (annotation_map is empty).",
            file=sys.stderr,
        )

    # ── final summary ──────────────────────────────────────────────────────────────
    elapsed_total = time.time() - start_time
    success_count = len(annotation_map)
    failure_count = len(pending_tasks) - success_count
    print(
        f"summary: total_pending={len(pending_tasks)} "
        f"annotated={success_count} "
        f"failed={failure_count} "
        f"elapsed_s={elapsed_total:.1f}"
    )
    if failures_jsonl_path.exists() and failures_jsonl_path.stat().st_size > 0:
        print(
            f"[info] failed batches saved to {failures_jsonl_path}",
            file=sys.stderr,
        )
        print(
            "[info] rerun with --resume to retry failed sentences.",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()
    main_thread_pool(args)


if __name__ == "__main__":
    main()
