"""Shanghai-TTS full pipeline test.

Tests the complete pipeline: split (llama.cpp) -> recall -> TTS.
Usage:
    python test_pipeline.py --skip-tts        # Skip TTS (GPU-heavy)
    python test_pipeline.py --full            # Full pipeline including TTS
    python test_pipeline.py --split-only      # Only test split
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT))

TEST_QUERIES = [
    "自行车",
    "开心怎么说",
    "阿拉上海闲话",
    "两头作虎状",
    "夸人厉害",
    "嘎讪胡",
    "螺蛳壳里做道场",
    "吃饭",
    "谢谢侬",
    "今天天气好",
    "不小心",
    "很厉害",
    "白相",
    "小囡",
    "作孽",
]


def test_split(queries: list[str]) -> dict:
    from split.llama_cpp_runtime import LlamaCppConfig, LlamaCppQueryPreprocessor
    from path_config import load_runtime_config
    from app import ZH_PROMPT

    config = load_runtime_config()
    llm = config["llm"]

    cpp_config = LlamaCppConfig(
        server_path=Path(llm["llama_cpp_server_path"]),
        gguf_model_path=Path(llm["gguf_model_path"]),
        host=str(llm.get("llama_cpp_host", "127.0.0.1")),
        port=int(llm.get("llama_cpp_port", 8091)),
        ctx_size=int(llm.get("llama_cpp_ctx_size", 2048)),
        n_gpu_layers=int(llm.get("llama_cpp_n_gpu_layers", 99)),
        threads=int(llm.get("llama_cpp_threads", 8)),
        batch_size=int(llm.get("llama_cpp_batch_size", 256)),
        ubatch_size=int(llm.get("llama_cpp_ubatch_size", 256)),
        timeout_seconds=int(llm.get("llama_cpp_timeout_seconds", 30)),
        auto_start=bool(llm.get("llama_cpp_auto_start", True)),
        flash_attention=bool(llm.get("llama_cpp_flash_attention", True)),
    )

    print(f"[split] Initializing llama.cpp preprocessor...")
    print(f"  server: {cpp_config.server_path}")
    print(f"  model:  {cpp_config.gguf_model_path}")
    print(f"  host:   {cpp_config.host}:{cpp_config.port}")

    t0 = time.perf_counter()
    preprocessor = LlamaCppQueryPreprocessor(
        config=cpp_config,
        base_model_path=Path(llm["base_model_dir"]),
        system_prompt=ZH_PROMPT,
    )
    load_s = time.perf_counter() - t0
    print(f"[split] Preprocessor loaded in {load_s:.1f}s")

    results = []
    times = []
    valid_json = 0
    has_core_text = 0
    has_type = 0
    has_keywords = 0

    print(f"\n[split] Testing {len(queries)} queries...")
    print("-" * 80)

    for i, query in enumerate(queries):
        ts = time.perf_counter()
        parsed = preprocessor.preprocess(query)
        elapsed = time.perf_counter() - ts
        times.append(elapsed)

        is_valid = bool(parsed and parsed.get("core_text"))
        if parsed and "{" in str(parsed):
            valid_json += 1
        if parsed.get("core_text"):
            has_core_text += 1
        if parsed.get("type"):
            has_type += 1
        if parsed.get("keywords"):
            has_keywords += 1

        raw = parsed.get("_raw", "")[:60]
        results.append({
            "query": query,
            "parsed": {k: v for k, v in parsed.items() if k != "_raw"},
            "raw_preview": raw,
            "time_s": round(elapsed, 3),
        })

        status = "OK" if parsed.get("core_text") else "WARN"
        print(f"  [{status}] {query:>20s} -> core_text={parsed.get('core_text','')!r:15s} "
              f"type={parsed.get('type','')!r:8s} "
              f"kw={parsed.get('keywords',[])} "
              f"({elapsed*1000:.0f}ms)")
        if raw:
            print(f"        raw: {raw}")

    print("-" * 80)
    n = len(queries)
    avg_ms = sum(times) / max(n, 1) * 1000
    print(f"\n[split] Results:")
    print(f"  valid_json:    {valid_json}/{n} ({valid_json/n*100:.0f}%)")
    print(f"  has_core_text: {has_core_text}/{n} ({has_core_text/n*100:.0f}%)")
    print(f"  has_type:      {has_type}/{n} ({has_type/n*100:.0f}%)")
    print(f"  has_keywords:  {has_keywords}/{n} ({has_keywords/n*100:.0f}%)")
    print(f"  avg_latency:   {avg_ms:.0f}ms")
    print(f"  total_time:    {sum(times):.1f}s")

    preprocessor.service.shutdown()

    return {
        "results": results,
        "valid_json_rate": valid_json / n,
        "has_core_text_rate": has_core_text / n,
        "avg_ms": avg_ms,
    }


def test_recall(queries: list[str]) -> dict:
    from app import ensure_preprocessor, build_search_terms, call_recall_service, rerank_recall_results, df, local_search

    print(f"\n[recall] Testing recall for {len(queries)} queries...")
    print("-" * 80)

    preprocessor = ensure_preprocessor()
    results_data = []
    times = []
    hit_count = 0

    for query in queries:
        ts = time.perf_counter()
        parsed = preprocessor.preprocess(query)
        search_terms = build_search_terms(query, parsed)
        results, source = call_recall_service(query, variants=search_terms, top_k=20, top_n=10)
        results = rerank_recall_results(query, parsed, results)[:3]
        if not results:
            for term in search_terms[:3]:
                results = local_search(df, term, top_n=3)
                if results:
                    source = "local_search"
                    break
        elapsed = time.perf_counter() - ts
        times.append(elapsed)

        hit = bool(results)
        if hit:
            hit_count += 1

        top_items = [
            {"shanghai": r.get("shanghai", ""), "definition": r.get("definition", "")[:30]}
            for r in results[:2]
        ]
        results_data.append({
            "query": query,
            "hit": hit,
            "source": source,
            "top_results": top_items,
            "time_s": round(elapsed, 3),
        })

        status = "OK" if hit else "MISS"
        top_str = " | ".join(r.get("shanghai", "") for r in results[:2])
        print(f"  [{status}] {query:>20s} -> source={source:15s} top=[{top_str}] ({elapsed*1000:.0f}ms)")

    print("-" * 80)
    n = len(queries)
    avg_ms = sum(times) / max(n, 1) * 1000
    print(f"\n[recall] Results:")
    print(f"  hit_rate:    {hit_count}/{n} ({hit_count/n*100:.0f}%)")
    print(f"  avg_latency: {avg_ms:.0f}ms")

    return {
        "results": results_data,
        "hit_rate": hit_count / n,
        "avg_ms": avg_ms,
    }


def test_tts(pinyin_texts: list[str]) -> dict:
    from tts_engine import synthesize
    import tempfile

    print(f"\n[tts] Testing TTS for {len(pinyin_texts)} inputs...")
    print("-" * 80)

    results = []
    times = []
    success = 0
    output_dir = Path(APP_ROOT / "static")
    output_dir.mkdir(exist_ok=True)

    for pinyin in pinyin_texts:
        out_path = str(output_dir / f"test_tts_{hash(pinyin) % 10000}.wav")
        ts = time.perf_counter()
        try:
            synthesize(pinyin, out_path)
            elapsed = time.perf_counter() - ts
            times.append(elapsed)
            success += 1
            file_size = Path(out_path).stat().st_size if Path(out_path).exists() else 0
            results.append({"pinyin": pinyin, "time_s": round(elapsed, 3), "file_size": file_size, "ok": True})
            print(f"  [OK] {pinyin:>30s} -> {file_size:,d} bytes ({elapsed*1000:.0f}ms)")
        except Exception as exc:
            elapsed = time.perf_counter() - ts
            results.append({"pinyin": pinyin, "time_s": round(elapsed, 3), "error": str(exc), "ok": False})
            print(f"  [FAIL] {pinyin:>30s} -> ERROR: {exc}")

    print("-" * 80)
    n = len(pinyin_texts)
    avg_ms = sum(times) / max(len(times), 1) * 1000
    print(f"\n[tts] Results:")
    print(f"  success:     {success}/{n}")
    print(f"  avg_latency: {avg_ms:.0f}ms")

    return {
        "results": results,
        "success_rate": success / n,
        "avg_ms": avg_ms,
    }


def test_full_pipeline(queries: list[str]) -> dict:
    from app import (
        ensure_preprocessor, build_search_terms, call_recall_service,
        rerank_recall_results, local_search, find_headword_row,
        synthesize, df, STATIC_DIR,
    )

    print(f"\n[pipeline] Testing FULL pipeline for {len(queries)} queries...")
    print("=" * 80)

    preprocessor = ensure_preprocessor()
    results = []
    times = []

    for query in queries:
        stage_times = {}
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        parsed = preprocessor.preprocess(query)
        stage_times["split"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        search_terms = build_search_terms(query, parsed)
        stage_times["build_terms"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        recall_results, source = call_recall_service(query, variants=search_terms, top_k=20, top_n=10)
        recall_results = rerank_recall_results(query, parsed, recall_results)[:3]
        if not recall_results:
            for term in search_terms[:3]:
                recall_results = local_search(df, term, top_n=3)
                if recall_results:
                    source = "local_search"
                    break
        stage_times["recall"] = time.perf_counter() - t0

        tts_ok = False
        tts_path = ""
        if recall_results:
            top = recall_results[0]
            shanghai = str(top.get("shanghai", "")).strip().strip("[]").strip("\u3010\u3011")
            row = find_headword_row(shanghai) if shanghai else None
            if row is not None and len(row) > 5:
                wu_pinyin = str(row[5]).strip()
                if wu_pinyin:
                    import uuid
                    t0 = time.perf_counter()
                    try:
                        fn = f"{uuid.uuid4().hex[:8]}.wav"
                        synthesize(wu_pinyin, str(STATIC_DIR / fn))
                        stage_times["tts"] = time.perf_counter() - t0
                        tts_ok = True
                        tts_path = f"/static/{fn}"
                    except Exception as exc:
                        stage_times["tts"] = time.perf_counter() - t0
                        tts_ok = False

        total = time.perf_counter() - t_total
        times.append(total)

        top_sh = recall_results[0].get("shanghai", "") if recall_results else ""
        top_def = recall_results[0].get("definition", "")[:25] if recall_results else ""
        status = "OK" if recall_results else "MISS"
        tts_status = "TTS" if tts_ok else "---"

        print(f"  [{status}|{tts_status}] {query:>15s} "
              f"-> {top_sh[:12]:12s} | {top_def[:25]:25s} "
              f"split={stage_times.get('split',0)*1000:5.0f}ms "
              f"recall={stage_times.get('recall',0)*1000:5.0f}ms "
              f"tts={stage_times.get('tts',0)*1000:5.0f}ms "
              f"total={total*1000:5.0f}ms")

        results.append({
            "query": query,
            "parsed": {k: v for k, v in parsed.items() if k != "_raw"},
            "top_result": top_sh,
            "source": source,
            "tts_ok": tts_ok,
            "stages_ms": {k: round(v * 1000, 1) for k, v in stage_times.items()},
            "total_ms": round(total * 1000, 1),
        })

    print("=" * 80)
    n = len(queries)
    avg_ms = sum(times) / max(n, 1) * 1000
    split_ms = [r["stages_ms"].get("split", 0) for r in results]
    recall_ms = [r["stages_ms"].get("recall", 0) for r in results]
    tts_ms = [r["stages_ms"].get("tts", 0) for r in results if "tts" in r["stages_ms"]]

    print(f"\n[pipeline] Summary:")
    print(f"  avg total:     {avg_ms:.0f}ms")
    print(f"  avg split:     {sum(split_ms)/max(len(split_ms),1):.0f}ms")
    print(f"  avg recall:    {sum(recall_ms)/max(len(recall_ms),1):.0f}ms")
    if tts_ms:
        print(f"  avg tts:       {sum(tts_ms)/max(len(tts_ms),1):.0f}ms")

    return {
        "results": results,
        "avg_total_ms": avg_ms,
        "avg_split_ms": sum(split_ms) / max(len(split_ms), 1),
        "avg_recall_ms": sum(recall_ms) / max(len(recall_ms), 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shanghai-TTS pipeline test")
    parser.add_argument("--full", action="store_true", help="Run full pipeline including TTS")
    parser.add_argument("--split-only", action="store_true", help="Only test split")
    parser.add_argument("--skip-tts", action="store_true", help="Skip TTS test")
    parser.add_argument("--queries", type=int, default=10, help="Number of test queries")
    args = parser.parse_args()

    queries = TEST_QUERIES[:args.queries]
    print("=" * 80)
    print("  Shanghai-TTS Pipeline Test")
    print("=" * 80)

    all_results = {}

    if args.split_only:
        all_results["split"] = test_split(queries)
    elif args.full:
        all_results["pipeline"] = test_full_pipeline(queries)
    else:
        all_results["split"] = test_split(queries)
        all_results["recall"] = test_recall(queries)
        if not args.skip_tts:
            tts_inputs = ["shi33 yan55 kua33 chi21", "ngu23 teq5 maon22"]
            all_results["tts"] = test_tts(tts_inputs)

    out_path = REPO_ROOT / "test_pipeline_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
