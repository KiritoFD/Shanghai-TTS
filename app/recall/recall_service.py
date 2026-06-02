from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from .engine import RecallEngine, discover_indexes
except ImportError:
    from engine import RecallEngine, discover_indexes


def safe_print(text: str) -> None:
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def normalize_source_name(value: str) -> str:
    source = str(value or "").strip().lower()
    if source in {"", "shanghai", "primary", "default", "shanghai_csv"}:
        return "shanghai"
    if source in {"shaoxing", "shaoxing_xlsx"}:
        return "shaoxing"
    return source


def print_result(result: dict[str, Any]) -> None:
    safe_print(f"query={result['query']}")
    extra_notes: list[str] = []
    for row in result["results"]:
        safe_print(f"output:[{row['shanghai']}] score={row['score']:.4f}")
        note = str(row.get("notes", "")).strip()
        if note:
            extra_notes.append(f"  [{row['shanghai']}] {note}")
    if extra_notes:
        safe_print("--- notes ---")
        for line in extra_notes:
            safe_print(line)
    safe_print(f"infer_s={result['infer_s']:.4f}\n")


def build_api_handler(engines: dict[str, RecallEngine], top_k: int, top_n: int):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                indexes = {
                    name: {
                        "device": str(engine.device),
                        "index_dir": str(engine.index_dir),
                        "ann": engine.ann,
                        "load_s": engine.load_s,
                    }
                    for name, engine in engines.items()
                }
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "default_source": "shanghai",
                        "indexes": indexes,
                    },
                )
                return
            self._send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/recall":
                self._send_json(404, {"ok": False, "error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                data = json.loads(raw.decode("utf-8")) if raw else {}
                query = str(data.get("query", "")).strip()
                if not query:
                    self._send_json(400, {"ok": False, "error": "query_required"})
                    return
                k = int(data.get("top_k", top_k))
                n = int(data.get("top_n", top_n))
                variants = data.get("variants", [])
                if not isinstance(variants, list):
                    variants = []
                source = normalize_source_name(str(data.get("source", "shanghai")))
                engine = engines.get(source)
                if engine is None:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": "unknown_source",
                            "source": source,
                            "available_sources": sorted(engines.keys()),
                        },
                    )
                    return
                result = engine.search(query, top_k=k, top_n=n, extra_variants=variants)
                self._send_json(200, {"ok": True, "source": source, "result": result})
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"ok": False, "error": str(exc)})

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return Handler


def choose_index_interactive(indexes: list[Path]) -> Path:
    safe_print("available index directories:")
    for i, path in enumerate(indexes, start=1):
        safe_print(f"{i}. {path.name}")
    while True:
        choice = input("select index number> ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(indexes):
                return indexes[idx - 1]
        safe_print("invalid selection, try again.")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Recall interface: API + interactive terminal")
    parser.add_argument("--mode", default="interactive", choices=["interactive", "api", "once"], type=str)
    parser.add_argument("--index_root", default=str(base), type=str)
    parser.add_argument("--index_dir", default=str(base / "index_local_bge_m3"), type=str)
    parser.add_argument("--shaoxing_index_dir", default=str(base / "index_shaoxing_bge_m3"), type=str)
    parser.add_argument("--ann", default="hnsw", choices=["hnsw", "flat"], type=str)
    parser.add_argument("--ef_search", default=64, type=int)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--top_n", default=3, type=int)
    parser.add_argument("--query", default="", type=str)
    parser.add_argument("--source", default="shanghai", type=str)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8088, type=int)
    return parser.parse_args()


def resolve_index_dir(args: argparse.Namespace) -> Path:
    if args.index_dir:
        path = Path(args.index_dir).resolve()
        if not (path / "meta.json").exists():
            raise FileNotFoundError(f"invalid index_dir: {path}")
        return path
    root = Path(args.index_root).resolve()
    indexes = discover_indexes(root)
    if not indexes:
        raise FileNotFoundError(f"no index directory found under: {root}")
    if args.mode in {"interactive", "api"}:
        return choose_index_interactive(indexes)
    return indexes[0]


def load_engines(args: argparse.Namespace) -> dict[str, RecallEngine]:
    engines: dict[str, RecallEngine] = {}
    source_to_path = {
        "shanghai": Path(args.index_dir).resolve(),
        "shaoxing": Path(args.shaoxing_index_dir).resolve(),
    }
    for source, index_dir in source_to_path.items():
        if not (index_dir / "meta.json").exists():
            safe_print(f"skip {source}: missing meta.json under {index_dir}")
            continue
        safe_print(f"loading {source} engine from: {index_dir}")
        engine = RecallEngine(index_dir=index_dir, ann=args.ann, ef_search=args.ef_search)
        engines[source] = engine
        safe_print(
            f"loaded {source}. device={engine.device}, ann={engine.ann}, "
            f"model={engine.meta.get('model_name_or_path', '')}, load_s={engine.load_s:.4f}"
        )
    if not engines:
        raise FileNotFoundError("no usable recall index found for shanghai or shaoxing")
    return engines


def run_interactive(engine: RecallEngine, top_k: int, top_n: int) -> None:
    safe_print("interactive started, input query; exit/quit to stop")
    while True:
        query = input("query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        result = engine.search(query, top_k=top_k, top_n=top_n)
        print_result(result)


def run_api(engines: dict[str, RecallEngine], host: str, port: int, top_k: int, top_n: int) -> None:
    handler = build_api_handler(engines, top_k=top_k, top_n=top_n)
    server = ThreadingHTTPServer((host, port), handler)
    safe_print(f"api started on http://{host}:{port}")
    safe_print('POST /recall with JSON: {"query":"你好","source":"shanghai","top_k":20,"top_n":3}')
    safe_print("GET  /health")
    server.serve_forever()


def main() -> None:
    args = parse_args()
    source = normalize_source_name(args.source)

    if args.mode == "once":
        index_dir = resolve_index_dir(args)
        safe_print(f"loading engine from: {index_dir}")
        engine = RecallEngine(index_dir=index_dir, ann=args.ann, ef_search=args.ef_search)
        safe_print(
            f"loaded. device={engine.device}, ann={engine.ann}, "
            f"model={engine.meta.get('model_name_or_path', '')}, load_s={engine.load_s:.4f}"
        )
        query = args.query.strip()
        if not query:
            raise ValueError("--mode once requires --query")
        result = engine.search(query, top_k=args.top_k, top_n=args.top_n)
        safe_print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.mode == "api":
        engines = load_engines(args)
        run_api(engines, host=args.host, port=args.port, top_k=args.top_k, top_n=args.top_n)
        return
    index_dir = resolve_index_dir(args)
    if source == "shaoxing" and Path(args.shaoxing_index_dir).resolve().exists():
        index_dir = Path(args.shaoxing_index_dir).resolve()
    safe_print(f"loading engine from: {index_dir}")
    engine = RecallEngine(index_dir=index_dir, ann=args.ann, ef_search=args.ef_search)
    safe_print(
        f"loaded. device={engine.device}, ann={engine.ann}, "
        f"model={engine.meta.get('model_name_or_path', '')}, load_s={engine.load_s:.4f}"
    )
    run_interactive(engine, top_k=args.top_k, top_n=args.top_n)


if __name__ == "__main__":
    main()
