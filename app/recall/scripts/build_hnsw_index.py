from __future__ import annotations

import argparse
import json
from pathlib import Path

import hnswlib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HNSW index from dense embeddings.npy")
    parser.add_argument("--index_dir", required=True, type=str, help="Directory containing embeddings.npy and meta.json")
    parser.add_argument("--m", default=32, type=int)
    parser.add_argument("--ef_construction", default=200, type=int)
    parser.add_argument("--ef_search", default=64, type=int)
    return parser.parse_args()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)
    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "meta.json"
    out_index = index_dir / "index_hnsw.bin"
    out_meta = index_dir / "hnsw_meta.json"

    vectors = np.load(emb_path).astype(np.float32)
    vectors = l2_normalize(vectors)
    n, dim = vectors.shape

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=args.ef_construction, M=args.m)
    labels = np.arange(n, dtype=np.int32)
    index.add_items(vectors, labels)
    index.set_ef(args.ef_search)
    index.save_index(str(out_index))

    base_meta = {}
    if meta_path.exists():
        base_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hnsw_meta = {
        "space": "cosine",
        "rows": int(n),
        "dim": int(dim),
        "m": int(args.m),
        "ef_construction": int(args.ef_construction),
        "ef_search": int(args.ef_search),
        "index_file": out_index.name,
        "base_meta": base_meta,
    }
    out_meta.write_text(json.dumps(hnsw_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(hnsw_meta, ensure_ascii=False, indent=2))
    print(f"saved={out_index}")


if __name__ == "__main__":
    main()
