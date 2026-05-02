from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dense vector index for Shanghai dictionary recall")
    parser.add_argument("--dict_csv", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--model_name_or_path", default="BAAI/bge-small-zh-v1.5", type=str)
    parser.add_argument("--id_col", default="0", type=str)
    parser.add_argument("--sh_col", default="1", type=str)
    parser.add_argument("--def_col", default="4", type=str)
    parser.add_argument("--header", default="infer", choices=["infer", "none"], type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument(
        "--text_mode",
        default="headword_definition",
        choices=["headword", "definition", "headword_definition"],
        type=str,
        help="How to compose index text for embedding.",
    )
    return parser.parse_args()


def pick_col(df: pd.DataFrame, key: str):
    if key in df.columns:
        return df[key]
    if key.isdigit():
        idx = int(key)
        if 0 <= idx < len(df.columns):
            return df.iloc[:, idx]
    raise KeyError(f"column not found: {key}")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    all_vecs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            out = model(**encoded)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output
            else:
                vec = mean_pool(out.last_hidden_state, encoded["attention_mask"])
            all_vecs.append(vec.cpu().numpy().astype(np.float32))
    return np.concatenate(all_vecs, axis=0)


def build_index_text(shanghai: str, definition: str, text_mode: str) -> str:
    shanghai = shanghai.strip()
    definition = definition.strip()
    if text_mode == "headword":
        return shanghai
    if text_mode == "definition":
        return definition
    return f"{shanghai}。释义：{definition}"


def make_stored_model_path(model_name_or_path: str, out_dir: Path) -> str:
    model_path = Path(model_name_or_path)
    if not model_path.exists():
        return model_name_or_path
    try:
        return os.path.relpath(model_path.resolve(), out_dir.resolve()).replace("\\", "/")
    except Exception:
        return str(model_path)


def run_build(
    dict_csv: str,
    out_dir: str,
    model_name_or_path: str = "BAAI/bge-small-zh-v1.5",
    id_col: str = "0",
    sh_col: str = "1",
    def_col: str = "4",
    header: str = "infer",
    batch_size: int = 128,
    max_length: int = 128,
    text_mode: str = "headword_definition",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """构建向量索引的核心逻辑，可被外部 import 调用。

    Parameters
    ----------
    dict_csv:           词典 CSV 路径
    out_dir:            输出目录
    model_name_or_path: HuggingFace 模型名或本地路径
    id_col:             ID 列名或列序号（字符串）
    sh_col:             上海话字段列名或列序号（优先使用 entry_alt，为空时 fallback 到 entry/id_col）
    def_col:            释义列名或列序号
    header:             "infer" 或 "none"
    batch_size:         编码批大小
    max_length:         tokenizer 截断长度
    text_mode:          "headword" | "definition" | "headword_definition"
    progress_cb:        可选进度回调，接受单个 str 消息

    Returns
    -------
    dict  meta.json 对应的字典
    """

    def _cb(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    # ── 1. 读 CSV ────────────────────────────────────────────────────────────
    _header = None if header == "none" else "infer"
    raw = pd.read_csv(dict_csv, header=_header)
    _cb(f"读取 CSV: {len(raw)} 行")

    # ── 2. 组装 DataFrame，处理 entry_alt fallback ───────────────────────────
    sh_series = pick_col(raw, sh_col).fillna("").astype(str).str.strip()

    # 如果 sh_col 对应列全为空，fallback 到 id_col（col0）
    if (sh_series == "").all() or (sh_series == "nan").all():
        _cb(f"sh_col='{sh_col}' 全为空，fallback 到 id_col='{id_col}'")
        sh_series = pick_col(raw, id_col).fillna("").astype(str).str.strip()
    else:
        # 逐行处理：entry_alt（sh_col）为空时，用 entry（id_col）填充
        id_series = pick_col(raw, id_col).fillna("").astype(str).str.strip()
        sh_series = sh_series.where(
            (sh_series != "") & (sh_series != "nan"),
            other=id_series,
        )

    df = pd.DataFrame(
        {
            "id": pick_col(raw, id_col),
            "shanghai": sh_series,
            "definition": pick_col(raw, def_col),
        }
    )

    # ── 3. 整理 ID 列 ────────────────────────────────────────────────────────
    numeric_id = pd.to_numeric(df["id"], errors="coerce")
    if numeric_id.notna().sum() < max(100, int(len(df) * 0.1)):
        df["id"] = list(range(len(df)))
    else:
        df["id"] = numeric_id.fillna(-1).astype(int)
        df = df[df["id"] >= 0].copy()

    # ── 4. 清洗文本 ──────────────────────────────────────────────────────────
    df["shanghai"] = df["shanghai"].fillna("").astype(str).str.strip()
    df["definition"] = df["definition"].fillna("").astype(str).str.strip()
    df = df[
        (df["shanghai"] != "") & (df["definition"] != "")
        & (df["shanghai"] != "nan") & (df["definition"] != "nan")
    ]
    df = df.reset_index(drop=True)
    _cb(f"有效词条: {len(df)} 行")

    # ── 5. 构造待编码文本 ────────────────────────────────────────────────────
    df["index_text"] = [
        build_index_text(sh, defn, text_mode)
        for sh, defn in zip(df["shanghai"].tolist(), df["definition"].tolist())
    ]
    texts = df["index_text"].tolist()

    # ── 6. 加载模型并编码 ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _cb(f"加载模型: {model_name_or_path}  device={device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)

    # 带进度回调的编码封装
    all_vecs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            out = model(**encoded)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vec = out.pooler_output
            else:
                vec = mean_pool(out.last_hidden_state, encoded["attention_mask"])
            all_vecs.append(vec.cpu().numpy().astype(np.float32))
            _cb(f"编码中 {min(i + batch_size, len(texts))}/{len(texts)}...")

    vecs = l2_normalize(np.concatenate(all_vecs, axis=0)).astype(np.float32)

    # ── 7. 保存文件 ──────────────────────────────────────────────────────────
    _cb("保存索引...")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "embeddings.npy", vecs)

    with (out_path / "records.jsonl").open("w", encoding="utf-8") as f:
        for row in df.to_dict("records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "model_name_or_path": make_stored_model_path(model_name_or_path, out_path),
        "rows": int(len(df)),
        "dim": int(vecs.shape[1]),
        "dict_csv": str(dict_csv),
        "text_mode": text_mode,
        "text_fields": ["shanghai", "definition"] if text_mode == "headword_definition" else [text_mode],
    }
    (out_path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _cb(f"索引已保存: {len(df)} 向量，dim={vecs.shape[1]}")

    # ── 8. 可选：构建 HNSW 索引 ──────────────────────────────────────────────
    try:
        import hnswlib  # type: ignore
        hnsw_path = out_path / "index_hnsw.bin"
        dim = vecs.shape[1]
        idx = hnswlib.Index(space="cosine", dim=dim)
        idx.init_index(max_elements=len(vecs) + 1000, ef_construction=200, M=16)
        idx.add_items(vecs, list(range(len(vecs))))
        idx.save_index(str(hnsw_path))
        _cb(f"HNSW 索引已保存: {len(vecs)} 向量")
    except ImportError:
        _cb("hnswlib 不可用，跳过 HNSW 构建")

    return meta


def main() -> None:
    args = parse_args()
    meta = run_build(
        dict_csv=args.dict_csv,
        out_dir=args.out_dir,
        model_name_or_path=args.model_name_or_path,
        id_col=args.id_col,
        sh_col=args.sh_col,
        def_col=args.def_col,
        header=args.header,
        batch_size=args.batch_size,
        max_length=args.max_length,
        text_mode=args.text_mode,
        progress_cb=print,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
