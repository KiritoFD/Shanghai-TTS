from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from shared.config import APP_ROOT, REPO_ROOT


GENERATED_DIR = APP_ROOT / "data" / "generated"
PRIMARY_INDEX_DIR = APP_ROOT / "recall" / "index_local_bge_m3"
SHAOXING_XLSX_PATH = APP_ROOT / "merged_result.xlsx"
LEGACY_SHAOXING_XLSX_PATH = REPO_ROOT / "merged_result.xlsx"
SHAOXING_CSV_PATH = GENERATED_DIR / "shaoxing_processed.csv"
SHAOXING_INDEX_DIR = APP_ROOT / "recall" / "index_shaoxing_bge_m3"

PRIMARY_SOURCE_ID = "shanghai_csv"
SHAOXING_SOURCE_ID = "shaoxing_xlsx"


@dataclass
class SourceStatus:
    source_id: str
    raw_path: Path
    processed_path: Path | None
    index_dir: Path
    available: bool
    message: str


def ensure_generated_dir() -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    return GENERATED_DIR


def index_complete(index_dir: Path) -> bool:
    required = ["meta.json", "records.jsonl", "embeddings.npy"]
    return all((index_dir / name).exists() for name in required)


def _pick_first_text(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value and value.lower() != "nan":
            return value
    return ""


def _is_heading_row(row: dict[str, Any], headword: str, definition: str, pinyin: str) -> bool:
    tag_blob = " ".join(
        str(row.get(key, "") or "").strip().lower()
        for key in ("标记", "revised_标记", "状态")
    )
    if "heading" in tag_blob or "non_entry" in tag_blob:
        return True
    if not pinyin and not definition:
        return True
    if headword.startswith(("名词（", "动词（", "形容词（", "副词（", "代词（", "介词（", "连词（", "助词（")):
        return True
    if headword[:2].isdigit() and "、" in headword:
        return True
    return False


def build_shaoxing_processed_csv(
    xlsx_path: Path = SHAOXING_XLSX_PATH,
    out_csv: Path = SHAOXING_CSV_PATH,
) -> Path:
    if not xlsx_path.exists() and xlsx_path == SHAOXING_XLSX_PATH and LEGACY_SHAOXING_XLSX_PATH.exists():
        xlsx_path = LEGACY_SHAOXING_XLSX_PATH
    if not xlsx_path.exists():
        raise FileNotFoundError(f"shaoxing xlsx not found: {xlsx_path}")

    ensure_generated_dir()
    frame = pd.read_excel(xlsx_path, dtype=str, keep_default_na=False)
    rows: list[dict[str, str]] = []

    for raw in frame.to_dict(orient="records"):
        headword = _pick_first_text(raw, ["汉字"])
        definition = _pick_first_text(raw, ["revised_后续汉字.1", "revised_后续汉字", "后续汉字"])
        pinyin = _pick_first_text(raw, ["revised_拼音", "拼音"])
        ipa = _pick_first_text(raw, ["revised_IPA识别", "IPA识别"])
        alt = _pick_first_text(raw, ["其他读法", "revised_其他读法"])
        if not headword or _is_heading_row(raw, headword, definition, pinyin):
            continue

        rows.append(
            {
                "entry": headword,
                "entry_alt": headword,
                "romanization": pinyin,
                "ipa": ipa,
                "definition": definition,
                "tone_notation": pinyin,
                "notes": f"source={SHAOXING_SOURCE_ID};alt={alt};status={str(raw.get('状态', '')).strip()}",
            }
        )

    if not rows:
        raise RuntimeError(f"no usable shaoxing rows found in: {xlsx_path}")

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    logging.info("[dict] built shaoxing processed csv: %s (%s rows)", out_csv, len(rows))
    return out_csv


def ensure_shaoxing_processed_csv(
    xlsx_path: Path = SHAOXING_XLSX_PATH,
    out_csv: Path = SHAOXING_CSV_PATH,
) -> Path | None:
    if not xlsx_path.exists():
        return None
    if out_csv.exists() and out_csv.stat().st_mtime >= xlsx_path.stat().st_mtime:
        return out_csv
    return build_shaoxing_processed_csv(xlsx_path=xlsx_path, out_csv=out_csv)


def ensure_dense_index(
    dict_csv: Path,
    index_dir: Path,
    *,
    source_label: str,
    model_name_or_path: str = "BAAI/bge-m3",
) -> SourceStatus:
    if not dict_csv.exists():
        return SourceStatus(
            source_id=source_label,
            raw_path=dict_csv,
            processed_path=None,
            index_dir=index_dir,
            available=False,
            message="raw_dictionary_missing",
        )
    if index_complete(index_dir):
        return SourceStatus(
            source_id=source_label,
            raw_path=dict_csv,
            processed_path=dict_csv,
            index_dir=index_dir,
            available=True,
            message="index_ready",
        )

    try:
        from recall.model_utils import ensure_model_path
        from recall.scripts.build_vector_index import run_build
    except ImportError:
        from app.recall.model_utils import ensure_model_path  # type: ignore[no-redef]
        from app.recall.scripts.build_vector_index import run_build  # type: ignore[no-redef]

    index_dir.mkdir(parents=True, exist_ok=True)
    model_path = ensure_model_path(model_name_or_path, APP_ROOT / "recall")
    run_build(
        dict_csv=str(dict_csv),
        out_dir=str(index_dir),
        model_name_or_path=model_path,
        id_col="0",
        sh_col="1",
        def_col="4",
        header="infer",
        batch_size=64,
        max_length=128,
        text_mode="headword_definition",
        progress_cb=lambda msg: logging.info("[%s-index] %s", source_label, msg),
    )
    return SourceStatus(
        source_id=source_label,
        raw_path=dict_csv,
        processed_path=dict_csv,
        index_dir=index_dir,
        available=index_complete(index_dir),
        message="index_built",
    )
