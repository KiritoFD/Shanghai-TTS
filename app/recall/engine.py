from __future__ import annotations

import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:
    import hnswlib
except ImportError:
    hnswlib = None

try:
    from .model_utils import ensure_model_path
except ImportError:
    from model_utils import ensure_model_path


LOW_INFORMATION_TOKENS = {
    "我",
    "你",
    "他",
    "她",
    "它",
    "侬",
    "的",
    "了",
    "啊",
    "呀",
    "呢",
    "吗",
    "吧",
    "很",
    "太",
    "最",
    "更",
    "一下",
    "一",
    "下",
    "怎么",
    "说",
    "讲",
}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_query(
    query: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int = 128,
) -> np.ndarray:
    with torch.no_grad():
        encoded = tokenizer(
            [query],
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
    arr = vec.cpu().numpy().astype(np.float32)
    arr = l2_normalize(arr)
    return arr[0]


def encode_queries_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """Batch-encode multiple strings into L2-normalized vectors.

    Returns np.ndarray of shape (len(texts), dim).
    """
    all_vecs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
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
    if not all_vecs:
        return np.zeros((0, 1), dtype=np.float32)
    return l2_normalize(np.concatenate(all_vecs, axis=0))


def load_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def load_csv_notes(csv_path: Path) -> dict[str, str]:
    """Load the 'notes' column (col index 6) from processed_results.csv.

    Returns a mapping from normalised entry text -> notes string.
    Keys are stripped of 【】 brackets so they match both bracketed and
    plain headwords stored in records.
    """
    notes: dict[str, str] = {}
    if not csv_path.exists():
        return notes
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            note_text = row[6].strip()
            if not note_text:
                continue
            # Index by col-0 entry (strip 【】) and also col-4 definition/display entry
            for col in (0, 4):
                key = row[col].strip().strip("\ufeff").strip("【】[] ").strip()
                if key and key not in ("entry", "entry_alt", "definition"):  # skip header row
                    notes[key] = note_text
    return notes


def normalize_query(query: str) -> str:
    text = query.strip()
    # Strip "上海话：" or "上海话:" prefix (colon variant)
    text = re.sub(r"^上海话[：:]\s*", "", text)
    patterns = [
        "用上海话怎么说",
        "上海话怎么说",
        "上海话怎么讲",
        "上海话里",
        "口语里",
        "怎么说",
        "怎么讲",
        "怎么表达",
        "是什么意思",
        "啥意思",
        "这个词",
        "这个",
        "请问",
        "麻烦问下",
        "麻烦",
        "想问下",
        "想问",
        "请看",
        "到底",
        "功能",
        "说法",
        "表达",
    ]
    for pattern in patterns:
        text = text.replace(pattern, "")
    text = re.sub(r"[\"'，,。.!！?？；;：:（）()\[\]【】]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or query.strip()


def build_query_variants(query: str) -> list[str]:
    base = normalize_query(query)
    variants: list[str] = [base]

    trimmed = base
    for prefix in ("我", "请", "帮我", "麻烦", "想问下", "想问"):
        if trimmed.startswith(prefix) and len(trimmed) > len(prefix) + 1:
            trimmed = trimmed[len(prefix):]
            break
    for suffix in ("了", "啊", "呀", "呢", "嘛", "吧", "吗"):
        if trimmed.endswith(suffix) and len(trimmed) > 2:
            trimmed = trimmed[:-1]
            break
    trimmed = trimmed.strip()
    if trimmed and trimmed != base and is_meaningful_variant(trimmed):
        variants.append(trimmed)

    # Aggressive noise removal: strip pronouns, particles, filler words
    removable_tokens = (
        "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
        "把", "给", "跟", "向", "对", "用", "到底", "一下", "讲", "说",
        "这个", "那个", "这些", "那些", "是不是", "这样", "那样",
        "应该", "可以", "能不能", "要不要", "想", "要",
    )
    generic = base
    for token in removable_tokens:
        generic = generic.replace(token, "")
    generic = re.sub(r"\s+", " ", generic).strip()
    if generic and generic != base and is_meaningful_variant(generic):
        variants.append(generic)

    # For short spoken phrases like "我爱你", also keep the core action chunk.
    suffix_pronouns = ("你", "他", "她", "它")
    for pronoun in suffix_pronouns:
        if base.endswith(pronoun) and len(base) > len(pronoun):
            core = base[:-len(pronoun)].strip()
            core = re.sub(r"\s+", " ", core)
            for token in removable_tokens:
                core = core.replace(token, "")
            core = core.strip()
            if core and core != base and is_meaningful_variant(core, allow_single_char=True):
                variants.append(core)
            break

    # Extract core concept: for multi-concept queries, try the longest segment
    cjk_segments = re.findall(r"[\u4e00-\u9fff]{2,}", base)
    if len(cjk_segments) >= 3:
        # For queries with 3+ segments, add the longest segment as a focused variant
        longest = max(cjk_segments, key=len)
        if longest != base.strip() and is_meaningful_variant(longest):
            variants.append(longest)

    # Use verb_negation_expansion.extract_core_concept for pattern-based extraction
    try:
        from verb_negation_expansion import extract_core_concept
        for concept in extract_core_concept(base):
            if concept and concept != base and is_meaningful_variant(concept):
                variants.append(concept)
    except ImportError:
        pass

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        variant = variant.strip()
        if not variant or variant in seen:
            continue
        seen.add(variant)
        deduped.append(variant)
    return deduped


def merge_query_variants(query: str, extra_variants: list[str] | None = None) -> list[str]:
    merged = build_query_variants(query)
    if extra_variants:
        merged.extend(str(v).strip() for v in extra_variants if str(v).strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in merged:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        deduped.append(variant)
    return deduped


def lexical_retrieve(query: str, records: list[dict], top_n: int) -> list[dict]:
    q = query.strip()
    if not q:
        return []
    normalized_q = re.sub(r"[。！？!?，,；;：:、\s]+", "", q)
    allow_contains = len(normalized_q) >= 2
    selected: list[dict] = []
    for row in records:
        shanghai = str(row.get("shanghai", "")).strip()
        definition = str(row.get("definition", "")).strip()
        headword = shanghai.strip("【】[] ")
        normalized_definition = definition.replace("。", "").replace("！", "").replace("？", "").strip()
        score = 0.0
        if q == headword:
            score += 20.0
        elif allow_contains and q in shanghai:
            score += 10.0
        if q == normalized_definition:
            score += 18.0
        elif allow_contains and definition.startswith(q):
            score += 8.0
        elif allow_contains and q in definition:
            score += 3.0
        if not score:
            continue
        score -= 0.01 * len(shanghai)
        item = dict(row)
        item["score"] = float(score)
        item["_source"] = "lexical"
        selected.append(item)
    selected.sort(key=lambda x: x["score"], reverse=True)
    return selected[:top_n]


def lexical_retrieve_multi(queries: list[str], records: list[dict], top_n: int) -> list[dict]:
    best_by_id: dict[int, dict] = {}
    for q in queries:
        for row in lexical_retrieve(q, records, top_n=top_n):
            rid = int(row["id"])
            if rid not in best_by_id or float(row["score"]) > float(best_by_id[rid]["score"]):
                best_by_id[rid] = row
    merged = list(best_by_id.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_n]


def discover_indexes(index_root: Path) -> list[Path]:
    candidates = []
    for p in index_root.iterdir():
        if p.is_dir() and (p / "meta.json").exists() and (p / "records.jsonl").exists():
            candidates.append(p)
    candidates.sort(key=lambda x: x.name)
    return candidates


def tokenize_sparse_text(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())
    if not cleaned:
        return []

    tokens: list[str] = []
    current_ascii: list[str] = []
    cjk_chars: list[str] = []
    for ch in cleaned:
        if re.match(r"[a-z0-9_]", ch):
            current_ascii.append(ch)
            if cjk_chars:
                chars = cjk_chars
                tokens.extend(chars)
                if len(chars) >= 2:
                    tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
                cjk_chars = []
        else:
            cjk_chars.append(ch)
            if current_ascii:
                tokens.append("".join(current_ascii))
                current_ascii = []

    if current_ascii:
        tokens.append("".join(current_ascii))
    if cjk_chars:
        tokens.extend(cjk_chars)
        if len(cjk_chars) >= 2:
            tokens.extend("".join(cjk_chars[i : i + 2]) for i in range(len(cjk_chars) - 1))

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def significant_sparse_tokens(text: str) -> list[str]:
    tokens = tokenize_sparse_text(text)
    return [token for token in tokens if token not in LOW_INFORMATION_TOKENS]


def is_meaningful_variant(text: str, allow_single_char: bool = False) -> bool:
    normalized = re.sub(r"\s+", "", text.strip())
    if not normalized:
        return False
    if len(normalized) == 1 and not allow_single_char:
        return False
    return bool(significant_sparse_tokens(normalized))


class SparseBM25:
    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: list[int] = []
        self.df: dict[str, int] = {}
        total_len = 0
        # Inverted posting list: token -> [(doc_id, tf)]
        self.postings: dict[str, list[tuple[int, int]]] = {}

        for doc_id, row in enumerate(records):
            text = str(row.get("index_text") or f"{row.get('shanghai', '')}。释义：{row.get('definition', '')}")
            tokens = tokenize_sparse_text(text)
            length = len(tokens)
            self.doc_len.append(length)
            total_len += length

            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token, freq in tf.items():
                self.df[token] = self.df.get(token, 0) + 1
                if token not in self.postings:
                    self.postings[token] = []
                self.postings[token].append((doc_id, freq))

        self.avgdl = total_len / max(len(records), 1)
        self.doc_count = len(records)

    def score_query(self, query: str) -> np.ndarray:
        q_tokens = significant_sparse_tokens(query)
        scores = np.zeros(self.doc_count, dtype=np.float32)
        if not q_tokens:
            return scores

        for token in q_tokens:
            posting = self.postings.get(token)
            if not posting:
                continue
            df = self.df[token]
            idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            for doc_id, freq in posting:
                dl = self.doc_len[doc_id]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[doc_id] += idf * (freq * (self.k1 + 1.0)) / denom
        return scores


def sparse_retrieve_multi(queries: list[str], bm25: SparseBM25, records: list[dict], top_n: int) -> list[dict]:
    best_by_id: dict[int, dict] = {}
    for query in queries:
        scores = bm25.score_query(query)
        if scores.size == 0:
            continue
        k = min(top_n, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k] if k > 0 else np.array([], dtype=np.int64)
        idx = idx[np.argsort(-scores[idx])] if k > 0 else idx
        for i in idx:
            score = float(scores[int(i)])
            if score <= 0:
                continue
            row = dict(records[int(i)])
            row["score"] = score
            row["_source"] = "sparse"
            rid = int(row["id"])
            if rid not in best_by_id or score > float(best_by_id[rid]["score"]):
                best_by_id[rid] = row
    merged = list(best_by_id.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_n]


class RecallEngine:
    def __init__(self, index_dir: Path, ann: str = "hnsw", ef_search: int = 64):
        t0 = time.perf_counter()
        self.index_dir = index_dir
        self.ann = ann
        self.ef_search = ef_search
        self.meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        self.records = load_records(index_dir / "records.jsonl")
        self.record_pos_by_id = {int(row["id"]): idx for idx, row in enumerate(self.records)}
        self.bm25 = SparseBM25(self.records)
        self.matrix = l2_normalize(np.load(index_dir / "embeddings.npy").astype(np.float32))
        self.hnsw_index = None

        if ann == "hnsw" and hnswlib is not None and (index_dir / "index_hnsw.bin").exists():
            dim = int(self.meta["dim"])
            self.hnsw_index = hnswlib.Index(space="cosine", dim=dim)
            self.hnsw_index.load_index(str(index_dir / "index_hnsw.bin"))
            self.hnsw_index.set_ef(self.ef_search)

        # Lexical index: pre-compute normalized headwords and definitions
        self._hw_exact: dict[str, int] = {}      # normalized headword -> record id
        self._def_exact: dict[str, list[int]] = {}  # normalized definition -> [record ids]
        for idx, row in enumerate(self.records):
            hw = str(row.get("shanghai", "")).strip().strip("【】[] ").strip()
            if hw:
                self._hw_exact[hw] = int(row["id"])
            defn = re.sub(r"[。！？!?，,；;：:、\s]+", "", str(row.get("definition", "")).strip())
            if defn:
                self._def_exact.setdefault(defn, []).append(int(row["id"]))

        # Load CSV notes (col-6) for entries that have supplementary descriptions
        base = Path(__file__).resolve().parent
        # Try sibling processed_results.csv first, then project root
        for _csv_candidate in [
            index_dir.parent / "processed_results.csv",
            base / "processed_results.csv",
            base.parent.parent / "processed_results.csv",
        ]:
            if _csv_candidate.exists():
                self._csv_notes: dict[str, str] = load_csv_notes(_csv_candidate)
                break
        else:
            self._csv_notes = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = ensure_model_path(str(self.meta["model_name_or_path"]), base)
        self.meta["model_name_or_path"] = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device).eval()
        self.load_s = round(time.perf_counter() - t0, 4)

    def _row_dense_score(self, row_id: int, qv: np.ndarray) -> float:
        pos = self.record_pos_by_id.get(int(row_id))
        if pos is None:
            return 0.0
        return float(np.dot(self.matrix[pos], qv))

    def _lexical_retrieve_fast(self, variants: list[str], top_n: int) -> list[dict]:
        """Fast lexical matching using pre-computed headword/definition indexes."""
        best_by_id: dict[int, dict] = {}
        for q in variants:
            q = q.strip()
            if not q:
                continue
            normalized_q = re.sub(r"[。！？!?，,；;：:、\s]+", "", q)
            # Exact headword match
            rid = self._hw_exact.get(q)
            if rid is not None:
                pos = self.record_pos_by_id.get(rid)
                if pos is not None and rid not in best_by_id:
                    item = dict(self.records[pos])
                    item["score"] = 20.0
                    item["_source"] = "lexical"
                    best_by_id[rid] = item
            # Exact definition match
            if normalized_q:
                for rid in self._def_exact.get(normalized_q, []):
                    if rid not in best_by_id:
                        pos = self.record_pos_by_id.get(rid)
                        if pos is not None:
                            item = dict(self.records[pos])
                            item["score"] = 18.0
                            item["_source"] = "lexical"
                            best_by_id[rid] = item
        merged = sorted(best_by_id.values(), key=lambda x: x["score"], reverse=True)
        return merged[:top_n]

    def _field_match_score(self, variants: list[str], row: dict[str, Any]) -> float:
        headword = str(row.get("shanghai", "")).strip()
        clean_headword = headword.strip("【】[] ").strip()
        definition = str(row.get("definition", "")).strip()
        normalized_definition = re.sub(r"[。！？!?，,；;：:、\s]+", "", definition)
        score = 0.0

        for idx, variant in enumerate(variants):
            q = variant.strip()
            if not q:
                continue
            normalized_q = re.sub(r"[。！？!?，,；;：:、\s]+", "", q)
            is_primary = idx == 0
            primary_weight = 1.5 if is_primary else 1.0
            allow_contains = len(normalized_q) >= 2

            if q == clean_headword:
                score += 6.0 * primary_weight
            elif allow_contains and clean_headword.startswith(q):
                score += 3.0 * primary_weight
            elif allow_contains and q in clean_headword:
                # Reduce substring match to avoid false positives like "颜色" in "看颜色"
                score += 1.0 * primary_weight

            if normalized_q and normalized_q == normalized_definition:
                score += 5.0 * primary_weight
            elif allow_contains and definition.startswith(q):
                score += 2.5 * primary_weight
            elif allow_contains and q in definition:
                score += 0.8 * primary_weight

        primary_q = variants[0] if variants else ""
        primary_tokens = set(significant_sparse_tokens(primary_q))

        if primary_tokens:
            headword_tokens = set(significant_sparse_tokens(clean_headword))
            definition_tokens = set(significant_sparse_tokens(definition))
            head_overlap = len(primary_tokens & headword_tokens) / max(len(primary_tokens), 1)
            definition_overlap = len(primary_tokens & definition_tokens) / max(len(primary_tokens), 1)
            score += 3.0 * head_overlap + 2.0 * definition_overlap

        exact_variant_match = any(v.strip() == clean_headword for v in variants)
        has_longer_variant = any(len(re.sub(r"\s+", "", v.strip())) >= 2 for v in variants)
        if len(clean_headword) <= 1 and has_longer_variant and not exact_variant_match:
            score -= 5.0

        return score

    def _rerank_results(
        self,
        fused_rows: list[dict[str, Any]],
        variants: list[str],
        qv: np.ndarray,
        top_n: int,
    ) -> list[dict[str, Any]]:
        if not fused_rows:
            return []

        fused_scores = [float(row["score"]) for row in fused_rows]
        min_fused = min(fused_scores)
        max_fused = max(fused_scores)
        fused_span = max(max_fused - min_fused, 1e-9)

        reranked: list[dict[str, Any]] = []
        for row in fused_rows:
            row_id = int(row["id"])
            dense_score = self._row_dense_score(row_id, qv)
            field_score = self._field_match_score(variants, row)
            fused_score = (float(row["score"]) - min_fused) / fused_span
            final_score = 0.80 * dense_score + 0.40 * field_score + 0.40 * fused_score
            if field_score < 0.25:
                final_score *= 0.60

            item = dict(row)
            item["score"] = float(final_score)
            reranked.append(item)

        reranked.sort(key=lambda x: float(x["score"]), reverse=True)
        return reranked[:top_n]

    def search(
        self,
        query: str,
        top_k: int = 20,
        top_n: int = 3,
        extra_variants: list[str] | None = None,
        precomputed_vecs: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        variants = merge_query_variants(query, extra_variants=extra_variants)
        normalized_query = variants[0]

        # Dialect expansion for lexical/sparse retrieval only
        try:
            from dialect_mappings import expand_query_variants
            dialect_variants = []
            for v in variants[:2]:  # expand top-2 variants only
                dialect_variants.extend(expand_query_variants(v))
            # Add dialect variants for lexical/sparse (not for dense encoding)
            all_lexical_variants = list(dict.fromkeys(variants + dialect_variants))
        except ImportError:
            all_lexical_variants = variants

        # Verb negation expansion (不→勿)
        try:
            from verb_negation_expansion import expand_verb_negation
            negation_variants = []
            for v in variants[:2]:
                negation_variants.extend(expand_verb_negation(v))
            all_lexical_variants = list(dict.fromkeys(all_lexical_variants + negation_variants))
        except ImportError:
            pass

        if precomputed_vecs is not None:
            # Use pre-computed vectors (batch mode)
            all_vecs = np.stack([precomputed_vecs[v] for v in variants], axis=0)
        else:
            # Batch-encode all variants in one forward pass
            all_vecs = encode_queries_batch(variants, self.tokenizer, self.model, self.device)

        if len(variants) > 1:
            weights = np.array([0.6] + [0.4 / max(len(variants) - 1, 1)] * (len(variants) - 1), dtype=np.float32)
            qv = (all_vecs * weights[:, None]).sum(axis=0).astype(np.float32)
        else:
            qv = all_vecs[0]
        qv = qv / (np.linalg.norm(qv) + 1e-12)

        lexical_hits = self._lexical_retrieve_fast(all_lexical_variants, top_n=max(top_n * 8, 30))
        sparse_hits = sparse_retrieve_multi(all_lexical_variants, self.bm25, self.records, top_n=max(top_n * 8, 30))
        # Use wider ANN window (50) to capture more candidates for reranking
        ann_k = max(top_k, 50)
        if self.hnsw_index is not None:
            labels, distances = self.hnsw_index.knn_query(qv.reshape(1, -1), k=ann_k)
            labels = labels[0]
            distances = distances[0]
            ann_selected = []
            for i, d in zip(labels[:ann_k], distances[:ann_k]):
                row = dict(self.records[int(i)])
                row["score"] = float(1.0 - float(d))
                row["_source"] = "ann"
                ann_selected.append(row)
        else:
            assert self.matrix is not None
            scores = self.matrix @ qv
            idx = np.argpartition(-scores, min(ann_k, len(scores) - 1))[:ann_k]
            idx = idx[np.argsort(-scores[idx])]
            ann_selected = []
            for i in idx[:ann_k]:
                row = dict(self.records[int(i)])
                row["score"] = float(scores[int(i)])
                row["_source"] = "ann"
                ann_selected.append(row)

        short_variants = [v for v in variants if 2 <= len(re.sub(r"\s+", "", v)) <= 4]
        if short_variants and self.hnsw_index is not None:
            sv_batch = short_variants[:3]
            if precomputed_vecs is not None:
                sv_batch = [v for v in sv_batch if v in precomputed_vecs]
                if sv_batch:
                    short_vecs = np.stack([precomputed_vecs[v] for v in sv_batch], axis=0)
                else:
                    short_vecs = np.zeros((0, self.matrix.shape[1]), dtype=np.float32)
            else:
                short_vecs = encode_queries_batch(sv_batch, self.tokenizer, self.model, self.device)
            for j, sv in enumerate(sv_batch):
                sv_vec = short_vecs[j]
                sv_labels, sv_distances = self.hnsw_index.knn_query(sv_vec.reshape(1, -1), k=min(top_k // 2, 10))
                for i, d in zip(sv_labels[0], sv_distances[0]):
                    rid = int(i)
                    row = dict(self.records[rid])
                    row["score"] = float(1.0 - float(d))
                    row["_source"] = "ann_short"
                    if rid not in [int(r.get("id")) for r in ann_selected]:
                        ann_selected.append(row)

        rrf_k = 60.0
        fused: dict[int, dict[str, Any]] = {}

        for rank, row in enumerate(lexical_hits, start=1):
            rid = int(row["id"])
            score = 1.5 / (rrf_k + rank)
            if rid not in fused:
                fused[rid] = dict(row)
                fused[rid]["score"] = 0.0
            fused[rid]["score"] += score

        for rank, row in enumerate(sparse_hits, start=1):
            rid = int(row["id"])
            score = 1.2 / (rrf_k + rank)
            if rid not in fused:
                fused[rid] = dict(row)
                fused[rid]["score"] = 0.0
            fused[rid]["score"] += score

        for rank, row in enumerate(ann_selected, start=1):
            rid = int(row["id"])
            weight = 0.8 if row.get("_source") == "ann_short" else 1.0
            score = weight / (rrf_k + rank)
            if rid not in fused:
                fused[rid] = dict(row)
                fused[rid]["score"] = 0.0
            fused[rid]["score"] += score

        coarse_merged = sorted(fused.values(), key=lambda x: float(x["score"]), reverse=True)[: max(top_n * 8, 24)]
        merged = self._rerank_results(coarse_merged, variants=all_lexical_variants, qv=qv, top_n=top_n)
        for row in merged:
            row.pop("_source", None)
            # Attach supplementary notes from CSV col-6 if present
            hw = str(row.get("shanghai", "")).strip().strip("【】[] ").strip()
            note = self._csv_notes.get(hw, "")
            if not note:
                # Also try matching against definition field
                defn_key = str(row.get("definition", "")).strip().strip("【】[] ").strip()
                note = self._csv_notes.get(defn_key, "")
            if note:
                row["notes"] = note

        elapsed = time.perf_counter() - t0
        return {
            "query": query,
            "normalized_query": normalized_query,
            "query_variants": variants,
            "top_k": top_k,
            "top_n": top_n,
            "ann": self.ann,
            "index_dir": str(self.index_dir),
            "model_name_or_path": str(self.meta.get("model_name_or_path", "")),
            "load_s": self.load_s,
            "infer_s": round(elapsed, 4),
            "results": merged,
        }
