# Architecture — Shanghai-TTS Pipeline

> **本文档是管线的权威定义。任何修改必须先更新本文档。**

## 生产管线（app.py 实际运行的流程）

```
用户输入: "傍晚用上海话怎么说"
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Query Preprocessing (普通话 → 搜索词)              │
│                                                             │
│  后端: bilstm_joint 或 qwen_lora (二选一，config 控制)       │
│  文件: app/joint/bilstm_runtime.py                          │
│        app/app.py → LocalQueryPreprocessor (Qwen LoRA)      │
│                                                             │
│  输入: 普通话用户 query 字符串                               │
│  处理: normalize_query() 去壳 → 分词 → 编码 embedding        │
│  输出: {                                                    │
│    core_text: "傍晚",        # 去壳后的核心文本              │
│    type: "词项",              # 词项 / 动作短语              │
│    keywords: ["傍晚", ...],  # 扩展搜索词                   │
│    segments: ["傍", "晚"],   # BMES 分词结果                │
│    embedding: [...]          # query 向量 (供 ANN 检索)      │
│  }                                                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Recall (搜索词 → 字典条目)                          │
│                                                             │
│  文件: app/recall/engine.py + recall_service.py (port 8088) │
│  索引: app/recall/index_local_bge_m3/                       │
│                                                             │
│  输入: Step 1 的 core_text + keywords + embedding           │
│  处理: 混合检索                                              │
│    - Lexical: headword/definition 精确匹配                   │
│    - BM25: CJK bigrams + ASCII 稀疏检索                     │
│    - ANN: bge-m3 dense embedding HNSW 近似最近邻            │
│    - RRF: reciprocal rank fusion 融合三路                    │
│    - Re-rank: dense(0.45) + field(0.85) + fused(0.55)       │
│  输出: top-N 字典条目列表:                                   │
│    [{                                                        │
│      shanghai: "挨夜快",       # 上海话词条（方括号已去掉）   │
│      definition: "傍晚...",    # 释义                        │
│      wu_pinyin: "a55ia23khua34",  # 吴语拼音（仅 CSV 有）    │
│      id: 1234,                                              │
│      score: 0.85                                            │
│    }, ...]                                                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: TTS (字典条目 → 语音)                               │
│                                                             │
│  文件: app/tts_engine.py                                    │
│  模型: VITS (懒加载，首次请求时才 load)                      │
│                                                             │
│  触发方式 A — 用户在聊天界面点击 "▶ 生成语音" 按钮:          │
│    → POST /api/chat 带 tts_headword="挨夜快"                │
│    → find_headword_row() 从 CSV 查 wu_pinyin                │
│    → synthesize(wu_pinyin)                                  │
│                                                             │
│  触发方式 B — 方括号直接匹配:                                │
│    → 用户输入 "[挨夜快]"                                     │
│    → find_headword_row("挨夜快") → CSV 查 wu_pinyin         │
│    → synthesize(wu_pinyin)                                  │
│                                                             │
│  synthesize() 内部:                                         │
│    wu_pinyin "a55ia23khua34"                                │
│    → prepare_text() → Unicode 映射符号                      │
│    → VITS 推理 → .wav 文件                                  │
└─────────────────────────────────────────────────────────────┘
```

## 重要：seg_pos 的定位

**seg_pos 目前不在生产管线中。** 它是一个 **query preprocessor 的替代方案**，和 bilstm_joint / qwen_lora 同级。

```
Query Preprocessor 后端选项 (三选一):
  1. bilstm_joint  — app/joint/bilstm_runtime.py  ← 当前默认
  2. qwen_lora     — app/app.py LocalQueryPreprocessor ← bench 测试用
  3. bilstm_seg_pos — app/seg_pos/seg_pos_runtime.py  ← 未接入生产
```

**seg_pos 的输入是普通话用户 query，不是上海话字典文本。**

`SegPosPreprocessor.preprocess()` 的处理流程:
1. `normalize_query(query)` — 去掉 "用上海话怎么说" 等壳
2. `clean_text(core_text)` — 去空白
3. `_predict_tags(segment_text)` — BMES 分词 + POS 标注
4. `_expand_chunks()` — 生成扩展搜索关键词
5. 返回 `{core_text, type, keywords, segments, ...}` — 和其他 preprocessor 同 schema

它和 bilstm_joint 的区别是分词模型不同（BiLSTM-CRF v3 vs BiLSTM joint），但功能相同：从普通话 query 提取搜索词。

**如果将来要把它用于上海话字典文本的韵律分词（tone sandhi domain segmentation），需要重新设计 `preprocess()` 接口。**

## 字典数据结构

CSV 文件: `processed_results.csv`

| 列号 | 字段名 | 示例 | 说明 |
|------|--------|------|------|
| 0 | shanghai | `【黄胖日头】` | 词条（带方括号） |
| 1 | shanghai_dup | `【黄胖日头】` | 重复词条 |
| 2 | romanization | `whangpangnikdhou` | 罗马字注音 |
| 3 | IPA | `ɦuɑ̃ pʰɑ̃ ȵiɪʔ dɤ` | 国际音标 |
| 4 | definition | `〈名〉夏天被薄云层...` | 释义 |
| 5 | wu_pinyin | `waon22phaon55gniq2deu21` | 吴语拼音（带声调数字） |

Recall 索引 (`records.jsonl`): `{id, shanghai, definition, index_text}`
- 注意: **wu_pinyin 不在索引中**，只在原始 CSV 里
- TTS 时通过 `find_headword_row()` 从 CSV 查 wu_pinyin

## 模块职责

### `app/shared/` — 共享工具
- `text.py` — `clean_text`, `normalize_query`, `is_meaningful`, `LOW_INFORMATION_TOKENS`
- `model_defs.py` — `CRF`, BMES/POS 常量, `encode_chars`, `PAD`, `UNK`
- `config.py` — `load_runtime_config()`, 路径解析

### `app/app.py` — Flask Web 应用 (port 8081)
- 路由: `/api/chat`, `/api/user/confinfo`, `/download/<filename>`
- 管理 RecallServiceManager (自动启动 recall 服务)
- 支持 text-only 模式 (`WUU_TEXT_ONLY=1`)

### `app/joint/` — Query Preprocessor (bilstm_joint)
- `bilstm_runtime.py` → `BilstmJointPreprocessor`
- 普通话 query → 分词 + embedding → 供 recall 搜索

### `app/seg_pos/` — Seg+POS 模型 (未接入生产)
- `seg_pos_runtime.py` → `SegPosPreprocessor` — 功能同 bilstm_joint，分词模型不同
- `train_bilstm_seg_pos_v3.py` — v3 训练脚本 (BiLSTM-CRF)
- `eval_seg_pos_recall.py` — 评估 seg_pos 作为 query preprocessor 对 recall 的影响

### `app/recall/` — 混合检索引擎
- `engine.py` — `RecallEngine`: lexical + BM25 + ANN(bge-m3) + RRF
- `recall_service.py` — HTTP API (port 8088)
- `model_utils.py` — 模型路径解析 + ModelScope 下载

### `app/tts_engine.py` — VITS TTS
- `synthesize(text, output_filename)` — wu_pinyin → VITS → .wav
- `prepare_text()` — 吴语拼音 → Unicode 映射符号
- 懒加载: 首次请求时才加载模型

### `app/split/` — Query Normalization (Teacher-Student)
- `src/split_distill/rules.py` — 规则化 query 归一化
- 蒸馏数据管线脚本

## Benchmark 模块 (bench.py)

`bench.py` 评估三个模块的质量:

| 模块 | 评估内容 | 测试数据 | 指标 |
|------|---------|---------|------|
| split | Qwen LoRA query 归一化 | test_clean_200.jsonl (人工标注) | core_text_em, type_acc, keywords_f1 |
| recall | 字典检索准确率 | test_clean_200_recall.jsonl (API 标注) | hit@1/3/10, MRR, NDCG |
| seg_pos | 分词模型质量 | seg_pos/data/test.jsonl (上海话) | seg_f1, pos_acc |

注意: bench.py 中 seg_pos 评估的是**模型本身的分词质量**（在上海话测试集上），不是端到端管线指标。

## 配置

`app/configs/runtime_paths.json`:

| Section | Key Fields |
|---------|-----------|
| `llm` | `base_model_dir`, `lora_checkpoint_dir`, `load_in_4bit` |
| `query_preprocessor` | `backend` (qwen_lora/bilstm_joint), `bilstm_checkpoint_dir` |
| `tts` | `model_dir`, `config_file`, `checkpoint_file` |
| `recall` | `script_path`, `index_dir`, `port`, `auto_start` |
| `data` | `dictionary_csv` |
