# Shanghai-TTS

这个仓库现在包含一套可运行的双方言应用：

- 上海话词典检索 + TTS
- 绍兴话词典检索 + TTS
- 统一召回服务，端口 `127.0.0.1:8088`
- 主 Web 应用，端口 `127.0.0.1:8081`
- 两套 TTS 模型独立加载、切换、卸载

本文档是唯一主文档。`app/README.md` 和 `app/recall/README.md` 只保留跳转说明。

## 目录结构

- `app/app.py`
  - 主应用入口
  - 提供 Web UI 和主 API
  - 负责词典查询、召回、TTS 调用
- `app/tts_engine.py`
  - 双模型 TTS 运行时
  - 管理上海 / 绍兴模型的加载、迁移、卸载、朗读
- `app/recall/recall_service.py`
  - 统一召回 HTTP 服务
  - 在一个 `8088` 进程里同时服务上海和绍兴两个索引
- `app/dictionary_sources.py`
  - 词典源准备逻辑
  - 包括绍兴 `merged_result.xlsx` 转标准化 CSV
- `model/`
  - 大模型和向量模型目录
  - 例如 `bge-m3`、Qwen、VITS 相关文件

## 运行时配置

主配置文件：

- `app/configs/runtime_paths.json`

当前关键路径和语义：

- 召回服务脚本：`app/recall/recall_service.py`
- 上海索引目录：`app/recall/index_local_bge_m3`
- 绍兴索引目录：`app/recall/index_shaoxing_bge_m3`
- 召回 API：`http://127.0.0.1:8088/recall`
- 召回健康检查：`http://127.0.0.1:8088/health`
- 默认 TTS 模型：`shanghai`

TTS ckpt / config 按模型分开维护：

- 上海
  - `config.json`
  - `checkpoint_48000.pth`
- 绍兴
  - `config_27000.json`
  - `checkpoint_27000.pth`

## 主应用启动

从仓库根目录启动：

```bash
python app/app.py
```

启动后会做这些事：

1. 读取 `app/configs/runtime_paths.json`
2. 准备词典和派生数据
3. 后台预热部分运行时资源
4. 启动 Flask，监听 `127.0.0.1:8081`
5. 若 `8088` 召回服务未就绪，则自动拉起 `app/recall/recall_service.py`

## 双词典 / 双召回架构

### 词典源

当前有两套数据源：

- 上海词典
- 绍兴词典

其中绍兴词典来自仓库根目录的：

- `merged_result.xlsx`

应用会将它整理成标准化 CSV，供后续向量构造和召回使用。

### 统一召回服务

`app/recall/recall_service.py` 现在不是单索引服务，而是一个双源服务：

- `source = shanghai`
- `source = shaoxing`

统一监听：

- `http://127.0.0.1:8088`

健康检查：

```bash
curl http://127.0.0.1:8088/health
```

查询接口：

```text
POST /recall
```

示例：

```json
{
  "query": "太阳",
  "source": "shaoxing",
  "variants": ["太阳", "日头"],
  "top_k": 20,
  "top_n": 3
}
```

支持的 `source` 别名：

- `shanghai`
- `shanghai_csv`
- `shaoxing`
- `shaoxing_xlsx`

调用示例：

```bash
curl -X POST "http://127.0.0.1:8088/recall" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"你好怎么说\",\"source\":\"shanghai\",\"top_k\":20,\"top_n\":3}"
```

```bash
curl -X POST "http://127.0.0.1:8088/recall" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"太阳\",\"source\":\"shaoxing\",\"top_k\":20,\"top_n\":3}"
```

## 向量索引怎么构造

### 原始数据位置

上海源文件：

- `app/recall/processed_results.csv`

绍兴标准化文件：

- `app/data/generated/shaoxing_processed.csv`

绍兴原始表格：

- `merged_result.xlsx`

Embedding 模型目录：

- `model/bge-m3`

### 每个索引目录里应该有什么

最少需要：

- `meta.json`
- `records.jsonl`
- `embeddings.npy`

如果额外构建 HNSW，还会有：

- `index_hnsw.bin`
- `hnsw_meta.json`

### vector 文本怎么拼

`app/recall/scripts/build_vector_index.py` 支持三种 `text_mode`：

- `headword`
- `definition`
- `headword_definition`

推荐使用：

```text
<headword> + "。释义：" + <definition>
```

例子：

```text
阿拉。释义：我们
```

原因很直接：

- 只嵌入 `headword`，按普通话释义检索会弱
- 只嵌入 `definition`，按词条精确命中会弱
- `headword_definition` 在当前应用里最均衡

如果你希望上海和绍兴召回行为可比较，两边要保持相同的 `text_mode`。

### 列映射

上海 CSV 构造参数：

- `id_col = 0`
- `sh_col = 0`
- `def_col = 4`
- `header = none`

绍兴标准化 CSV 构造参数：

- `id_col = 0`
- `sh_col = 0`
- `def_col = 4`
- `header = infer`

### 构造上海索引

```bash
python app/recall/scripts/build_vector_index.py \
  --dict_csv app/recall/processed_results.csv \
  --out_dir app/recall/index_local_bge_m3 \
  --model_name_or_path model/bge-m3 \
  --id_col 0 \
  --sh_col 0 \
  --def_col 4 \
  --header none \
  --text_mode headword_definition
```

### 构造绍兴索引

先确保：

- `app/data/generated/shaoxing_processed.csv` 已存在

如果没有，先由应用侧词典准备流程从 `merged_result.xlsx` 生成。

然后执行：

```bash
python app/recall/scripts/build_vector_index.py \
  --dict_csv app/data/generated/shaoxing_processed.csv \
  --out_dir app/recall/index_shaoxing_bge_m3 \
  --model_name_or_path model/bge-m3 \
  --id_col 0 \
  --sh_col 0 \
  --def_col 4 \
  --header infer \
  --text_mode headword_definition
```

## HNSW 怎么构造

HNSW 构造脚本：

- `app/recall/scripts/build_hnsw_index.py`

它会：

1. 读取 `embeddings.npy`
2. 转成 `float32`
3. 做 L2 normalization
4. 构建 `hnswlib.Index(space="cosine", dim=...)`
5. 写出 `index_hnsw.bin` 和 `hnsw_meta.json`

常用参数：

- `m`
- `ef_construction`
- `ef_search`

当前推荐值：

- `m = 32`
- `ef_construction = 200`
- `ef_search = 64`

构造命令：

```bash
python app/recall/scripts/build_hnsw_index.py \
  --index_dir app/recall/index_local_bge_m3 \
  --m 32 \
  --ef_construction 200 \
  --ef_search 64
```

```bash
python app/recall/scripts/build_hnsw_index.py \
  --index_dir app/recall/index_shaoxing_bge_m3 \
  --m 32 \
  --ef_construction 200 \
  --ef_search 64
```

## TTS 运行时语义

当前是双模型独立管理：

- `shanghai`
- `shaoxing`

每个模型都可以单独处于以下状态之一：

- `cuda`
- `cpu`
- `unloaded`

另外还有一个单独维度：

- 当前拼音直读模型

这个状态和设备状态不是一回事。

例如可以同时出现：

- 上海：`cuda`
- 绍兴：`cpu`
- 当前拼音直读：上海

### 直读是什么意思

拼音直读的语义是：

- 输入直接当作方言拼音
- 不先走词典查词
- 直接交给“当前拼音直读模型”合成音频

这和普通查词朗读不同。普通查词朗读会先查词典，再按命中的词条和拼音去读。

## 常用 API

主应用健康相关：

```bash
curl http://127.0.0.1:8081/api/user/confinfo
curl http://127.0.0.1:8081/api/tts/status
```

常用接口：

- `POST /api/chat`
- `GET /api/tts/status`
- `POST /api/tts/load`
- `POST /api/tts/direct_model`
- `POST /api/tts/read_headword`

加载 / 迁移 / 卸载模型示例：

```json
{
  "model": "shaoxing",
  "device": "cpu"
}
```

## 前端文件

- 模板：`app/templates/index.html`
- 样式：`app/static/app.css`
- 交互：`app/static/app.js`

## 故障排查

### 8088 冷启动慢

这是预期现象。Windows 下同时加载上海和绍兴两个索引时，冷启动会明显慢一些。先等 `/health` 正常，再测查询延迟。

### 上海能查，绍兴不能查

按这个顺序检查：

1. `app/data/generated/shaoxing_processed.csv` 是否存在
2. `app/recall/index_shaoxing_bge_m3/meta.json` 是否存在
3. `app/recall/index_shaoxing_bge_m3/records.jsonl` 是否存在
4. `app/recall/index_shaoxing_bge_m3/embeddings.npy` 是否存在
5. `http://127.0.0.1:8088/health` 是否报告了 `shaoxing`

### 召回质量退化

优先比对：

- `meta.json`
- `text_mode`
- 源 CSV 是否是最新版本
- HNSW 参数是否一致

不要拿一个带 HNSW 的上海索引和一个平搜的绍兴索引直接比较延迟或召回，再把差异归因到模型本身。
