# 代码阅读指南

本指南按实用顺序帮你阅读项目，并更安全地修改代码。

## 1）快速路径：按此顺序读文件

1. `src/run_qwen2_vl_chat.py`
2. `src/vlmrag_utils.py`（VLM 路径）
3. `src/index_corpus.py`
4. `src/vlmrag_utils.py`（RAG 路径）
5. `src/query_with_rag.py`

为何这样排：从最小的可运行入口开始，再展开共享内部实现，最后回到完整 RAG 流程。

## 2）逐文件地图

### `src/run_qwen2_vl_chat.py`

最小的 CLI 入口。

- 解析 `--image`、`--prompt` 以及共享的模型参数。
- 调用 `vlmrag_utils.py` 中的 `generate_multimodal_answer(...)`。
- 打印答案与运行摘要。

当你只想调试模型加载、图像处理或生成设置时，用这个文件。

### `src/index_corpus.py`

构建本地检索索引。

- 加载 `corpus.jsonl` 记录。
- 通过调用 `build_text_embeddings(...)` 构建嵌入。
- 保存包含 `entries`、`embedding` 和 `vectors` 的 JSON 索引。

当你要改语料结构或检索后端时，用这个文件。

### `src/query_with_rag.py`

端到端对比入口。

- 加载索引并检索 top-k 上下文。
- 运行纯 VLM 回答。
- 将上下文拼到提示前，运行 RAG 增强回答。
- 并排打印两种输出。

当你要调提示结构、检索条数或对比行为时，用这个文件。

### `src/vlmrag_utils.py`

核心实现文件，主要有两块：VLM 推理与检索。

#### VLM 推理相关函数

- `add_common_model_args(parser)`
  - 在各脚本间集中定义共享 CLI 参数。
- `choose_device(force_cpu)` 与 `choose_dtype(device)`
  - 选择运行设备与精度。
- `load_image(image_path, resize_max_edge)`
  - 打开图像并可选择缩放。
- `_load_model_class()`
  - 根据 transformers 可用性选择模型类。
- `load_qwen2_vl(model_name, device, dtype)`
  - 加载模型与 processor。
  - 使用 `_MODEL_CACHE`，同一进程内不重复加载权重。
- `generate_multimodal_answer(...)`
  - 完整多模态生成路径：
    1. device / dtype / 图像
    2. chat 模板
    3. 分词
    4. 生成
    5. 解码与运行摘要

#### 检索相关函数

- `tokenize_text(text)`
  - 用于后备嵌入的基础词法分词。
- `hashed_bow_embed(texts)` + `l2_normalize(matrix)`
  - 内置的轻量检索嵌入后端。
- `build_text_embeddings(texts, embedding_backend, embedding_model)`
  - 优先尝试 `sentence-transformers`，否则回退到哈希词袋。
- `load_corpus_jsonl(corpus_path)`
  - 读取语料记录并校验必填字段。
- `save_index(output_path, payload)` / `load_index(index_path)`
  - 索引持久化读写。
- `retrieve_topk(query, index_payload, topk)`
  - 编码查询，计算点积相似度，返回 top-k 条目。
- `format_rag_context(results)`
  - 将检索命中格式化为可直接拼进提示的文本。

## 3）数据流一图流

### 纯 VLM

`run_qwen2_vl_chat.py`
-> `generate_multimodal_answer(...)`
-> 模型输出 + 运行摘要

### VLM-RAG

`index_corpus.py`
-> 构建向量并保存 `indexes/rag_corpus_index.json`

`query_with_rag.py`
-> `load_index(...)`
-> `retrieve_topk(...)`
-> `format_rag_context(...)`
-> 纯 VLM 回答
-> RAG VLM 回答
-> 对比输出

## 4）建议优先调的参数

若你在学习并希望快速迭代，可先调这些：

- `--max-new-tokens`
  - 越大答案越长，但越慢。
- `--resize-max-edge`
  - 越小显存压力通常越低，推理往往更快。
- `--topk`
  - 检索条数越多上下文越丰富，但可能冲淡相关性。
- `--embedding-backend`
  - `auto` 使用当前环境能拿到的较好质量；`hashed-bow` 为不额外加载模型的后备方案。

## 5）安全修改流程

1. 一次只改一种行为。
2. 固定一张测试图、一条固定提示，便于快速 A/B。
3. 再分别重跑：
   - 纯 VLM（`run_qwen2_vl_chat.py`）
   - RAG 对比（`query_with_rag.py`）
4. 最后再动语料或提示格式。

这样可避免在同一轮调试里把检索改动与生成改动混在一起。

## 6）当前版本的已知限制

- 演示用检索语料很小，top-k 质量可能看起来不稳定。
- `query_with_rag.py` 在同一进程内对比输出，但每次单独执行脚本仍是新进程。
- 使用简单的文本上下文注入策略；尚无引用/出处约束。

## 7）建议的后续重构方向

1. 在两个入口增加 `--system-prompt`，便于更干净地做提示实验。
2. 在一轮检索后增加可选重排序。
3. 增加 JSON 输出模式，便于记录与画图。
4. 为索引 I/O 与检索排序行为补充单元测试。
