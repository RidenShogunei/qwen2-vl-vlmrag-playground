# 中文代码阅读与实验指南

这份文档的目标是：

1. 帮你快速看懂当前项目代码结构。
2. 帮你可重复地跑出 baseline，并继续做 RAG 质量实验。

项目路径：`/home/chenj/.openclaw/workspace/qwen2-vl-vlmrag-playground`

---

## 一、先看代码：建议阅读顺序

先按这个顺序看（每一步都能对应一个可运行脚本）：

1. `src/run_qwen2_vl_chat.py`
2. `src/vlmrag_utils.py`（先看 VLM 推理相关函数）
3. `src/index_corpus.py`
4. `src/query_with_rag.py`
5. `src/evaluate_rag.py`
6. 回到 `src/vlmrag_utils.py`（再看检索/评估相关函数）

### 1) `run_qwen2_vl_chat.py`：最小入口

你先把它当作“模型通路探针”：

- 输入：`--image`、`--prompt`
- 输出：模型回答 + 运行摘要（设备、耗时、显存）

如果这一步稳定，说明模型加载、图像处理、生成路径没问题。

### 2) `vlmrag_utils.py`：核心公共逻辑

这个文件分两块：

- VLM 推理块
  - `choose_device` / `choose_dtype`
  - `load_qwen2_vl`
  - `generate_multimodal_answer`
- 检索与提示块
  - `build_text_embeddings`
  - `retrieve_topk`
  - `rerank_results_by_overlap`
  - `render_rag_prompt`

你可以重点看这几个概念：

- 为什么 `query_with_rag.py` 可以在同一进程里复用模型（`_MODEL_CACHE`）
- 为什么评估阶段要复用 embedding 模型（`_EMBED_MODEL_CACHE`）
- prompt 模板如何影响最终输出（`direct/cited/strict`）

### 3) `index_corpus.py`：索引构建

当前支持：

- embedding backend：`auto | sentence-transformers | hashed-bow`
- chunking：`--chunk-size-words`、`--chunk-overlap-words`

它把语料转成：

- `entries`
- `vectors`
- `embedding`
- `index_config`

### 4) `query_with_rag.py`：单条 query 的对比实验

它会在同一条 query 上输出：

- plain VLM answer
- RAG-enhanced answer

并支持这些实验开关：

- `--prompt-template`
- `--retrieval-k`
- `--rerank`
- `--show-evidence`
- `--json-output`

### 5) `evaluate_rag.py`：批量评估入口（最关键）

这是你后续学习“先测量再优化”的主入口。

它输出两层指标：

- 检索层：`retrieval_hit_rate`、`retrieval_mrr`
- 生成层（可选）：`plain_keyword_hit_rate`、`rag_keyword_hit_rate`

并导出：

- JSON 总报告（带配置、指标、失败样例）
- CSV 行级结果（便于 Excel/Notebook 分析）

---

## 二、实验前准备

先激活环境：

```bash
cd /home/chenj/.openclaw/workspace/qwen2-vl-vlmrag-playground
source .venv/bin/activate
```

如果你重建过环境，确保 `torch` 是 cu121 版本（本机驱动兼容）：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

---

## 三、从 baseline 开始（必须做）

### 实验 A：仅检索 baseline

```bash
python src/index_corpus.py \
  --corpus examples/rag_corpus/corpus.jsonl \
  --output indexes/rag_corpus_index.json \
  --embedding-backend auto

python src/evaluate_rag.py \
  --eval-set examples/rag_eval_set.jsonl \
  --index indexes/rag_corpus_index.json \
  --retrieval-k 3 \
  --prompt-template direct \
  --output-json reports/baseline_direct.json \
  --output-csv reports/baseline_direct_rows.csv
```

### 实验 B：检索 + 生成 baseline

```bash
python src/evaluate_rag.py \
  --eval-set examples/rag_eval_set.jsonl \
  --index indexes/rag_corpus_index.json \
  --retrieval-k 2 \
  --prompt-template strict \
  --rerank \
  --run-generation \
  --max-new-tokens 32 \
  --resize-max-edge 640 \
  --output-json reports/baseline_with_generation.json \
  --output-csv reports/baseline_with_generation_rows.csv
```

### 如何读结果

先看 JSON 里的 `metrics`：

- `retrieval_hit_rate`：top-k 是否命中目标 id
- `retrieval_mrr`：正确结果出现得越靠前越好
- `plain_keyword_hit_rate` vs `rag_keyword_hit_rate`：生成层关键字命中

再看 `failure_examples`：

- `retrieval_miss`
- `generation_not_improved`

这两个列表是你下一轮优化最值钱的样本。

---

## 四、建议实验路线（课程式）

你可以按下面 4 轮推进，每轮都要产出 JSON/CSV 报告并 commit。

### 第 1 轮：模板 A/B

固定：

- 同一个 index
- 同一个 `retrieval-k`

只改：

- `--prompt-template direct`
- `--prompt-template cited`
- `--prompt-template strict`

看哪种模板在生成层更稳。

### 第 2 轮：k 值实验

固定模板后，测试：

- `--retrieval-k 1`
- `--retrieval-k 2`
- `--retrieval-k 3`
- `--retrieval-k 4`

观察：

- 检索命中提升是否会换来生成层退化（上下文噪声增加）。

### 第 3 轮：rerank 开关

固定模板与 k，比较：

- `--rerank` 关闭
- `--rerank` 开启

关注：

- `retrieval_mrr` 是否提升
- 生成层关键词命中是否同步提升

### 第 4 轮：chunking

重建两个索引后比较：

- 不 chunk（当前默认）
- chunk（如 `--chunk-size-words 40 --chunk-overlap-words 10`）

看 chunk 是否改善召回或反而打碎语义。

---

## 五、每轮实验后的记录模板（建议）

建议你每次 commit 备注这三件事：

1. 改了什么配置（模板/k/rerank/chunk）。
2. 指标变化（至少写 retrieval_hit_rate、retrieval_mrr、rag_keyword_hit_rate）。
3. 结论一句话（保留/回退/继续观察）。

示例：

```text
实验: strict + k=2 + rerank
变化: retrieval_mrr 1.0 -> 1.0, rag_keyword_hit_rate 0.12 -> 0.17
结论: 保留 strict，下一步只测 chunk
```

---

## 六、常见坑

1. 只看单条样例就下结论。
2. 一次改两个变量（例如同时改模板和 k），导致无法归因。
3. 检索分数升了，但生成层没升，就误判“系统整体变好”。
4. 没有固定评测集，每次结果不可比。

---

## 七、你现在最值得做的下一步

直接跑“三模板 A/B + 同一 k”的对比，把三份报告放到 `reports/`，我们再一起复盘哪种模板最适合你当前语料。
