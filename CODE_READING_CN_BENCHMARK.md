# 多模态 Benchmark 代码阅读与实验指南（中文）

这份文档专门对应当前仓库的“多模态 RAG baseline”实现，目标是让你快速看懂：

1. 为什么文本 RAG 在图文 benchmark 上会失效
2. 双塔多模态检索在代码里怎么落地
3. 怎么复现 DocVQA / InfographicVQA 的可比实验

---

## 1. 先读什么（建议顺序）

1. `src/mm_retrieval_utils.py`
2. `src/index_corpus.py`
3. `src/evaluate_rag.py`
4. `src/run_benchmark_grid.py`
5. `BENCHMARK_IMPROVEMENT_CN.md`

阅读重点：

- `mm_retrieval_utils.py`：多模态检索核心（图像 embedding + 文本 embedding + late fusion）
- `index_corpus.py`：如何把 corpus 变成“文本向量 + 图像向量”的联合索引
- `evaluate_rag.py`：text/hybrid/multimodal 三种模式如何统一评测
- `run_benchmark_grid.py`：如何批量跑 `k × alpha × template × rerank`

---

## 2. 关键代码入口

## 2.1 `src/mm_retrieval_utils.py`

- `compute_image_embeddings(...)`
  - 用双塔模型（默认 SigLIP）把图像转成向量。
- `retrieve_topk_multimodal(...)`
  - 计算三组分数：
  - 文本相似度（query text vs text vectors）
  - 图像相似度（query image vs image vectors）
  - 线性融合分数（`text_alpha * text_norm + image_alpha * image_norm`）

这里是第一版 baseline 的核心假设：

- 不做重型 cross-attention reranker；
- 先用轻量 late fusion 验证“视觉信号是否真的能提升召回”。

## 2.2 `src/index_corpus.py`

新增多模态相关参数：

- `--with-image-embeddings`
- `--multimodal-model`
- `--eval-set-for-images`
- `--mm-cpu`

多模态索引要点：

- 索引仍保留文本向量（兼容 text/hybrid）。
- 当 `--with-image-embeddings` 打开时，额外写入 `image_vectors`。
- 如果某条语料没有可解析图像路径，会直接报错，避免“假多模态索引”。

## 2.3 `src/evaluate_rag.py`

新增参数：

- `--retrieval-mode text|dense|hybrid|multimodal`
- `--multimodal-model`
- `--image-alpha`
- `--text-alpha`
- `--mm-cpu`

逻辑：

- `text/dense/hybrid` 走原有 `retrieve_topk(...)`
- `multimodal` 走 `retrieve_topk_multimodal(...)`

主结果约束：

- 默认禁用 `--restrict-source-group`，保证 benchmark 可比（open retrieval）。

## 2.4 `src/run_benchmark_grid.py`

新增多模态网格：

- `--multimodal`
- `--multimodal-model`
- `--image-alpha-grid`
- `--text-alpha`

产物固定：

- `summary.json`
- `rows.csv`
- `failure_cases.json`

---

## 3. 为什么文本 RAG 在 InfographicVQA 上容易低分

一句话：问题依赖版面视觉证据，而我们此前只检索文本。

常见失败模式：

- 相似文本在不同图中高度重复，文本向量难区分正确样本。
- query 实际问的是“图里哪个区域/图例/布局关系”，文本无法完整表达。
- top-k evidence 看起来“语义相关”，但不是当前图片对应证据。

所以这轮改造的意义，不是“调一个更好的 alpha”，而是把视觉证据纳入检索本身。

---

## 4. 复现实验（推荐命令）

先激活环境：

```bash
cd /home/chenj/.openclaw/workspace/qwen2-vl-vlmrag-playground
source .venv/bin/activate
```

### 4.1 构建多模态 doc-level 索引

```bash
python src/index_corpus.py \
  --corpus benchmarks/val_docvqa/corpus.jsonl \
  --doc-output indexes/bench_docvqa_doc_mm_siglip.json \
  --with-image-embeddings \
  --multimodal-model google/siglip-base-patch16-224 \
  --eval-set-for-images benchmarks/val_docvqa/eval_set.jsonl
```

### 4.2 单次多模态评测

```bash
python src/evaluate_rag.py \
  --eval-set benchmarks/val_docvqa/eval_set.jsonl \
  --index indexes/bench_docvqa_doc_mm_siglip.json \
  --retrieval-mode multimodal \
  --retrieval-k 3 \
  --image-alpha 0.7 \
  --text-alpha 0.5 \
  --prompt-template strict \
  --run-generation \
  --output-json reports/docvqa_mm_k3.json \
  --output-csv reports/docvqa_mm_k3.csv \
  --failure-json reports/docvqa_mm_k3_failures.json
```

### 4.3 多模态 smoke 网格

```bash
python src/run_benchmark_grid.py \
  --eval-set benchmarks/val_infographic_vqa/eval_set.jsonl \
  --index indexes/bench_infographic_doc_mm_siglip.json \
  --output-dir reports/benchmark_grid_infographic_mm_smoke \
  --multimodal \
  --retrieval-k 1,3,5 \
  --image-alpha-grid 0.3,0.5,0.7 \
  --text-alpha 0.5 \
  --smoke-max-samples 30
```

---

## 5. 如何解读结果（最实用）

先看 `summary.json`：

- 关注 `hit_rate_at_k` 和 `retrieval_mrr` 的变化。
- 对比同 `k` 下不同 `image_alpha`，判断视觉分数贡献是否稳定。

再看 `failure_cases.json`：

- `generation_improvement`：找“图像信号明显帮助召回”的样本。
- `retrieval_miss`：找“即使有图像 embedding 也失败”的样本。

最后看 `rows.csv`：

- 筛 3 条提升样例 + 3 条失败样例，写出可解释原因。

---

## 6. 下一步学习路线（基于当前实现）

1. 先固定一组 DocVQA/Infographic 都不过度退化的默认配置。
2. 在 doc-level 稳定后，再做 chunk-level 多模态扩展。
3. 引入更强的第二阶段 rerank（仍不改主 VLM）。
4. 每轮只改一个变量，并保留 `summary + rows + failure` 三件套报告。
