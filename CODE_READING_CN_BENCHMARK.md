# VLM-RAG 代码阅读与实验指南（中文）

这份文档用于快速带你看懂当前仓库里与 benchmark 提分最相关的代码，并能复现我们现在的开放检索实验流程。

## 1. 先看什么（阅读顺序）

1. `src/vlmrag_utils.py`
2. `src/evaluate_rag.py`
3. `src/index_corpus.py`
4. `src/run_benchmark_grid.py`
5. `BENCHMARK_IMPROVEMENT_CN.md`

建议顺序原因：

- `vlmrag_utils.py` 是能力底座（embedding、检索、hybrid、rerank、prompt 拼接）。
- `evaluate_rag.py` 是主评测入口（指标、报告结构、可比性约束）。
- `index_corpus.py` 负责生成 doc/chunk 两类索引。
- `run_benchmark_grid.py` 把网格实验自动化并产出汇总。
- `BENCHMARK_IMPROVEMENT_CN.md` 记录阶段结论与取舍。

## 2. 关键函数怎么读

### 2.1 `src/vlmrag_utils.py`

- `build_text_embeddings(...)`
  - 负责文本向量化，支持 `sentence-transformers` 与 fallback。
- `_bm25_scores(...)`
  - 词法 BM25 打分，用于 hybrid。
- `retrieve_topk(...)`
  - 当前检索主逻辑：
  - dense 打分；
  - 可选 hybrid 融合（`hybrid_alpha`）；
  - 可选 source-group 过滤（仅诊断用，不用于主榜）。
- `rerank_results_by_overlap(...)`
  - 轻量重排（基于 query 与证据词重叠）。
- `format_rag_context(...)` / `render_rag_prompt(...)`
  - 证据注入与模板渲染（`direct/cited/strict`）。

### 2.2 `src/evaluate_rag.py`

- `parse_args()`
  - 重点参数：`--retrieval-mode --hybrid-alpha --rerank --rerank-pool-size --prompt-template --retrieval-k`。
  - `--max-samples` 用于 smoke。
  - `--restrict-source-group` 默认被禁止（除非加 `--allow-diagnostic-group-restriction`）。
- `main()`
  - 流程：加载 eval -> 检索/重排 -> 可选生成 -> 统计指标 -> 导出 JSON/CSV/failure。
- 报告字段
  - 聚合：`hit_rate_at_k`, `retrieval_mrr`, `EM`, `Token F1`
  - 行级：`retrieved_ids`, `retrieval_hit`, `retrieval_mrr`
  - 元信息：时间戳、git sha、配置签名

### 2.3 `src/index_corpus.py`

- 支持三种输出模式：
  - 兼容旧版：`--output`
  - 新版双索引：`--doc-output + --chunk-output`
- `index_config.level` 会标记 `doc/chunk/single`，便于后续审计。

### 2.4 `src/run_benchmark_grid.py`

- 网格维度：`k × alpha × template × rerank`
- 输出固定三件：
  - `summary.json`（聚合）
  - `rows.csv`（平铺对比表）
  - `failure_cases.json`（失败样例集合）

## 3. 复现实验（开放检索）

先激活环境：

```bash
cd /home/chenj/.openclaw/workspace/qwen2-vl-vlmrag-playground
source .venv/bin/activate
```

### 3.1 建索引（doc + chunk）

```bash
python src/index_corpus.py \
  --corpus benchmarks/val_docvqa/corpus.jsonl \
  --doc-output indexes/bench_docvqa_doc.json \
  --chunk-output indexes/bench_docvqa_chunk_40_10_v2.json \
  --chunk-size-words 40 \
  --chunk-overlap-words 10 \
  --embedding-backend auto
```

### 3.2 跑 smoke 网格（先选默认配置）

```bash
python src/run_benchmark_grid.py \
  --eval-set benchmarks/val_docvqa/eval_set.jsonl \
  --index indexes/bench_docvqa_doc.json \
  --output-dir reports/benchmark_grid_docvqa_open_smoke \
  --retrieval-k 1,3 \
  --hybrid-alpha 0.3,0.7 \
  --smoke-max-samples 20
```

### 3.3 跑 val 全量网格

```bash
python src/run_benchmark_grid.py \
  --eval-set benchmarks/val_docvqa/eval_set.jsonl \
  --index indexes/bench_docvqa_doc.json \
  --output-dir reports/benchmark_grid_docvqa_open_full \
  --retrieval-k 1,3,5 \
  --hybrid-alpha 0.3,0.5,0.7
```

InfographicVQA 同理，把 `--eval-set` 和 `--output-dir` 换成对应路径。

## 4. 怎么读结果（最实用）

1. 先看 `summary.json`
  - 找 `retrieval_mrr` 与 `hit_rate_at_k` 最优配置。
2. 再看 `rows.csv`
  - 横向比较 `k/alpha/template/rerank` 的变化趋势。
3. 最后看 `failure_cases.json`
  - 提取 3 条提升样例和 3 条失败样例，写出“为什么”。

## 5. 结果解释边界（非常重要）

- 主结论只认开放检索（无 group 限制）。
- `--restrict-source-group` 仅用于诊断上限，不用于可比跑分。
- 如果某配置在 DocVQA 提升、但 Infographic 仍低，优先怀疑语料可区分信号不足，而不是先盲调 alpha。
