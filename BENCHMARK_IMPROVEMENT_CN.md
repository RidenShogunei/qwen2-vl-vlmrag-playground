# VLM-RAG Benchmark 改进记录（中文）

更新时间：2026-03-29

## 1. 背景与问题

在 `InfographicVQA val` 上，原始检索基线（全语料直接检索）表现很低：

- `hit_rate@3 = 0.005`
- `MRR = 0.0031`

我们之前已经试过“按图聚合语料（v2）”，但指标没有改善，说明瓶颈不在简单的语料聚合方式。

## 2. 本轮改进目标

目标：验证“同 source group 候选池约束 + 轻量 rerank”是否能显著提升检索命中。

这里的 `source group` 采用 `prefix_before_img` 规则：

- 原 `source_id` 形如：`infographic::{shard}::{row}::img{n}`
- 分组 key 形如：`infographic::{shard}::{row}`

也就是检索时只在同一文档/样本组内竞争，而不是和全量语料竞争。

## 3. 代码改动

### 3.1 `src/vlmrag_utils.py`

新增：

- `derive_source_group(source_id, mode)`
- `retrieve_topk(...)` 扩展参数：
  - `source_group_key`
  - `source_group_mode`

行为变化：

- 当传入 `source_group_key` 时，对不属于同组的条目打极低分并过滤。

### 3.2 `src/evaluate_rag.py`

新增 CLI 参数：

- `--restrict-source-group`
- `--source-group-mode {auto,exact,prefix_before_img}`
- `--rerank-pool-size`

行为变化：

- 可先取更大的候选池（`rerank_pool_size`），再做 overlap rerank，再裁到 `retrieval_k`。
- 评测报告 `config_signature` 会记录以上新参数，便于复现。

## 4. 本轮实验命令

### 4.1 仅做同组候选池约束

```bash
python src/evaluate_rag.py \
  --eval-set benchmarks/val_infographic_v2/eval_set.jsonl \
  --index indexes/bench_infographic_v2_nochunk.json \
  --retrieval-k 3 \
  --restrict-source-group \
  --source-group-mode prefix_before_img \
  --config-name infographic_group_only \
  --data-version infographic_v2 \
  --output-json reports/infographic_retrieval_group_only.json \
  --output-csv reports/infographic_retrieval_group_only.csv \
  --failure-json reports/infographic_retrieval_group_only_failures.json
```

结果：

- `samples = 600`
- `hit_rate@3 = 1.0`
- `MRR = 1.0`

### 4.2 同组候选池 + rerank（pool=20）

```bash
python src/evaluate_rag.py \
  --eval-set benchmarks/val_infographic_v2/eval_set.jsonl \
  --index indexes/bench_infographic_v2_nochunk.json \
  --retrieval-k 3 \
  --rerank \
  --rerank-pool-size 20 \
  --restrict-source-group \
  --source-group-mode prefix_before_img \
  --config-name infographic_group_rerank_pool20 \
  --data-version infographic_v2 \
  --output-json reports/infographic_retrieval_group_rerank_pool20.json \
  --output-csv reports/infographic_retrieval_group_rerank_pool20.csv \
  --failure-json reports/infographic_retrieval_group_rerank_pool20_failures.json
```

结果：

- `samples = 600`
- `hit_rate@3 = 1.0`
- `MRR = 1.0`

## 5. 结果解读

这轮改进在当前数据组织下把检索指标“拉满”了，说明：

- 之前的主要误差来自“跨文档竞争导致的候选污染”；
- 而不是 embedding 相似度在同文档内部无法区分。

但也要注意：

- 这种约束在某些任务设定里可能属于“已知分组先验”；
- 若未来目标是开放域检索（不知道 query 属于哪个 source group），该策略不能直接迁移。

## 6. 下一步建议（学习价值更高）

建议按下面顺序继续：

1. 在 DocVQA 上复现同样策略，确认是否同样有效。
2. 做“弱先验”版本：先粗召回 top-N 文档组，再在组内精排（避免硬约束过强）。
3. 打开 `--run-generation`，比较 plain vs rag 的 EM/F1，确认检索提升是否转化到生成质量。
4. 固化一份“可公开报告”的设置：
   - 什么是允许的先验
   - 什么是严格开放检索
   - 两种设定分别汇报。

## 7. 产物清单

- `reports/infographic_retrieval_group_only.json`
- `reports/infographic_retrieval_group_only.csv`
- `reports/infographic_retrieval_group_only_failures.json`
- `reports/infographic_retrieval_group_rerank_pool20.json`
- `reports/infographic_retrieval_group_rerank_pool20.csv`
- `reports/infographic_retrieval_group_rerank_pool20_failures.json`

## 8. 新尝试：非作弊版 Hybrid 检索（开放检索）

为了避免依赖 `--restrict-source-group` 的强先验，本轮新增并测试了开放检索设置：

- 检索模式：`dense` / `hybrid`（BM25 + Dense 融合）
- 可选参数：`--retrieval-mode`, `--hybrid-alpha`
- 保持 `retrieval_k=3`，不做 source group 限制

### 8.1 代码层新增能力

- `src/vlmrag_utils.py`
  - 新增 `_bm25_scores(...)`
  - 新增 `_minmax_normalize(...)`
  - `retrieve_topk(...)` 新增参数：
    - `retrieval_mode`（`dense` 或 `hybrid`）
    - `hybrid_alpha`（dense 权重）
  - 返回中额外保留 `dense_score`

- `src/evaluate_rag.py`
  - 新增 CLI：
    - `--retrieval-mode {dense,hybrid}`
    - `--hybrid-alpha`
  - 保持和已有 `rerank`、`rerank-pool-size` 兼容

### 8.2 开放检索实验命令与结果

1) Dense baseline（open）

```bash
python src/evaluate_rag.py \
  --eval-set benchmarks/val_infographic_v2/eval_set.jsonl \
  --index indexes/bench_infographic_v2_nochunk.json \
  --retrieval-k 3 \
  --retrieval-mode dense \
  --config-name infographic_open_dense_k3 \
  --data-version infographic_v2 \
  --output-json reports/infographic_open_dense_k3.json
```

结果：`hit_rate@3=0.005, MRR=0.0031`

2) Hybrid（alpha=0.7）

```bash
python src/evaluate_rag.py \
  --eval-set benchmarks/val_infographic_v2/eval_set.jsonl \
  --index indexes/bench_infographic_v2_nochunk.json \
  --retrieval-k 3 \
  --retrieval-mode hybrid \
  --hybrid-alpha 0.7 \
  --config-name infographic_open_hybrid_a07_k3 \
  --data-version infographic_v2 \
  --output-json reports/infographic_open_hybrid_a07_k3.json
```

结果：`hit_rate@3=0.005, MRR=0.0031`

3) Hybrid + rerank（pool=20）

```bash
python src/evaluate_rag.py \
  --eval-set benchmarks/val_infographic_v2/eval_set.jsonl \
  --index indexes/bench_infographic_v2_nochunk.json \
  --retrieval-k 3 \
  --retrieval-mode hybrid \
  --hybrid-alpha 0.7 \
  --rerank \
  --rerank-pool-size 20 \
  --config-name infographic_open_hybrid_a07_rerank_pool20_k3 \
  --data-version infographic_v2 \
  --output-json reports/infographic_open_hybrid_a07_rerank_pool20_k3.json
```

结果：`hit_rate@3=0.005, MRR=0.0031`

### 8.3 结论（重要）

在当前 `corpus` 组织方式下，开放检索没有因为 Hybrid 或轻量 rerank 得到提升。这个负结果是有价值的：

- 当前瓶颈不是“单纯打分公式（dense vs hybrid）”；
- 更可能是“语料构造/可区分信号不足”，导致大量样本之间可检索特征高度重复。

### 8.4 下一步（优先级）

建议优先做这两项，而不是继续盲调 alpha：

1. 语料重构：从“QA 列表文本”切到“更接近真实可检索证据”的文本（如 OCR/版面块级文本）。
2. 两跳检索：先召回文档级候选，再在候选文档内做 chunk 精排（而不是全局直接 top-k）。

## 9. 本轮落地（按提分计划实施）

这轮已完成三项工程化升级：

1. `evaluate_rag.py` 标准化
- 新增 `--max-samples` 用于 smoke 子集。
- 主配置默认禁止 `--restrict-source-group`，避免不可比“先验作弊”。
- 若确实做诊断，可显式加 `--allow-diagnostic-group-restriction`。
- 报告 `config_signature` 记录上述配置，便于审计。

2. `index_corpus.py` 支持双索引输出
- 新增 `--doc-output` 与 `--chunk-output`。
- 一次命令可同时产出 doc-level 与 chunk-level 索引。
- 索引元信息包含 `level/doc/chunk` 与 expanded 规模。

3. `run_benchmark_grid.py` 网格标准化
- 网格维度：`k × alpha × template × rerank`。
- 新增 `--smoke-max-samples`，先小样本选配置再跑全量。
- 汇总产物固定为：`summary.json + rows.csv + failure_cases.json`。

### 9.1 Smoke 验证结果（DocVQA, 20样本）

- 路径：`reports/benchmark_grid_docvqa_open_smoke/`
- 共 24 组运行，均成功输出。
- 可复现性检查：同配置重复两次结果一致：
  - `hit_rate@3 = 0.7`
  - `MRR = 0.5`

### 9.2 当前观察

- 在 smoke 子集上，`k=3` 明显优于 `k=1`，满足 `hit_rate@k` 的单调性预期。
- `alpha=0.3` 在该子集上优于 `alpha=0.7`（说明 BM25 成分对当前文本组织更有帮助）。
- 后续应基于这套流程跑 val 全量，输出正式对比结论。
