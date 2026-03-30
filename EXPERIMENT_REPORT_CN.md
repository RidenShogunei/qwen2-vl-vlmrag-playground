# 全量实验报告（多模态 RAG，2026-03-30）

## 1. 实验目标

本轮目标：在不改主 VLM（`Qwen2-VL-2B-Instruct`）的前提下，验证“多模态检索（SigLIP 双塔 + late fusion）”对 benchmark 检索质量的提升。

对比对象：

- 开放检索文本基线（`open_full`，已有）
- 多模态开放检索全量网格（`mm_full`，本轮新增）

数据：

- DocVQA（val，可用样本 200）
- InfographicVQA（val，可用样本 600）

## 2. 实验配置

多模态 full grid：

- `retrieval_mode = multimodal`
- `k ∈ {1,3,5}`
- `image_alpha ∈ {0.3,0.5,0.7}`
- `text_alpha = 0.5`
- prompt template：`direct/cited/strict`
- rerank：`off/on`
- 共 `54 runs` / benchmark

输出目录（已整理为三件套）：

- `reports/benchmark_grid_docvqa_mm_full/`
- `reports/benchmark_grid_infographic_mm_full/`

每个目录仅保留：

- `summary.json`
- `rows.csv`
- `failure_cases.json`

## 3. 关键结果

## 3.1 与 open baseline 的 best 对比

### DocVQA

- open_full best
  - `hit_rate@k = 0.675`
  - `MRR = 0.3705`
- mm_full best
  - `hit_rate@k = 0.965`（`k5_a0.7_direct_rerank`）
  - `MRR = 0.5794`（`k5_a0.7_direct_norank`）

结论：DocVQA 上多模态检索相对文本开放检索有显著提升。

### InfographicVQA

- open_full best
  - `hit_rate@k = 0.0083`
  - `MRR = 0.0038`
- mm_full best
  - `hit_rate@k = 0.9983`（`k5_a0.7_direct_norank`）
  - `MRR = 0.9815`（`k5_a0.7_direct_norank`）

结论：InfographicVQA 从几乎不可检索提升到接近满分级别。

## 3.2 趋势分析（mm_full）

### DocVQA（按均值）

- `k_mean_hit`：`k1=0.285`，`k3=0.6825`，`k5=0.8583`
- `alpha_mean_hit`：`0.3=0.4883`，`0.5=0.6383`，`0.7=0.6992`

观察：增大 `k` 和增大 `image_alpha` 都有稳定正向收益。

### InfographicVQA（按均值）

- `k_mean_hit`：`k1=0.7828`，`k3=0.8306`，`k5=0.8511`
- `alpha_mean_hit`：`0.3=0.555`，`0.5=0.9222`，`0.7=0.9872`

观察：视觉分数权重对该任务影响极大，`image_alpha=0.7` 明显优于低权重。

## 4. 结果解释与边界

1. 本轮是“检索层 full 网格”，未开启生成层（`run_generation=false`），结论主要针对召回/排序能力。
2. InfographicVQA 的提升幅度非常大，工程上是积极信号，但仍建议继续做样例审计（检查 evidence 与 source 映射）以排除数据组织上的异常先验。
3. 由于我们严格使用 benchmark 官方可追溯语料且主结果为 open retrieval，当前结论具备可比价值。

## 5. 推荐默认配置（下一轮）

建议先用以下配置做生成层验证：

- `retrieval_mode=multimodal`
- `retrieval_k=5`
- `image_alpha=0.7`
- `text_alpha=0.5`
- `prompt_template=direct`
- `rerank=off`（先做稳定基线，再评估 rerank 是否带来净收益）

## 6. 下一步实验

1. 在上述默认检索配置下，开启 `--run-generation` 对比 plain vs rag 的 `EM/Token F1`。
2. 从 `rows.csv` 与 `failure_cases.json` 抽取：
   - 3 条提升样例
   - 3 条失败样例
   - 并给出“检索命中为何变化”的证据解释。
3. 若生成层收益不及检索层提升，优先优化 prompt 注入与证据压缩策略。
