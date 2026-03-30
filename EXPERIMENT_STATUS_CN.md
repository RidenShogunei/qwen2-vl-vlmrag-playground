# 实验现状与下一步（2026-03-30）

## 1. 本地目录已清理后的实验产物

当前 `reports/` 只保留 4 套核心结果，每套仅保留三件套：

- `summary.json`
- `rows.csv`
- `failure_cases.json`

目录：

1. `reports/benchmark_grid_docvqa_open_full/`
2. `reports/benchmark_grid_infographic_open_full/`
3. `reports/benchmark_grid_docvqa_mm_smoke/`
4. `reports/benchmark_grid_infographic_mm_smoke/`

说明：

- 大量逐配置中间文件（`k*_*.json/csv`）已删除。
- 旧版散落在 `reports/` 根目录的 baseline 单次文件已删除。

## 2. 当前关键结果（来自 summary.json）

### 2.1 文本/混合检索（open full）

- DocVQA（54 runs）
  - best `hit_rate@k = 0.675`（`k5_a0.3_direct_norank`）
  - best `MRR = 0.3705`（`k5_a0.3_direct_rerank`）

- InfographicVQA（54 runs）
  - best `hit_rate@k = 0.0083`
  - best `MRR = 0.0038`
  - 仍然非常低，说明纯文本信号在该任务上几乎失效。

### 2.2 多模态检索（multimodal smoke）

- DocVQA（24 runs）
  - best `hit_rate@k = 0.8333`
  - best `MRR = 0.5889`
  - 相比 DocVQA open full 有明显提升迹象。

- InfographicVQA（24 runs）
  - best `hit_rate@k = 0.9667`
  - best `MRR = 0.9667`
  - 与文本 open full 相比提升巨大。

## 3. 结果解释边界（非常重要）

这轮多模态结果是 `smoke` 级别，不应直接作为最终结论：

1. 目前对比是 `open full (text)` vs `smoke (multimodal)`，规模不一致。
2. Infographic 的超高分可能包含数据组织/映射上的强先验效应，需进一步审计。
3. 主结论必须用同规模（val 全量）重跑后再定。

## 4. 新计划（直接可执行）

### 阶段 A：可比验证（优先级最高）

1. 用同一 eval 集规模，分别跑：
   - `retrieval-mode hybrid`
   - `retrieval-mode multimodal`
2. 固定同一网格：`k={1,3,5}`、模板 `direct/cited/strict`、rerank `on/off`。
3. 产出同结构报告并做逐项对比。

### 阶段 B：排查“异常高分”来源

1. 抽样检查 `rows.csv` 里的检索证据与 `source_id` 映射。
2. 逐条核对 30 条高分样本，确认是否存在隐性泄漏或过强先验。
3. 记录 3 条提升样例 + 3 条失败样例（含证据解释）。

### 阶段 C：确定默认配置

目标：确定一组“DocVQA 不退化 + Infographic 有提升”的默认参数：

- `retrieval_k`
- `image_alpha / text_alpha`
- `prompt_template`
- `rerank` 与 `rerank_pool_size`

## 5. 当前建议默认起点

- `retrieval-mode = multimodal`
- `image_alpha = 0.7`
- `text_alpha = 0.5`
- `retrieval_k = 3`
- `prompt_template = direct`
- `rerank = off`（先保留简洁链路，再决定是否启用）

