# Qwen2-VL VLM-RAG Playground

一个用于学习 `Qwen/Qwen2-VL-2B-Instruct` 与 VLM-RAG 的可复现实验仓库，当前已支持：

- 文本检索基线（dense / hybrid）
- 多模态检索基线（SigLIP 双塔 + late fusion）
- DocVQA / InfographicVQA benchmark 评测与网格实验

## 核心脚本

- `src/run_qwen2_vl_chat.py`：单图问答最小入口
- `src/index_corpus.py`：构建 doc/chunk 索引（可选图像向量）
- `src/query_with_rag.py`：单条 query 的 plain vs rag 对比
- `src/evaluate_rag.py`：主评测入口（hit_rate/MRR/EM/F1）
- `src/run_benchmark_grid.py`：批量网格实验
- `src/mm_retrieval_utils.py`：多模态检索工具（SigLIP 双塔）

中文导读：

- `GUIDE_CN.md`
- `CODE_READING_CN_BENCHMARK.md`
- `BENCHMARK_IMPROVEMENT_CN.md`

## 环境

```bash
cd /home/chenj/.openclaw/workspace/qwen2-vl-vlmrag-playground
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
```

## 文本/混合检索基线

### 构建索引

```bash
python src/index_corpus.py \
  --corpus benchmarks/val_docvqa/corpus.jsonl \
  --doc-output indexes/bench_docvqa_doc.json \
  --chunk-output indexes/bench_docvqa_chunk_40_10.json \
  --chunk-size-words 40 \
  --chunk-overlap-words 10
```

### 单次评测（open retrieval）

```bash
python src/evaluate_rag.py \
  --eval-set benchmarks/val_docvqa/eval_set.jsonl \
  --index indexes/bench_docvqa_doc.json \
  --retrieval-mode hybrid \
  --hybrid-alpha 0.7 \
  --retrieval-k 3 \
  --prompt-template strict \
  --rerank \
  --run-generation \
  --output-json reports/docvqa_hybrid_k3.json \
  --output-csv reports/docvqa_hybrid_k3.csv \
  --failure-json reports/docvqa_hybrid_k3_failures.json
```

## 多模态检索基线（SigLIP 双塔）

### 1) 构建多模态索引

`corpus.jsonl` 保持 `id/text/source_id/meta`，图像路径优先从语料条目 `image` 字段读取；
如果语料没有 `image`，可用 `--eval-set-for-images` 按 `source_id` 映射。

```bash
python src/index_corpus.py \
  --corpus benchmarks/val_infographic_vqa/corpus.jsonl \
  --doc-output indexes/bench_infographic_doc_mm_siglip.json \
  --with-image-embeddings \
  --multimodal-model google/siglip-base-patch16-224 \
  --eval-set-for-images benchmarks/val_infographic_vqa/eval_set.jsonl
```

### 2) 单次多模态评测

```bash
python src/evaluate_rag.py \
  --eval-set benchmarks/val_infographic_vqa/eval_set.jsonl \
  --index indexes/bench_infographic_doc_mm_siglip.json \
  --retrieval-mode multimodal \
  --multimodal-model google/siglip-base-patch16-224 \
  --image-alpha 0.7 \
  --text-alpha 0.5 \
  --retrieval-k 3 \
  --prompt-template cited \
  --run-generation \
  --output-json reports/infographic_mm_k3.json \
  --output-csv reports/infographic_mm_k3.csv \
  --failure-json reports/infographic_mm_k3_failures.json
```

### 3) 多模态网格实验（smoke）

```bash
python src/run_benchmark_grid.py \
  --eval-set benchmarks/val_infographic_vqa/eval_set.jsonl \
  --index indexes/bench_infographic_doc_mm_siglip.json \
  --output-dir reports/benchmark_grid_infographic_mm_smoke \
  --multimodal \
  --multimodal-model google/siglip-base-patch16-224 \
  --retrieval-k 1,3,5 \
  --image-alpha-grid 0.3,0.5,0.7 \
  --text-alpha 0.5 \
  --smoke-max-samples 30
```

固定输出：

- `summary.json`
- `rows.csv`
- `failure_cases.json`

## 可比性约束（重要）

- 主结果必须是 `open retrieval`。
- 默认禁止 `--restrict-source-group`（仅诊断时才可加 `--allow-diagnostic-group-restriction`）。
- 不引入 benchmark 外部语料。

## 常见问题

- 首次模型下载慢：正常。
- SigLIP 报 tokenizer / protobuf 错误：确认安装 `sentencepiece` 与 `protobuf`。
- 显存不足：先降 `retrieval_k`、缩小图像，必要时加 `--mm-cpu` / `--cpu` 做 smoke。

