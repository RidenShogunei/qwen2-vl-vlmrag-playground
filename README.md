# Qwen2-VL VLM-RAG Playground

A small learning project for running `Qwen/Qwen2-VL-2B-Instruct` in WSL and building a minimal VLM-RAG workflow on top of it.

## What this project includes

- `CODE_READING.md`: guided walkthrough for understanding and modifying the code
- `src/run_qwen2_vl_chat.py`: single-image chat with Qwen2-VL
- `src/index_corpus.py`: build a tiny local text retrieval index for RAG
- `src/query_with_rag.py`: compare plain VLM output vs RAG-augmented output
- `src/evaluate_rag.py`: evaluate retrieval/generation metrics and export reports
- `examples/rag_corpus/corpus.jsonl`: starter corpus for retrieval
- `examples/rag_eval_set.jsonl`: 24-sample evaluation set for baseline tracking

## Environment notes

This project is designed for the current WSL setup:

- Python 3.10+
- GPU preferred, CPU supported for smoke tests
- First run will download `Qwen/Qwen2-VL-2B-Instruct` because it is not currently cached
- On this machine, install PyTorch with the CUDA 12.1 wheel. The generic latest wheel may pick CUDA 13 and fail against the current NVIDIA driver

## Quick start

### 1. Create a virtual environment

```bash
cd /home/chenj/.openclaw/workspace/qwen2-vl-vlmrag-playground
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
```

### 2. Run plain image chat

```bash
python src/run_qwen2_vl_chat.py \
  --image /absolute/path/to/your/image.png \
  --prompt "Describe the UI and list anything notable."
```

### 3. Build retrieval index

```bash
python src/index_corpus.py \
  --corpus examples/rag_corpus/corpus.jsonl \
  --output indexes/rag_corpus_index.json \
  --embedding-backend auto
```

Optional chunking:

```bash
python src/index_corpus.py \
  --corpus examples/rag_corpus/corpus.jsonl \
  --output indexes/rag_corpus_index_chunked.json \
  --embedding-backend auto \
  --chunk-size-words 40 \
  --chunk-overlap-words 10
```

### 4. Query with and without RAG

```bash
python src/query_with_rag.py \
  --image /absolute/path/to/your/image.png \
  --prompt "Which product area does this screen belong to?" \
  --index indexes/rag_corpus_index.json \
  --retrieval-k 3 \
  --prompt-template cited \
  --rerank \
  --show-evidence
```

## Prompt templates

`query_with_rag.py` supports three templates:

- `direct`: straightforward context injection
- `cited`: asks model to cite evidence ids like `[1]`
- `strict`: restricts claims to evidence and encourages explicit uncertainty

Use `--prompt-template {direct|cited|strict}` for A/B testing.

## Evaluation workflow (single command)

Run baseline retrieval metrics only:

```bash
python src/evaluate_rag.py \
  --eval-set examples/rag_eval_set.jsonl \
  --index indexes/rag_corpus_index.json \
  --retrieval-k 3 \
  --prompt-template direct \
  --output-json reports/baseline_direct.json \
  --output-csv reports/baseline_direct_rows.csv
```

Run retrieval + generation keyword checks:

```bash
python src/evaluate_rag.py \
  --eval-set examples/rag_eval_set.jsonl \
  --index indexes/rag_corpus_index.json \
  --retrieval-k 3 \
  --prompt-template strict \
  --rerank \
  --run-generation \
  --output-json reports/strict_rerank_with_gen.json \
  --output-csv reports/strict_rerank_with_gen_rows.csv
```

The JSON report includes config, metrics, timestamp, optional git sha, row-level outcomes, and failure examples.

## How the minimal VLM-RAG flow works

1. A small corpus of reference descriptions is embedded into vectors.
2. A user question retrieves top-k reference entries.
3. Optional lexical rerank adjusts order based on query-term overlap.
4. Retrieved context is injected by a selected prompt template.
5. The script compares plain VLM answer vs RAG-enhanced answer.

## Corpus format

`corpus.jsonl` expects one JSON object per line:

```json
{"id":"entry-1","title":"Analytics Dashboard","image":"examples/screens/dashboard.png","text":"A product analytics dashboard with KPI cards, trend charts, and conversion breakdowns."}
```

Fields:

- `id`: required unique id
- `title`: optional short label
- `image`: optional local image path for reference only
- `text`: required retrieval text

## Evaluation set format

`rag_eval_set.jsonl` expects one JSON object per line:

```json
{"id":"q01","query":"Where are KPI cards?","expected_retrieval_ids":["analytics-dashboard"],"expected_keywords":["analytics","dashboard"],"image":"examples/test_ui.png"}
```

Fields:

- `id`: required query id
- `query`: required retrieval/generation question
- `expected_retrieval_ids`: required list for retrieval hit/MRR
- `expected_keywords`: required list for keyword-hit checks
- `image`: optional path; required only if `--run-generation`

## Suggested milestone commits

1. Baseline index + baseline evaluation report
2. Prompt-template A/B report (`direct` vs `cited` vs `strict`)
3. Rerank comparison report (`--rerank` on/off)
4. Chunking comparison report (`chunk_size_words`/`chunk_overlap_words` variants)

## Common issues

- Out of memory on 8 GB VRAM:
  - retry with a smaller image
  - use `--resize-max-edge 1024` or lower
  - temporarily run with `--cpu` for a smoke test
- Slow first run:
  - model download is expected on first invocation
- Missing optional embedding deps:
  - indexing falls back to built-in hashed bag-of-words
