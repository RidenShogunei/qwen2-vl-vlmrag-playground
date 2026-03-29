# Qwen2-VL VLM-RAG Playground

A small learning project for running `Qwen/Qwen2-VL-2B-Instruct` in WSL and building a minimal VLM-RAG workflow on top of it.

## What this project includes

- `src/run_qwen2_vl_chat.py`: single-image chat with Qwen2-VL
- `src/index_corpus.py`: build a tiny local text retrieval index for RAG
- `src/query_with_rag.py`: compare plain VLM output vs RAG-augmented output
- `examples/rag_corpus/corpus.jsonl`: starter corpus for the retrieval demo

## Environment notes

This project is designed for the current WSL setup:

- Python 3.10+
- GPU preferred, CPU supported for smoke tests
- First run will download Qwen/Qwen2-VL-2B-Instruct because it is not currently cached
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

Useful flags:

- `--model`: override model repo or local path
- `--max-new-tokens`: cap generation length
- `--cpu`: force CPU if you only want to validate the path
- `--resize-max-edge`: downscale very large images before sending them to the model

### 3. Build a tiny RAG index

```bash
python src/index_corpus.py \
  --corpus examples/rag_corpus/corpus.jsonl \
  --output indexes/rag_corpus_index.json
```

### 4. Query with and without RAG

```bash
python src/query_with_rag.py \
  --image /absolute/path/to/your/image.png \
  --prompt "Which product area does this screen belong to?" \
  --index indexes/rag_corpus_index.json \
  --topk 3
```

## How the minimal VLM-RAG flow works

1. A small corpus of reference descriptions is embedded into vectors.
2. A user question retrieves the top-k most relevant reference entries.
3. Those references are formatted into a short context block.
4. The context, current image, and current question are sent to Qwen2-VL.
5. The script prints both plain VLM output and RAG-augmented output for comparison.

This project intentionally keeps retrieval text-first. That makes the workflow easier to inspect before moving to image retrieval, OCR-heavy pipelines, or agentic orchestration.

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

## Embedding backends

`index_corpus.py` tries these backends in order:

1. `sentence-transformers`, if installed
2. A built-in hashed bag-of-words fallback

The fallback is intentionally simple but keeps the project runnable without a large embedding stack.

## Suggested milestone commits

1. Project skeleton + plain Qwen2-VL chat
2. Retrieval index builder
3. RAG query comparison
4. README cleanup and example expansion

## Common issues

- Out of memory on 8 GB VRAM:
  - retry with a smaller image
  - use `--resize-max-edge 1024` or lower
  - temporarily run with `--cpu` for a smoke test
- Slow first run:
  - the model download is expected on the first invocation
- Missing optional retrieval deps:
  - the project will fall back to the built-in lexical embedding path

