import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_HASH_DIM = 2048
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL_CACHE = {}
_EMBED_MODEL_CACHE = {}
PROMPT_TEMPLATES = {
    "direct": (
        "You are answering a question about the current image.\n"
        "Use retrieval context when relevant, and avoid unsupported claims.\n\n"
        "Retrieval context:\n{context}\n\n"
        "Question: {question}"
    ),
    "cited": (
        "Answer the question about the current image using ONLY the retrieval context below.\n"
        "When using evidence, cite bracket ids like [1] and [2].\n\n"
        "Retrieval context:\n{context}\n\n"
        "Question: {question}"
    ),
    "strict": (
        "You must answer using only the provided retrieval context and the current image.\n"
        "If evidence is insufficient, explicitly say 'I am not sure based on the provided evidence.'\n\n"
        "Retrieval context:\n{context}\n\n"
        "Question: {question}"
    ),
}


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model repo id or local path.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--resize-max-edge",
        type=int,
        default=1280,
        help="Resize the image so the longest edge is at most this value before inference.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")


def choose_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_image(image_path: str, resize_max_edge: int) -> Image.Image:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    if resize_max_edge > 0:
        max_edge = max(image.size)
        if max_edge > resize_max_edge:
            scale = resize_max_edge / max_edge
            resized = (
                max(1, math.floor(image.size[0] * scale)),
                max(1, math.floor(image.size[1] * scale)),
            )
            image = image.resize(resized)
    return image


def _load_model_class():
    if Qwen2VLForConditionalGeneration is not None:
        return Qwen2VLForConditionalGeneration
    if AutoModelForVision2Seq is not None:
        return AutoModelForVision2Seq
    raise RuntimeError("No compatible vision-language model loader found in transformers.")


def load_qwen2_vl(model_name: str, device: str, dtype: torch.dtype):
    cache_key = (model_name, device, str(dtype))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_cls = _load_model_class()
    kwargs = {"trust_remote_code": True}
    if device == "cuda":
        kwargs["torch_dtype"] = dtype
        kwargs["device_map"] = "auto"

    model = model_cls.from_pretrained(model_name, **kwargs)
    if device == "cpu":
        model = model.to(device=device, dtype=dtype)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    _MODEL_CACHE[cache_key] = (model, processor)
    return model, processor


def generate_multimodal_answer(
    image_path: str,
    prompt: str,
    model_name: str,
    max_new_tokens: int,
    resize_max_edge: int,
    force_cpu: bool = False,
):
    device = choose_device(force_cpu=force_cpu)
    dtype = choose_dtype(device)
    image = load_image(image_path=image_path, resize_max_edge=resize_max_edge)

    started_at = time.perf_counter()
    model, processor = load_qwen2_vl(model_name=model_name, device=device, dtype=dtype)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[chat_prompt], images=[image], return_tensors="pt")

    if device == "cuda":
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_token_count = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_token_count:]
    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    duration_s = time.perf_counter() - started_at
    gpu_mem = None
    if device == "cuda":
        current_device = torch.cuda.current_device()
        gpu_mem = torch.cuda.max_memory_allocated(current_device) / (1024 ** 3)

    summary = {
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "duration_s": round(duration_s, 2),
        "image_size": list(image.size),
        "max_memory_allocated_gb": None if gpu_mem is None else round(gpu_mem, 2),
    }
    return answer, summary


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def hashed_bow_embed(texts: Sequence[str], dim: int = DEFAULT_HASH_DIM) -> np.ndarray:
    rows = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        counts = Counter(tokenize_text(text))
        for token, count in counts.items():
            column = hash(token) % dim
            rows[row_idx, column] += float(count)
    return l2_normalize(rows)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_text_embeddings(
    texts: Sequence[str],
    embedding_backend: str = "auto",
    embedding_model: str = DEFAULT_EMBED_MODEL,
) -> Tuple[np.ndarray, Dict[str, str]]:
    if embedding_backend not in {"auto", "sentence-transformers", "hashed-bow"}:
        raise ValueError(f"Unsupported embedding backend: {embedding_backend}")

    if embedding_backend in {"auto", "sentence-transformers"}:
        try:
            from sentence_transformers import SentenceTransformer

            if embedding_model not in _EMBED_MODEL_CACHE:
                _EMBED_MODEL_CACHE[embedding_model] = SentenceTransformer(embedding_model)
            model = _EMBED_MODEL_CACHE[embedding_model]
            vectors = model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            return vectors, {"backend": "sentence-transformers", "embedding_model": embedding_model}
        except Exception as exc:
            if embedding_backend == "sentence-transformers":
                raise RuntimeError(f"Failed to use sentence-transformers backend: {exc}") from exc

    vectors = hashed_bow_embed(texts)
    return vectors, {"backend": "hashed-bow", "embedding_model": "builtin-hashed-bow"}


def load_corpus_jsonl(corpus_path: str) -> List[Dict[str, str]]:
    path = Path(corpus_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")

    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if "id" not in item or "text" not in item:
            raise ValueError(f"Line {line_no} must contain at least 'id' and 'text'.")
        records.append(item)
    if not records:
        raise ValueError(f"No corpus records found in: {path}")
    return records


def split_text_to_chunks(text: str, chunk_size_words: int, chunk_overlap_words: int) -> List[str]:
    words = text.split()
    if chunk_size_words <= 0 or len(words) <= chunk_size_words:
        return [text]
    step = max(1, chunk_size_words - max(0, chunk_overlap_words))
    chunks = []
    for start in range(0, len(words), step):
        part = words[start : start + chunk_size_words]
        if not part:
            break
        chunks.append(" ".join(part))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def expand_records_with_chunking(
    records: Sequence[Dict[str, str]],
    chunk_size_words: int = 0,
    chunk_overlap_words: int = 0,
) -> List[Dict[str, str]]:
    expanded = []
    for record in records:
        chunks = split_text_to_chunks(record["text"], chunk_size_words, chunk_overlap_words)
        if len(chunks) == 1:
            item = dict(record)
            item["parent_id"] = record["id"]
            item["chunk_id"] = f"{record['id']}#0"
            expanded.append(item)
            continue
        for idx, chunk in enumerate(chunks):
            item = dict(record)
            item["text"] = chunk
            item["parent_id"] = record["id"]
            item["chunk_id"] = f"{record['id']}#{idx}"
            expanded.append(item)
    return expanded


def save_index(output_path: str, payload: Dict) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_index(index_path: str) -> Dict:
    path = Path(index_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Index not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_v) / (max_v - min_v)


def _bm25_scores(query_tokens: List[str], entries: Sequence[Dict]) -> np.ndarray:
    n_docs = len(entries)
    if n_docs == 0:
        return np.zeros((0,), dtype=np.float32)
    tokenized_docs = [tokenize_text(item.get("text", "")) for item in entries]
    doc_lens = np.array([max(1, len(toks)) for toks in tokenized_docs], dtype=np.float32)
    avgdl = float(doc_lens.mean()) if n_docs else 1.0

    tf_per_doc = [Counter(toks) for toks in tokenized_docs]
    df = defaultdict(int)
    for toks in tokenized_docs:
        for tok in set(toks):
            df[tok] += 1

    k1 = 1.2
    b = 0.75
    scores = np.zeros((n_docs,), dtype=np.float32)
    for q in query_tokens:
        doc_freq = df.get(q, 0)
        if doc_freq == 0:
            continue
        idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        for i, tf in enumerate(tf_per_doc):
            f = tf.get(q, 0)
            if f == 0:
                continue
            denom = f + k1 * (1.0 - b + b * (doc_lens[i] / avgdl))
            scores[i] += float(idf * ((f * (k1 + 1.0)) / denom))
    return scores


def derive_source_group(source_id: str, mode: str = "auto") -> str:
    if mode not in {"auto", "exact", "prefix_before_img"}:
        raise ValueError(f"Unsupported source group mode: {mode}")
    if mode == "exact":
        return source_id
    if mode == "prefix_before_img":
        return re.sub(r"::img\d+$", "", source_id)
    if "::img" in source_id:
        return re.sub(r"::img\d+$", "", source_id)
    return source_id


def retrieve_topk(
    query: str,
    index_payload: Dict,
    topk: int,
    source_group_key: str = "",
    source_group_mode: str = "auto",
    retrieval_mode: str = "dense",
    hybrid_alpha: float = 0.7,
) -> List[Dict]:
    if retrieval_mode not in {"text", "dense", "hybrid"}:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
    entries = index_payload["entries"]
    backend = index_payload["embedding"]["backend"]
    model_name = index_payload["embedding"]["embedding_model"]
    query_vecs, _ = build_text_embeddings(
        texts=[query],
        embedding_backend=backend,
        embedding_model=model_name if backend == "sentence-transformers" else DEFAULT_EMBED_MODEL,
    )
    query_vec = query_vecs[0]
    vectors = np.array(index_payload["vectors"], dtype=np.float32)
    dense_scores = vectors @ query_vec
    scores = dense_scores.copy()
    if retrieval_mode == "hybrid":
        query_tokens = tokenize_text(query)
        bm25_scores = _bm25_scores(query_tokens, entries)
        dense_norm = _minmax_normalize(dense_scores)
        bm25_norm = _minmax_normalize(bm25_scores)
        alpha = max(0.0, min(1.0, float(hybrid_alpha)))
        scores = alpha * dense_norm + (1.0 - alpha) * bm25_norm

    if source_group_key:
        for idx, entry in enumerate(entries):
            entry_source_id = entry.get("source_id", "")
            if derive_source_group(entry_source_id, mode=source_group_mode) != source_group_key:
                scores[idx] = -1e9

    ranked = np.argsort(-scores)[:topk]
    results = []
    for idx in ranked:
        if float(scores[idx]) <= -1e8:
            continue
        item = dict(entries[idx])
        item["dense_score"] = round(float(dense_scores[idx]), 4)
        item["score"] = round(float(scores[idx]), 4)
        results.append(item)
    return results


def rerank_results_by_overlap(query: str, results: Sequence[Dict]) -> List[Dict]:
    q_tokens = set(tokenize_text(query))
    reranked = []
    for item in results:
        text_tokens = set(tokenize_text(item.get("text", "")))
        overlap = len(q_tokens & text_tokens)
        boosted = dict(item)
        boosted["rerank_overlap"] = overlap
        boosted["rerank_score"] = round(float(item.get("score", 0.0)) + overlap * 0.02, 4)
        reranked.append(boosted)
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


def format_rag_context(results: Sequence[Dict]) -> str:
    if not results:
        return "No retrieval context found."
    lines = []
    for idx, item in enumerate(results, start=1):
        title = item.get("title") or item["id"]
        lines.append(f"[{idx}] {title} (score={item['score']})")
        lines.append(item["text"])
    return "\n".join(lines)


def render_rag_prompt(template_name: str, question: str, context: str) -> str:
    if template_name not in PROMPT_TEMPLATES:
        allowed = ", ".join(sorted(PROMPT_TEMPLATES.keys()))
        raise ValueError(f"Unsupported prompt template '{template_name}'. Allowed: {allowed}")
    return PROMPT_TEMPLATES[template_name].format(question=question, context=context)

