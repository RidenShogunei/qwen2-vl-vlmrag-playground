import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


DEFAULT_MM_MODEL = "google/siglip-base-patch16-224"
_MM_MODEL_CACHE: Dict[Tuple[str, str], Tuple[AutoModel, AutoProcessor]] = {}


def _choose_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _choose_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _load_model(model_name: str, force_cpu: bool = False):
    device = _choose_device(force_cpu=force_cpu)
    cache_key = (model_name, device)
    if cache_key in _MM_MODEL_CACHE:
        return _MM_MODEL_CACHE[cache_key], device

    dtype = _choose_dtype(device)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    _MM_MODEL_CACHE[cache_key] = (model, processor)
    return (model, processor), device



def _feature_to_numpy(output) -> np.ndarray:
    if isinstance(output, torch.Tensor):
        tensor = output
    elif hasattr(output, "image_embeds") and output.image_embeds is not None:
        tensor = output.image_embeds
    elif hasattr(output, "text_embeds") and output.text_embeds is not None:
        tensor = output.text_embeds
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        tensor = output.pooler_output
    elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        tensor = output.last_hidden_state[:, 0, :]
    else:
        raise ValueError(f"Unsupported feature output type: {type(output)}")
    return tensor.detach().float().cpu().numpy().astype(np.float32)

def _open_image(path: str, resize_max_edge: int = 768) -> Image.Image:
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


def compute_image_embeddings(
    image_paths: Sequence[str],
    model_name: str = DEFAULT_MM_MODEL,
    force_cpu: bool = False,
    resize_max_edge: int = 768,
) -> np.ndarray:
    (model, processor), device = _load_model(model_name=model_name, force_cpu=force_cpu)
    vectors: List[np.ndarray] = []
    with torch.inference_mode():
        for path in image_paths:
            image = _open_image(path, resize_max_edge=resize_max_edge)
            inputs = processor(images=[image], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            image_features = model.get_image_features(**inputs)
            vec = _feature_to_numpy(image_features)[0]
            vectors.append(vec.astype(np.float32))
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return _l2_normalize(np.vstack(vectors))


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_v) / (max_v - min_v)


def retrieve_topk_multimodal(
    query: str,
    query_image_path: str,
    index_payload: Dict,
    topk: int,
    model_name: str = DEFAULT_MM_MODEL,
    image_alpha: float = 0.5,
    text_alpha: float = 0.5,
    force_cpu: bool = False,
) -> List[Dict]:
    if "image_vectors" not in index_payload:
        raise ValueError("Index missing image_vectors. Rebuild index with --with-image-embeddings.")
    if not query_image_path:
        raise ValueError("Multimodal retrieval requires query image path.")

    entries = index_payload["entries"]
    text_vectors = np.array(index_payload["vectors"], dtype=np.float32)
    image_vectors = np.array(index_payload["image_vectors"], dtype=np.float32)
    if text_vectors.shape[0] != image_vectors.shape[0]:
        raise ValueError("Index vectors/image_vectors length mismatch.")

    from vlmrag_utils import build_text_embeddings

    embed_info = index_payload["embedding"]
    text_backend = embed_info["backend"]
    text_model = embed_info["embedding_model"]
    query_text_vec, _ = build_text_embeddings(
        texts=[query],
        embedding_backend=text_backend,
        embedding_model=text_model if text_backend == "sentence-transformers" else text_model,
    )
    text_scores = text_vectors @ query_text_vec[0]

    query_img_vec = compute_image_embeddings(
        image_paths=[query_image_path],
        model_name=model_name,
        force_cpu=force_cpu,
    )[0]
    image_scores = image_vectors @ query_img_vec

    text_norm = minmax_normalize(text_scores)
    image_norm = minmax_normalize(image_scores)
    fusion = float(text_alpha) * text_norm + float(image_alpha) * image_norm

    ranked = np.argsort(-fusion)[:topk]
    rows: List[Dict] = []
    for idx in ranked:
        item = dict(entries[idx])
        item["score"] = round(float(fusion[idx]), 4)
        item["text_score"] = round(float(text_scores[idx]), 4)
        item["image_score"] = round(float(image_scores[idx]), 4)
        rows.append(item)
    return rows


