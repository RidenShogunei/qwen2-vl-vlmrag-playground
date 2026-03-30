import argparse
import json
from pathlib import Path

from mm_retrieval_utils import DEFAULT_MM_MODEL, compute_image_embeddings
from vlmrag_utils import (
    build_text_embeddings,
    expand_records_with_chunking,
    load_corpus_jsonl,
    save_index,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build local retrieval index(es) from a JSONL corpus.")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSONL file.")
    parser.add_argument("--output", default=None, help="Single output path (backward compatible).")
    parser.add_argument("--doc-output", default=None, help="Optional doc-level index output path.")
    parser.add_argument("--chunk-output", default=None, help="Optional chunk-level index output path.")
    parser.add_argument(
        "--embedding-backend",
        default="auto",
        choices=["auto", "sentence-transformers", "hashed-bow"],
        help="Embedding backend to use for corpus indexing.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name when using sentence-transformers.",
    )
    parser.add_argument(
        "--chunk-size-words",
        type=int,
        default=0,
        help="If > 0, split each corpus text into word chunks of this size.",
    )
    parser.add_argument(
        "--chunk-overlap-words",
        type=int,
        default=0,
        help="Word overlap between adjacent chunks when chunking is enabled.",
    )
    parser.add_argument("--with-image-embeddings", action="store_true", help="Also build image embeddings for multimodal retrieval.")
    parser.add_argument("--multimodal-model", default=DEFAULT_MM_MODEL, help="Vision-language embedding model for image tower.")
    parser.add_argument("--eval-set-for-images", default=None, help="Optional eval_set.jsonl path to map source_id to image path.")
    parser.add_argument("--mm-cpu", action="store_true", help="Force CPU for multimodal image embedding computation.")
    return parser.parse_args()


def _load_source_image_map(eval_set_path: str):
    if not eval_set_path:
        return {}
    path = Path(eval_set_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"eval_set_for_images not found: {path}")
    mapping = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        src = row.get("source_id")
        img = row.get("image")
        if src and img and src not in mapping:
            mapping[src] = img
    return mapping


def _resolve_image_paths(entries, source_image_map):
    image_paths = []
    missing = []
    for row in entries:
        p = row.get("image")
        if not p:
            p = source_image_map.get(row.get("source_id", ""))
        if not p:
            missing.append(row.get("id", "<unknown>"))
            image_paths.append("")
            continue
        resolved = str(Path(p).expanduser().resolve())
        if not Path(resolved).exists():
            missing.append(row.get("id", "<unknown>"))
            image_paths.append("")
            continue
        image_paths.append(resolved)
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"Missing image path for {len(missing)} entries. Example ids: {sample}")
    return image_paths


def _build_and_save(
    records,
    output_path,
    embedding_backend,
    embedding_model,
    chunk_size_words,
    chunk_overlap_words,
    level,
    with_image_embeddings,
    multimodal_model,
    source_image_map,
    mm_cpu,
):
    expanded_records = expand_records_with_chunking(
        records,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
    )
    texts = [record["text"] for record in expanded_records]
    vectors, embedding_info = build_text_embeddings(
        texts=texts,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )

    payload = {
        "entries": expanded_records,
        "embedding": embedding_info,
        "index_config": {
            "level": level,
            "chunk_size_words": chunk_size_words,
            "chunk_overlap_words": chunk_overlap_words,
            "source_records": len(records),
            "expanded_records": len(expanded_records),
        },
        "vectors": vectors.tolist(),
    }

    if with_image_embeddings:
        image_paths = _resolve_image_paths(expanded_records, source_image_map=source_image_map)
        image_vectors = compute_image_embeddings(
            image_paths=image_paths,
            model_name=multimodal_model,
            force_cpu=mm_cpu,
        )
        payload["image_vectors"] = image_vectors.tolist()
        payload["multimodal"] = {
            "model": multimodal_model,
            "with_image_embeddings": True,
            "image_entries": len(image_paths),
        }

    output = save_index(output_path, payload)
    out = {
        "output": str(output),
        "level": level,
        "source_records": len(records),
        "expanded_records": len(expanded_records),
        **embedding_info,
    }
    if with_image_embeddings:
        out["multimodal_model"] = multimodal_model
        out["image_entries"] = len(expanded_records)
    return out


def main() -> int:
    args = parse_args()
    records = load_corpus_jsonl(args.corpus)
    source_image_map = _load_source_image_map(args.eval_set_for_images)

    outputs = []
    if args.output:
        outputs.append(
            _build_and_save(
                records=records,
                output_path=args.output,
                embedding_backend=args.embedding_backend,
                embedding_model=args.embedding_model,
                chunk_size_words=args.chunk_size_words,
                chunk_overlap_words=args.chunk_overlap_words,
                level="single",
                with_image_embeddings=args.with_image_embeddings,
                multimodal_model=args.multimodal_model,
                source_image_map=source_image_map,
                mm_cpu=args.mm_cpu,
            )
        )

    if args.doc_output:
        outputs.append(
            _build_and_save(
                records=records,
                output_path=args.doc_output,
                embedding_backend=args.embedding_backend,
                embedding_model=args.embedding_model,
                chunk_size_words=0,
                chunk_overlap_words=0,
                level="doc",
                with_image_embeddings=args.with_image_embeddings,
                multimodal_model=args.multimodal_model,
                source_image_map=source_image_map,
                mm_cpu=args.mm_cpu,
            )
        )

    if args.chunk_output:
        outputs.append(
            _build_and_save(
                records=records,
                output_path=args.chunk_output,
                embedding_backend=args.embedding_backend,
                embedding_model=args.embedding_model,
                chunk_size_words=args.chunk_size_words,
                chunk_overlap_words=args.chunk_overlap_words,
                level="chunk",
                with_image_embeddings=args.with_image_embeddings,
                multimodal_model=args.multimodal_model,
                source_image_map=source_image_map,
                mm_cpu=args.mm_cpu,
            )
        )

    if not outputs:
        raise ValueError("Provide at least one output path: --output or --doc-output/--chunk-output")

    print(json.dumps({"indexes": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
