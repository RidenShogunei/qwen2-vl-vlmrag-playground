import argparse
import json

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
    return parser.parse_args()


def _build_and_save(records, output_path, embedding_backend, embedding_model, chunk_size_words, chunk_overlap_words, level):
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
    output = save_index(output_path, payload)
    return {
        "output": str(output),
        "level": level,
        "source_records": len(records),
        "expanded_records": len(expanded_records),
        **embedding_info,
    }


def main() -> int:
    args = parse_args()
    records = load_corpus_jsonl(args.corpus)

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
            )
        )

    if not outputs:
        raise ValueError("Provide at least one output path: --output or --doc-output/--chunk-output")

    print(json.dumps({"indexes": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
