import argparse
import json

from vlmrag_utils import (
    build_text_embeddings,
    expand_records_with_chunking,
    load_corpus_jsonl,
    save_index,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a small local retrieval index from a JSONL corpus.")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the output index JSON.")
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


def main() -> int:
    args = parse_args()
    records = load_corpus_jsonl(args.corpus)
    expanded_records = expand_records_with_chunking(
        records,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
    )
    texts = [record["text"] for record in expanded_records]
    vectors, embedding_info = build_text_embeddings(
        texts=texts,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
    )
    payload = {
        "entries": expanded_records,
        "embedding": embedding_info,
        "index_config": {
            "chunk_size_words": args.chunk_size_words,
            "chunk_overlap_words": args.chunk_overlap_words,
            "source_records": len(records),
            "expanded_records": len(expanded_records),
        },
        "vectors": vectors.tolist(),
    }
    output_path = save_index(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "source_records": len(records),
                "expanded_records": len(expanded_records),
                **embedding_info,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
