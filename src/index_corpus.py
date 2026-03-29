import argparse
import json

from vlmrag_utils import build_text_embeddings, load_corpus_jsonl, save_index


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_corpus_jsonl(args.corpus)
    texts = [record["text"] for record in records]
    vectors, embedding_info = build_text_embeddings(
        texts=texts,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
    )
    payload = {
        "entries": records,
        "embedding": embedding_info,
        "vectors": vectors.tolist(),
    }
    output_path = save_index(args.output, payload)
    print(json.dumps({"output": str(output_path), "entries": len(records), **embedding_info}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
