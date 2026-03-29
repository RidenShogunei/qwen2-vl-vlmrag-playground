import argparse
import json
import sys

from vlmrag_utils import (
    add_common_model_args,
    format_rag_context,
    generate_multimodal_answer,
    load_index,
    retrieve_topk,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare plain Qwen2-VL output against a minimal RAG-enhanced prompt.")
    parser.add_argument("--image", required=True, help="Absolute or relative path to the current image.")
    parser.add_argument("--prompt", required=True, help="Question asked about the current image.")
    parser.add_argument("--index", required=True, help="Path to the retrieval index JSON.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved entries to include.")
    add_common_model_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        index_payload = load_index(args.index)
        retrieval_results = retrieve_topk(query=args.prompt, index_payload=index_payload, topk=args.topk)
        rag_context = format_rag_context(retrieval_results)

        plain_answer, plain_summary = generate_multimodal_answer(
            image_path=args.image,
            prompt=args.prompt,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            resize_max_edge=args.resize_max_edge,
            force_cpu=args.cpu,
        )

        rag_prompt = (
            "You are answering a question about the current image.\n"
            "Use the retrieval context below when it is relevant, but do not invent facts that are not supported.\n\n"
            f"Retrieval context:\n{rag_context}\n\n"
            f"Question: {args.prompt}"
        )
        rag_answer, rag_summary = generate_multimodal_answer(
            image_path=args.image,
            prompt=rag_prompt,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            resize_max_edge=args.resize_max_edge,
            force_cpu=args.cpu,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=== Retrieved Context ===")
    print(rag_context)
    print("\n=== Plain VLM Answer ===")
    print(plain_answer)
    print("\n=== RAG-Enhanced Answer ===")
    print(rag_answer)
    print("\n=== Run Summary ===")
    print(
        json.dumps(
            {
                "plain": plain_summary,
                "rag": rag_summary,
                "topk": args.topk,
                "retrieved_ids": [item["id"] for item in retrieval_results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
