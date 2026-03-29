import argparse
import json
import sys
from datetime import datetime

from vlmrag_utils import (
    PROMPT_TEMPLATES,
    add_common_model_args,
    format_rag_context,
    generate_multimodal_answer,
    load_index,
    render_rag_prompt,
    rerank_results_by_overlap,
    retrieve_topk,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare plain Qwen2-VL output against a minimal RAG-enhanced prompt.")
    parser.add_argument("--image", required=True, help="Absolute or relative path to the current image.")
    parser.add_argument("--prompt", required=True, help="Question asked about the current image.")
    parser.add_argument("--index", required=True, help="Path to the retrieval index JSON.")
    parser.add_argument("--topk", type=int, default=3, help="Backward-compatible alias for retrieval-k.")
    parser.add_argument("--retrieval-k", type=int, default=None, help="Number of retrieved entries to include.")
    parser.add_argument(
        "--prompt-template",
        default="direct",
        choices=sorted(PROMPT_TEMPLATES.keys()),
        help="Prompt template strategy for RAG answer generation.",
    )
    parser.add_argument("--rerank", action="store_true", help="Apply lightweight lexical overlap reranking.")
    parser.add_argument("--show-evidence", action="store_true", help="Print retrieval context and evidence IDs.")
    parser.add_argument("--json-output", default=None, help="Optional path to save structured run output JSON.")
    add_common_model_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    retrieval_k = args.retrieval_k if args.retrieval_k is not None else args.topk
    try:
        index_payload = load_index(args.index)
        retrieval_results = retrieve_topk(query=args.prompt, index_payload=index_payload, topk=retrieval_k)
        if args.rerank:
            retrieval_results = rerank_results_by_overlap(args.prompt, retrieval_results)
        rag_context = format_rag_context(retrieval_results)

        plain_answer, plain_summary = generate_multimodal_answer(
            image_path=args.image,
            prompt=args.prompt,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            resize_max_edge=args.resize_max_edge,
            force_cpu=args.cpu,
        )

        rag_prompt = render_rag_prompt(args.prompt_template, question=args.prompt, context=rag_context)
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

    if args.show_evidence:
        print("=== Retrieved Context ===")
        print(rag_context)

    print("=== Plain VLM Answer ===")
    print(plain_answer)
    print("\n=== RAG-Enhanced Answer ===")
    print(rag_answer)
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "prompt_template": args.prompt_template,
            "retrieval_k": retrieval_k,
            "rerank": args.rerank,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "resize_max_edge": args.resize_max_edge,
        },
        "retrieval": retrieval_results,
        "plain": {"answer": plain_answer, "summary": plain_summary},
        "rag": {"answer": rag_answer, "summary": rag_summary},
    }

    print("\n=== Run Summary ===")
    print(
        json.dumps(
            {
                "plain": plain_summary,
                "rag": rag_summary,
                "retrieval_k": retrieval_k,
                "retrieved_ids": [item["id"] for item in retrieval_results],
                "prompt_template": args.prompt_template,
                "rerank": args.rerank,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON output to: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
