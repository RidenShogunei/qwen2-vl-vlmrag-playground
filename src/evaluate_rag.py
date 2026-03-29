import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

from vlmrag_utils import (
    PROMPT_TEMPLATES,
    format_rag_context,
    generate_multimodal_answer,
    load_index,
    render_rag_prompt,
    rerank_results_by_overlap,
    retrieve_topk,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and generation quality for the local RAG setup.")
    parser.add_argument("--eval-set", required=True, help="Path to evaluation JSONL.")
    parser.add_argument("--index", required=True, help="Path to retrieval index JSON.")
    parser.add_argument("--retrieval-k", type=int, default=3, help="Top-k retrieval size.")
    parser.add_argument(
        "--prompt-template",
        default="direct",
        choices=sorted(PROMPT_TEMPLATES.keys()),
        help="Prompt template used for RAG generation.",
    )
    parser.add_argument("--rerank", action="store_true", help="Enable lexical overlap reranking.")
    parser.add_argument("--run-generation", action="store_true", help="Run plain vs RAG generation checks.")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct", help="Model repo id or local path.")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Max new tokens for generation checks.")
    parser.add_argument("--resize-max-edge", type=int, default=1024, help="Image resize max edge for generation checks.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for generation checks.")
    parser.add_argument("--output-json", default=None, help="Where to save JSON report.")
    parser.add_argument("--output-csv", default=None, help="Where to save per-sample CSV report.")
    return parser.parse_args()


def load_eval_set(path: str):
    items = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        required = ["id", "query", "expected_retrieval_ids", "expected_keywords"]
        missing = [k for k in required if k not in item]
        if missing:
            raise ValueError(f"Eval line {line_no} missing required fields: {missing}")
        items.append(item)
    if not items:
        raise ValueError("Evaluation set is empty.")
    return items


def contains_keywords(answer: str, keywords):
    text = answer.lower()
    hits = [k for k in keywords if k.lower() in text]
    return len(hits) > 0, hits


def compute_mrr(found_ids, expected_ids):
    expected = set(expected_ids)
    for rank, item_id in enumerate(found_ids, start=1):
        if item_id in expected:
            return 1.0 / rank
    return 0.0


def git_sha_or_none():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def main():
    args = parse_args()
    index_payload = load_index(args.index)
    eval_items = load_eval_set(args.eval_set)

    rows = []
    retrieval_hits = 0
    retrieval_mrr_sum = 0.0
    plain_keyword_hits = 0
    rag_keyword_hits = 0

    for item in eval_items:
        query = item["query"]
        retrieved = retrieve_topk(query=query, index_payload=index_payload, topk=args.retrieval_k)
        if args.rerank:
            retrieved = rerank_results_by_overlap(query, retrieved)

        retrieved_ids = [r["id"] for r in retrieved]
        expected_ids = item["expected_retrieval_ids"]
        hit = any(rid in set(expected_ids) for rid in retrieved_ids)
        mrr = compute_mrr(retrieved_ids, expected_ids)
        retrieval_hits += int(hit)
        retrieval_mrr_sum += mrr

        row = {
            "id": item["id"],
            "query": query,
            "expected_retrieval_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
            "retrieval_hit": hit,
            "retrieval_mrr": round(mrr, 4),
            "plain_keyword_hit": None,
            "rag_keyword_hit": None,
            "plain_answer": None,
            "rag_answer": None,
        }

        if args.run_generation:
            image_path = item.get("image")
            if image_path:
                plain_answer, _ = generate_multimodal_answer(
                    image_path=image_path,
                    prompt=query,
                    model_name=args.model,
                    max_new_tokens=args.max_new_tokens,
                    resize_max_edge=args.resize_max_edge,
                    force_cpu=args.cpu,
                )
                rag_context = format_rag_context(retrieved)
                rag_prompt = render_rag_prompt(args.prompt_template, question=query, context=rag_context)
                rag_answer, _ = generate_multimodal_answer(
                    image_path=image_path,
                    prompt=rag_prompt,
                    model_name=args.model,
                    max_new_tokens=args.max_new_tokens,
                    resize_max_edge=args.resize_max_edge,
                    force_cpu=args.cpu,
                )
                plain_ok, _ = contains_keywords(plain_answer, item["expected_keywords"])
                rag_ok, _ = contains_keywords(rag_answer, item["expected_keywords"])
                plain_keyword_hits += int(plain_ok)
                rag_keyword_hits += int(rag_ok)
                row["plain_keyword_hit"] = plain_ok
                row["rag_keyword_hit"] = rag_ok
                row["plain_answer"] = plain_answer
                row["rag_answer"] = rag_answer

        rows.append(row)

    total = len(rows)
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": git_sha_or_none(),
        "config": {
            "eval_set": args.eval_set,
            "index": args.index,
            "retrieval_k": args.retrieval_k,
            "prompt_template": args.prompt_template,
            "rerank": args.rerank,
            "run_generation": args.run_generation,
            "model": args.model,
        },
        "metrics": {
            "samples": total,
            "retrieval_hit_rate": round(retrieval_hits / total, 4),
            "retrieval_mrr": round(retrieval_mrr_sum / total, 4),
            "plain_keyword_hit_rate": None if not args.run_generation else round(plain_keyword_hits / total, 4),
            "rag_keyword_hit_rate": None if not args.run_generation else round(rag_keyword_hits / total, 4),
        },
        "rows": rows,
        "failure_examples": {
            "retrieval_miss": [r for r in rows if not r["retrieval_hit"]][:3],
            "generation_not_improved": [
                r
                for r in rows
                if args.run_generation and r["plain_keyword_hit"] and not r["rag_keyword_hit"]
            ][:3],
        },
    }

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON report to: {args.output_json}")

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "query",
                    "expected_retrieval_ids",
                    "retrieved_ids",
                    "retrieval_hit",
                    "retrieval_mrr",
                    "plain_keyword_hit",
                    "rag_keyword_hit",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "id": row["id"],
                        "query": row["query"],
                        "expected_retrieval_ids": "|".join(row["expected_retrieval_ids"]),
                        "retrieved_ids": "|".join(row["retrieved_ids"]),
                        "retrieval_hit": row["retrieval_hit"],
                        "retrieval_mrr": row["retrieval_mrr"],
                        "plain_keyword_hit": row["plain_keyword_hit"],
                        "rag_keyword_hit": row["rag_keyword_hit"],
                    }
                )
        print(f"Saved CSV report to: {args.output_csv}")


if __name__ == "__main__":
    main()
