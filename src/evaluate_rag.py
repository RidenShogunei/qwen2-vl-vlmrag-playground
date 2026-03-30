import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from mm_retrieval_utils import DEFAULT_MM_MODEL, retrieve_topk_multimodal
from vlmrag_utils import (
    PROMPT_TEMPLATES,
    derive_source_group,
    format_rag_context,
    generate_multimodal_answer,
    load_index,
    render_rag_prompt,
    rerank_results_by_overlap,
    retrieve_topk,
)


_WORD_RE = re.compile(r"[a-z0-9]+")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and generation quality for local RAG.")
    parser.add_argument("--eval-set", required=True, help="Path to evaluation JSONL")
    parser.add_argument("--index", required=True, help="Path to retrieval index JSON")
    parser.add_argument("--retrieval-k", type=int, default=3)
    parser.add_argument("--prompt-template", default="direct", choices=sorted(PROMPT_TEMPLATES.keys()))
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-pool-size", type=int, default=None, help="Retrieve this many candidates before rerank; defaults to retrieval-k.")
    parser.add_argument("--restrict-source-group", action="store_true", help="Restrict retrieval candidates to the same source group as each sample.")
    parser.add_argument("--source-group-mode", default="auto", choices=["auto", "exact", "prefix_before_img"], help="How source_id is mapped to source group.")
    parser.add_argument("--retrieval-mode", default="hybrid", choices=["text", "dense", "hybrid", "multimodal"], help="Retrieval scoring mode.")
    parser.add_argument("--hybrid-alpha", type=float, default=0.7, help="Dense score weight in hybrid mode (0~1).")
    parser.add_argument("--multimodal-model", default=DEFAULT_MM_MODEL, help="Dual-tower multimodal retrieval model.")
    parser.add_argument("--image-alpha", type=float, default=0.5, help="Image score weight in multimodal fusion.")
    parser.add_argument("--text-alpha", type=float, default=0.5, help="Text score weight in multimodal fusion.")
    parser.add_argument("--mm-cpu", action="store_true", help="Force CPU for multimodal retrieval tower.")
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--resize-max-edge", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--failure-json", default=None, help="Optional path for failure cases JSON")
    parser.add_argument("--data-version", default="unknown")
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick smoke evaluation.")
    parser.add_argument(
        "--allow-diagnostic-group-restriction",
        action="store_true",
        help="Allow --restrict-source-group for diagnostic runs. Main benchmark runs should keep this off.",
    )
    return parser.parse_args()


def load_eval_set(path: str):
    rows = []
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = ["id", "query", "source_id"]
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"Eval row {line_no} missing required fields: {missing}")
        if "expected_retrieval_ids" not in row:
            row["expected_retrieval_ids"] = [row["source_id"]]
        if "answers" not in row:
            kws = row.get("expected_keywords", [])
            row["answers"] = kws if isinstance(kws, list) and kws else [row["query"]]
        rows.append(row)
    if not rows:
        raise ValueError("Evaluation set is empty")
    return rows


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    tokens = _WORD_RE.findall(text)
    return " ".join(tokens)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = defaultdict(int)
    gold_counts = defaultdict(int)
    for t in pred_tokens:
        pred_counts[t] += 1
    for t in gold_tokens:
        gold_counts[t] += 1
    common = 0
    for token, c in pred_counts.items():
        common += min(c, gold_counts.get(token, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_em_f1(pred: str, answers: Sequence[str]) -> Tuple[float, float, str]:
    best = (0.0, 0.0, "")
    for ans in answers:
        em = exact_match(pred, ans)
        f1 = token_f1(pred, ans)
        if (em, f1) > (best[0], best[1]):
            best = (em, f1, ans)
    return best


def compute_mrr(found_ids: Sequence[str], expected_ids: Sequence[str]) -> float:
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


def get_field_groups(row: Dict) -> List[str]:
    groups = []
    meta = row.get("meta")
    if isinstance(meta, dict):
        if meta.get("benchmark"):
            groups.append(f"benchmark:{meta['benchmark']}")
        if meta.get("split"):
            groups.append(f"split:{meta['split']}")
    if not groups:
        groups.append("benchmark:unknown")
    return groups


def _effective_retrieval_mode(mode: str) -> str:
    return "text" if mode == "dense" else mode


def main():
    args = parse_args()
    if args.restrict_source_group and not args.allow_diagnostic_group_restriction:
        raise ValueError(
            "restrict_source_group is disabled for comparable benchmark runs. "
            "Use --allow-diagnostic-group-restriction only for diagnostic analysis."
        )
    mode = _effective_retrieval_mode(args.retrieval_mode)

    index_payload = load_index(args.index)
    eval_rows = load_eval_set(args.eval_set)
    if args.max_samples is not None:
        eval_rows = eval_rows[: max(0, args.max_samples)]

    result_rows = []
    retrieval_hits = 0
    retrieval_mrr_sum = 0.0
    plain_em_sum = 0.0
    plain_f1_sum = 0.0
    rag_em_sum = 0.0
    rag_f1_sum = 0.0

    group_stats = defaultdict(lambda: {"samples": 0, "retrieval_hit": 0, "retrieval_mrr": 0.0, "plain_em": 0.0, "plain_f1": 0.0, "rag_em": 0.0, "rag_f1": 0.0})

    for row in eval_rows:
        query = row["query"]
        expected_ids = row.get("expected_retrieval_ids", [row["source_id"]])
        retrieval_pool = args.retrieval_k if args.rerank_pool_size is None else max(args.retrieval_k, args.rerank_pool_size)

        if mode == "multimodal":
            retrieved = retrieve_topk_multimodal(
                query=query,
                query_image_path=row.get("image", ""),
                index_payload=index_payload,
                topk=retrieval_pool,
                model_name=args.multimodal_model,
                image_alpha=args.image_alpha,
                text_alpha=args.text_alpha,
                force_cpu=args.mm_cpu,
            )
        else:
            source_group_key = ""
            if args.restrict_source_group:
                source_group_key = derive_source_group(row["source_id"], mode=args.source_group_mode)
            retrieved = retrieve_topk(
                query=query,
                index_payload=index_payload,
                topk=retrieval_pool,
                source_group_key=source_group_key,
                source_group_mode=args.source_group_mode,
                retrieval_mode=mode,
                hybrid_alpha=args.hybrid_alpha,
            )

        if args.rerank:
            retrieved = rerank_results_by_overlap(query, retrieved)
        retrieved = retrieved[: args.retrieval_k]

        retrieved_ids = [x["source_id"] if x.get("source_id") else x["id"] for x in retrieved]
        hit = any(item_id in set(expected_ids) for item_id in retrieved_ids)
        mrr = compute_mrr(retrieved_ids, expected_ids)
        retrieval_hits += int(hit)
        retrieval_mrr_sum += mrr

        out = {
            "id": row["id"],
            "query": query,
            "source_id": row["source_id"],
            "expected_retrieval_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
            "retrieval_hit": bool(hit),
            "retrieval_mrr": round(mrr, 4),
            "plain_answer": None,
            "rag_answer": None,
            "plain_em": None,
            "plain_f1": None,
            "rag_em": None,
            "rag_f1": None,
            "best_answer_plain": None,
            "best_answer_rag": None,
            "retrieval": retrieved,
            "meta": row.get("meta", {}),
        }

        if args.run_generation and row.get("image"):
            plain_answer, _ = generate_multimodal_answer(
                image_path=row["image"],
                prompt=query,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                resize_max_edge=args.resize_max_edge,
                force_cpu=args.cpu,
            )
            rag_prompt = render_rag_prompt(args.prompt_template, question=query, context=format_rag_context(retrieved))
            rag_answer, _ = generate_multimodal_answer(
                image_path=row["image"],
                prompt=rag_prompt,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                resize_max_edge=args.resize_max_edge,
                force_cpu=args.cpu,
            )

            p_em, p_f1, p_best = best_em_f1(plain_answer, row["answers"])
            r_em, r_f1, r_best = best_em_f1(rag_answer, row["answers"])
            plain_em_sum += p_em
            plain_f1_sum += p_f1
            rag_em_sum += r_em
            rag_f1_sum += r_f1

            out["plain_answer"] = plain_answer
            out["rag_answer"] = rag_answer
            out["plain_em"] = round(p_em, 4)
            out["plain_f1"] = round(p_f1, 4)
            out["rag_em"] = round(r_em, 4)
            out["rag_f1"] = round(r_f1, 4)
            out["best_answer_plain"] = p_best
            out["best_answer_rag"] = r_best

        result_rows.append(out)

        for group in get_field_groups(row):
            group_stats[group]["samples"] += 1
            group_stats[group]["retrieval_hit"] += int(hit)
            group_stats[group]["retrieval_mrr"] += mrr
            if args.run_generation:
                group_stats[group]["plain_em"] += out["plain_em"] or 0.0
                group_stats[group]["plain_f1"] += out["plain_f1"] or 0.0
                group_stats[group]["rag_em"] += out["rag_em"] or 0.0
                group_stats[group]["rag_f1"] += out["rag_f1"] or 0.0

    total = len(result_rows)
    metrics = {
        "samples": total,
        "hit_rate_at_k": round(retrieval_hits / total, 4),
        "retrieval_mrr": round(retrieval_mrr_sum / total, 4),
        "plain_em": None,
        "plain_f1": None,
        "rag_em": None,
        "rag_f1": None,
    }
    if args.run_generation:
        metrics["plain_em"] = round(plain_em_sum / total, 4)
        metrics["plain_f1"] = round(plain_f1_sum / total, 4)
        metrics["rag_em"] = round(rag_em_sum / total, 4)
        metrics["rag_f1"] = round(rag_f1_sum / total, 4)

    groups = {}
    for key, val in group_stats.items():
        n = max(1, val["samples"])
        groups[key] = {
            "samples": val["samples"],
            "hit_rate_at_k": round(val["retrieval_hit"] / n, 4),
            "retrieval_mrr": round(val["retrieval_mrr"] / n, 4),
            "plain_em": None if not args.run_generation else round(val["plain_em"] / n, 4),
            "plain_f1": None if not args.run_generation else round(val["plain_f1"] / n, 4),
            "rag_em": None if not args.run_generation else round(val["rag_em"] / n, 4),
            "rag_f1": None if not args.run_generation else round(val["rag_f1"] / n, 4),
        }

    failure_examples = {
        "retrieval_miss": [r for r in result_rows if not r["retrieval_hit"]][:3],
        "generation_regression": [
            r
            for r in result_rows
            if args.run_generation and (r["rag_f1"] or 0.0) < (r["plain_f1"] or 0.0)
        ][:3],
        "generation_improvement": [
            r
            for r in result_rows
            if args.run_generation and (r["rag_f1"] or 0.0) > (r["plain_f1"] or 0.0)
        ][:3],
    }

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": git_sha_or_none(),
        "data_version": args.data_version,
        "config_signature": {
            "config_name": args.config_name,
            "index": args.index,
            "retrieval_k": args.retrieval_k,
            "prompt_template": args.prompt_template,
            "rerank": args.rerank,
            "rerank_pool_size": args.rerank_pool_size,
            "restrict_source_group": args.restrict_source_group,
            "source_group_mode": args.source_group_mode,
            "retrieval_mode": mode,
            "hybrid_alpha": args.hybrid_alpha,
            "multimodal_model": args.multimodal_model,
            "image_alpha": args.image_alpha,
            "text_alpha": args.text_alpha,
            "mm_cpu": args.mm_cpu,
            "run_generation": args.run_generation,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "resize_max_edge": args.resize_max_edge,
            "cpu": args.cpu,
            "max_samples": args.max_samples,
            "allow_diagnostic_group_restriction": args.allow_diagnostic_group_restriction,
        },
        "metrics": metrics,
        "group_metrics": groups,
        "rows": result_rows,
        "failure_examples": failure_examples,
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON report to: {out}")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "query",
                    "source_id",
                    "expected_retrieval_ids",
                    "retrieved_ids",
                    "retrieval_hit",
                    "retrieval_mrr",
                    "plain_em",
                    "plain_f1",
                    "rag_em",
                    "rag_f1",
                ],
            )
            writer.writeheader()
            for row in result_rows:
                writer.writerow(
                    {
                        "id": row["id"],
                        "query": row["query"],
                        "source_id": row["source_id"],
                        "expected_retrieval_ids": "|".join(row["expected_retrieval_ids"]),
                        "retrieved_ids": "|".join(row["retrieved_ids"]),
                        "retrieval_hit": row["retrieval_hit"],
                        "retrieval_mrr": row["retrieval_mrr"],
                        "plain_em": row["plain_em"],
                        "plain_f1": row["plain_f1"],
                        "rag_em": row["rag_em"],
                        "rag_f1": row["rag_f1"],
                    }
                )
        print(f"Saved CSV report to: {out}")

    if args.failure_json:
        out = Path(args.failure_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(failure_examples, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved failure cases to: {out}")


if __name__ == "__main__":
    main()
