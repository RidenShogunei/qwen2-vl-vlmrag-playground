import argparse
import csv
import itertools
import json
import subprocess
from pathlib import Path

from mm_retrieval_utils import DEFAULT_MM_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark grid and produce aggregated summary.")
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", default="reports/benchmark_grid")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--data-version", default="docvqa_infographicvqa_val_v1")
    parser.add_argument("--retrieval-k", default="1,3,5", help="Comma separated retrieval-k list")

    parser.add_argument("--retrieval-mode", default="hybrid", choices=["text", "hybrid", "multimodal"], help="Grid retrieval mode")
    parser.add_argument("--hybrid-alpha", default="0.3,0.5,0.7", help="Comma separated hybrid alpha list")

    parser.add_argument("--multimodal", action="store_true", help="Shortcut for --retrieval-mode multimodal")
    parser.add_argument("--multimodal-model", default=DEFAULT_MM_MODEL)
    parser.add_argument("--image-alpha-grid", default="0.3,0.5,0.7", help="Comma separated image-alpha list for multimodal")
    parser.add_argument("--text-alpha", type=float, default=0.5)
    parser.add_argument("--mm-cpu", action="store_true")

    parser.add_argument("--smoke-max-samples", type=int, default=None, help="Optional max samples for smoke run")
    parser.add_argument("--rerank-pool-size", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "multimodal" if args.multimodal else args.retrieval_mode
    k_values = [int(x.strip()) for x in args.retrieval_k.split(",") if x.strip()]
    templates = ["direct", "cited", "strict"]
    rerank_opts = [False, True]

    if mode == "multimodal":
        grid_values = [float(x.strip()) for x in args.image_alpha_grid.split(",") if x.strip()]
    elif mode == "hybrid":
        grid_values = [float(x.strip()) for x in args.hybrid_alpha.split(",") if x.strip()]
    else:
        grid_values = [0.0]

    summaries = []
    all_failures = []

    for k, g, template, rerank in itertools.product(k_values, grid_values, templates, rerank_opts):
        tag = f"a{g}" if mode in {"hybrid", "multimodal"} else "text"
        config_name = f"k{k}_{tag}_{template}_{'rerank' if rerank else 'norank'}"
        json_path = output_dir / f"{config_name}.json"
        csv_path = output_dir / f"{config_name}.csv"
        fail_path = output_dir / f"{config_name}_failures.json"

        cmd = [
            "python",
            "src/evaluate_rag.py",
            "--eval-set",
            args.eval_set,
            "--index",
            args.index,
            "--retrieval-k",
            str(k),
            "--prompt-template",
            template,
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
            "--failure-json",
            str(fail_path),
            "--model",
            args.model,
            "--data-version",
            args.data_version,
            "--config-name",
            config_name,
            "--retrieval-mode",
            mode,
        ]

        if mode == "hybrid":
            cmd.extend(["--hybrid-alpha", str(g)])
        elif mode == "multimodal":
            cmd.extend([
                "--multimodal-model",
                args.multimodal_model,
                "--image-alpha",
                str(g),
                "--text-alpha",
                str(args.text_alpha),
            ])
            if args.mm_cpu:
                cmd.append("--mm-cpu")

        if rerank:
            cmd.extend(["--rerank", "--rerank-pool-size", str(args.rerank_pool_size)])
        if args.run_generation:
            cmd.append("--run-generation")
        if args.cpu:
            cmd.append("--cpu")
        if args.smoke_max_samples is not None:
            cmd.extend(["--max-samples", str(args.smoke_max_samples)])

        subprocess.run(cmd, check=True)
        report = json.loads(json_path.read_text(encoding="utf-8"))
        row = {
            "config": config_name,
            "retrieval_mode": mode,
            "retrieval_k": k,
            "grid_value": g,
            "template": template,
            "rerank": rerank,
            **report["metrics"],
            "report": str(json_path),
        }
        summaries.append(row)

        failures = json.loads(fail_path.read_text(encoding="utf-8"))
        all_failures.append({"config": config_name, **failures})

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"runs": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")

    rows_path = output_dir / "rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "retrieval_mode",
                "retrieval_k",
                "grid_value",
                "template",
                "rerank",
                "samples",
                "hit_rate_at_k",
                "retrieval_mrr",
                "plain_em",
                "plain_f1",
                "rag_em",
                "rag_f1",
                "report",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    failures_path = output_dir / "failure_cases.json"
    failures_path.write_text(json.dumps({"runs": all_failures}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary": str(summary_path.resolve()),
                "rows": str(rows_path.resolve()),
                "failure_cases": str(failures_path.resolve()),
                "runs": len(summaries),
                "retrieval_mode": mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
