import argparse
import itertools
import json
import subprocess
from pathlib import Path


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
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = [int(x.strip()) for x in args.retrieval_k.split(",") if x.strip()]
    templates = ["direct", "cited", "strict"]
    rerank_opts = [False, True]

    summaries = []
    for k, template, rerank in itertools.product(k_values, templates, rerank_opts):
        config_name = f"k{k}_{template}_{'rerank' if rerank else 'norank'}"
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
        ]
        if rerank:
            cmd.append("--rerank")
        if args.run_generation:
            cmd.append("--run-generation")
        if args.cpu:
            cmd.append("--cpu")

        subprocess.run(cmd, check=True)
        report = json.loads(json_path.read_text(encoding="utf-8"))
        row = {
            "config": config_name,
            "retrieval_k": k,
            "template": template,
            "rerank": rerank,
            **report["metrics"],
            "report": str(json_path),
        }
        summaries.append(row)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"runs": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path.resolve()), "runs": len(summaries)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
