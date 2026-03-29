import argparse
import json
import sys

from vlmrag_utils import add_common_model_args, generate_multimodal_answer


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-image chat with Qwen2-VL.")
    parser.add_argument("--image", required=True, help="Absolute or relative path to the input image.")
    parser.add_argument("--prompt", required=True, help="Question or instruction for the model.")
    add_common_model_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        answer, summary = generate_multimodal_answer(
            image_path=args.image,
            prompt=args.prompt,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            resize_max_edge=args.resize_max_edge,
            force_cpu=args.cpu,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: if the model download is slow or VRAM is tight, retry with --resize-max-edge 1024 or --cpu.",
            file=sys.stderr,
        )
        return 1

    print("=== Answer ===")
    print(answer)
    print("\n=== Run Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
