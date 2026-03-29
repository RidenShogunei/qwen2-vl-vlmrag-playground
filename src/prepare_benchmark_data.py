import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from datasets import get_dataset_split_names, load_dataset


DATASET_CANDIDATES = {
    "docvqa": ["nielsr/docvqa_1200_examples"],
    "infographicvqa": ["ayoubkirouane/infographic-VQA", "Minchael/infographicVQA_temp"],
}


def _safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _pick_first(row: Dict, keys: Sequence[str]):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _answers_from_row(row: Dict) -> List[str]:
    value = _pick_first(row, ["answers", "answer", "gold_answers", "label"])
    if value is None:
        return []
    if isinstance(value, list):
        out = [_safe_text(v) for v in value if _safe_text(v)]
        return out
    if isinstance(value, dict):
        out = []
        for v in value.values():
            t = _safe_text(v)
            if t:
                out.append(t)
        return out
    single = _safe_text(value)
    return [single] if single else []


def _text_from_row(row: Dict, fallback_question: str) -> str:
    parts: List[str] = []
    for key in ["ucsf_text", "ocr_text", "context", "document_text", "passage"]:
        value = _safe_text(row.get(key))
        if value:
            parts.append(value)
    for key in ["words", "ocr_tokens", "tokens"]:
        value = row.get(key)
        if isinstance(value, list):
            joined = " ".join([_safe_text(v) for v in value if _safe_text(v)])
            if joined:
                parts.append(joined)
    text = "\n".join(parts).strip()
    return text if text else fallback_question


def _select_dataset_name(kind: str) -> str:
    for name in DATASET_CANDIDATES[kind]:
        try:
            _ = get_dataset_split_names(name)
            return name
        except Exception:
            continue
    raise RuntimeError(f"No accessible dataset found for kind={kind}. Candidates={DATASET_CANDIDATES[kind]}")


def _select_split(dataset_name: str, requested_split: str) -> str:
    available = get_dataset_split_names(dataset_name)
    preferred = [requested_split]
    if requested_split == "validation":
        preferred.extend(["val", "test", "train"])
    elif requested_split == "test":
        preferred.extend(["validation", "val", "train"])
    else:
        preferred.extend(["validation", "val", "test", "train"])

    for split in preferred:
        if split in available:
            return split
    return available[0]


def _save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_rows(kind: str, dataset_name: str, split: str, images_root: Path) -> Tuple[List[Dict], List[Dict]]:
    ds = load_dataset(dataset_name, split=split)

    eval_rows: List[Dict] = []
    corpus_by_source: Dict[str, Dict] = {}

    for idx, row in enumerate(ds):
        raw_id = _safe_text(_pick_first(row, ["id", "questionId", "question_id", "qid"])) or f"{kind}_{idx}"
        question = _safe_text(_pick_first(row, ["question", "query", "prompt"]))
        answers = _answers_from_row(row)
        if not question or not answers:
            continue

        doc_id = _safe_text(_pick_first(row, ["ucsf_document_id", "doc_id", "document_id", "image_id", "image_local_name"]))
        page_no = _safe_text(_pick_first(row, ["ucsf_document_page_no", "page_no", "page", "page_id"]))
        source_base = doc_id if doc_id else raw_id
        source_id = source_base if not page_no else f"{source_base}_p{page_no}"

        image_obj = _pick_first(row, ["image", "img"])
        image_path = images_root / kind / split / f"{raw_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if image_obj is not None:
            try:
                image_obj.save(str(image_path))
            except Exception:
                continue
        else:
            continue

        eval_rows.append(
            {
                "id": f"{kind}::{raw_id}",
                "image": str(image_path),
                "query": question,
                "answers": answers,
                "source_id": source_id,
                "expected_retrieval_ids": [source_id],
                "meta": {
                    "benchmark": kind,
                    "dataset_name": dataset_name,
                    "split": split,
                    "raw_id": raw_id,
                },
            }
        )

        text = _text_from_row(row, question)
        if source_id not in corpus_by_source:
            corpus_by_source[source_id] = {
                "id": f"{kind}::{source_id}",
                "text": text,
                "source_id": source_id,
                "meta": {
                    "benchmark": kind,
                    "dataset_name": dataset_name,
                    "split": split,
                },
            }

    return eval_rows, list(corpus_by_source.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark data into unified JSONL files.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset", default="all", choices=["docvqa", "infographicvqa", "all"])
    parser.add_argument("--output-dir", default="benchmarks")
    parser.add_argument("--images-dir", default="benchmarks/images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    images_dir = Path(args.images_dir)

    dataset_kinds = ["docvqa", "infographicvqa"] if args.dataset == "all" else [args.dataset]

    eval_rows: List[Dict] = []
    corpus_rows: List[Dict] = []
    used: Dict[str, Dict] = {}

    for kind in dataset_kinds:
        dataset_name = _select_dataset_name(kind)
        resolved_split = _select_split(dataset_name, args.split)
        rows_eval, rows_corpus = _build_rows(kind, dataset_name, resolved_split, images_dir)
        eval_rows.extend(rows_eval)
        corpus_rows.extend(rows_corpus)
        used[kind] = {"dataset_name": dataset_name, "split": resolved_split, "eval": len(rows_eval), "corpus": len(rows_corpus)}

    eval_rows = [r for r in eval_rows if Path(r["image"]).exists()]

    eval_path = output_dir / "eval_set.jsonl"
    corpus_path = output_dir / "corpus.jsonl"
    _save_jsonl(eval_path, eval_rows)
    _save_jsonl(corpus_path, corpus_rows)

    summary = {
        "requested_dataset": args.dataset,
        "requested_split": args.split,
        "used": used,
        "eval_samples": len(eval_rows),
        "corpus_entries": len(corpus_rows),
        "eval_path": str(eval_path.resolve()),
        "corpus_path": str(corpus_path.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
