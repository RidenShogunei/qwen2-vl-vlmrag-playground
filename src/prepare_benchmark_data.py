import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from datasets import get_dataset_split_names, load_dataset


DATASET_CANDIDATES = {
    "docvqa": ["nielsr/docvqa_1200_examples"],
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


def _answers_from_row(row: Dict):
    value = _pick_first(row, ["answers", "answer", "gold_answers", "label"])
    if value is None:
        return []
    if isinstance(value, list):
        return [_safe_text(v) for v in value if _safe_text(v)]
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
    parts = []
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


def _select_docvqa_name() -> str:
    for name in DATASET_CANDIDATES["docvqa"]:
        try:
            _ = get_dataset_split_names(name)
            return name
        except Exception:
            continue
    raise RuntimeError("No accessible dataset found for docvqa")


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


def _load_existing_ids(path: Path, id_key: str) -> set:
    if not path.exists():
        return set()
    out = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get(id_key):
                    out.add(row[id_key])
            except Exception:
                continue
    return out


def _append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _process_docvqa(dataset_name: str, split: str, images_root: Path, eval_path: Path, corpus_path: Path, checkpoint_every: int, max_samples: Optional[int]) -> Dict:
    ds = load_dataset(dataset_name, split=split)

    existing_eval_ids = _load_existing_ids(eval_path, "id")
    existing_corpus_source_ids = _load_existing_ids(corpus_path, "source_id")

    processed = 0
    added_eval = 0
    added_corpus = 0
    skipped = 0

    for idx, row in enumerate(ds):
        if max_samples is not None and processed >= max_samples:
            break

        raw_id = _safe_text(_pick_first(row, ["id", "questionId", "question_id", "qid"])) or f"docvqa_{idx}"
        global_id = f"docvqa::{raw_id}"
        if global_id in existing_eval_ids:
            skipped += 1
            processed += 1
            continue

        question = _safe_text(_pick_first(row, ["question", "query", "prompt"]))
        answers = _answers_from_row(row)
        if not question or not answers:
            skipped += 1
            processed += 1
            continue

        doc_id = _safe_text(_pick_first(row, ["ucsf_document_id", "doc_id", "document_id", "image_id", "image_local_name"]))
        page_no = _safe_text(_pick_first(row, ["ucsf_document_page_no", "page_no", "page", "page_id"]))
        source_base = doc_id if doc_id else raw_id
        source_id = source_base if not page_no else f"{source_base}_p{page_no}"

        image_obj = _pick_first(row, ["image", "img"])
        image_path = images_root / "docvqa" / split / f"{raw_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if image_obj is not None and not image_path.exists():
            try:
                image_obj.save(str(image_path))
            except Exception:
                skipped += 1
                processed += 1
                continue

        if not image_path.exists():
            skipped += 1
            processed += 1
            continue

        eval_row = {
            "id": global_id,
            "image": str(image_path),
            "query": question,
            "answers": answers,
            "source_id": source_id,
            "expected_retrieval_ids": [source_id],
            "meta": {
                "benchmark": "docvqa",
                "dataset_name": dataset_name,
                "split": split,
                "raw_id": raw_id,
            },
        }
        _append_jsonl(eval_path, eval_row)
        existing_eval_ids.add(global_id)
        added_eval += 1

        if source_id not in existing_corpus_source_ids:
            corpus_row = {
                "id": f"docvqa::{source_id}",
                "text": _text_from_row(row, question),
                "source_id": source_id,
                "meta": {
                    "benchmark": "docvqa",
                    "dataset_name": dataset_name,
                    "split": split,
                },
            }
            _append_jsonl(corpus_path, corpus_row)
            existing_corpus_source_ids.add(source_id)
            added_corpus += 1

        processed += 1
        if checkpoint_every > 0 and processed % checkpoint_every == 0:
            print(json.dumps({"kind": "docvqa", "processed": processed, "added_eval": added_eval, "added_corpus": added_corpus, "skipped": skipped}, ensure_ascii=False))

    return {"dataset_name": dataset_name, "split": split, "processed": processed, "added_eval": added_eval, "added_corpus": added_corpus, "skipped": skipped}


def _process_infographic_local(raw_dir: Path, split: str, images_root: Path, eval_path: Path, corpus_path: Path, checkpoint_every: int, max_samples: Optional[int]) -> Dict:
    import pyarrow.parquet as pq
    from PIL import Image
    from io import BytesIO

    parquet_files = sorted(glob.glob(str(raw_dir / "data" / "*.parquet")))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {raw_dir}/data")

    existing_eval_ids = _load_existing_ids(eval_path, "id")
    existing_corpus_source_ids = _load_existing_ids(corpus_path, "source_id")

    processed = 0
    added_eval = 0
    added_corpus = 0
    skipped = 0

    for pf in parquet_files:
        table = pq.read_table(pf)
        rows = table.to_pylist()
        shard = Path(pf).stem

        for ridx, row in enumerate(rows):
            texts = row.get("texts") or []
            images = row.get("images") or []
            if not isinstance(texts, list) or not isinstance(images, list) or not texts or not images:
                continue

            # Build per-image grouped corpus content.
            grouped = {}
            for tidx, qa in enumerate(texts):
                if not isinstance(qa, dict):
                    continue
                question = _safe_text(qa.get("user"))
                answer = _safe_text(qa.get("assistant"))
                if not question or not answer:
                    continue
                img_idx = min(tidx, len(images) - 1)
                grouped.setdefault(img_idx, []).append((question, answer, tidx))

            # Save image files and corpus rows per image index.
            for img_idx, qa_list in grouped.items():
                source_id = f"infographic::{shard}::{ridx}::img{img_idx}"
                img_obj = images[img_idx]
                img_bytes = img_obj.get("bytes") if isinstance(img_obj, dict) else None
                img_path = images_root / "infographicvqa" / split / f"{shard}_{ridx}_{img_idx}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                if not img_path.exists():
                    if not img_bytes:
                        continue
                    try:
                        Image.open(BytesIO(img_bytes)).save(str(img_path))
                    except Exception:
                        continue

                if source_id not in existing_corpus_source_ids:
                    lines = [f"Q: {q}\nA: {a}" for q, a, _ in qa_list]
                    corpus_row = {
                        "id": f"infographicvqa::{source_id}",
                        "text": "\n\n".join(lines),
                        "source_id": source_id,
                        "meta": {"benchmark": "infographicvqa", "dataset_name": "local_parquet", "split": split, "image_idx": img_idx},
                    }
                    _append_jsonl(corpus_path, corpus_row)
                    existing_corpus_source_ids.add(source_id)
                    added_corpus += 1

                for question, answer, tidx in qa_list:
                    if max_samples is not None and processed >= max_samples:
                        break
                    global_id = f"infographicvqa::{shard}::{ridx}::{tidx}"
                    if global_id in existing_eval_ids:
                        skipped += 1
                        processed += 1
                        continue

                    eval_row = {
                        "id": global_id,
                        "image": str(img_path),
                        "query": question,
                        "answers": [answer],
                        "source_id": source_id,
                        "expected_retrieval_ids": [source_id],
                        "meta": {
                            "benchmark": "infographicvqa",
                            "dataset_name": "local_parquet",
                            "split": split,
                            "raw_id": f"{shard}:{ridx}:{tidx}",
                            "image_idx": img_idx,
                        },
                    }
                    _append_jsonl(eval_path, eval_row)
                    existing_eval_ids.add(global_id)
                    added_eval += 1
                    processed += 1

                    if checkpoint_every > 0 and processed % checkpoint_every == 0:
                        print(json.dumps({"kind": "infographicvqa", "processed": processed, "added_eval": added_eval, "added_corpus": added_corpus, "skipped": skipped}, ensure_ascii=False))

                if max_samples is not None and processed >= max_samples:
                    break

            if max_samples is not None and processed >= max_samples:
                break

        if max_samples is not None and processed >= max_samples:
            break

    return {"dataset_name": "local_parquet", "split": split, "processed": processed, "added_eval": added_eval, "added_corpus": added_corpus, "skipped": skipped}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark data into unified JSONL files with resume support.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset", default="all", choices=["docvqa", "infographicvqa", "all"])
    parser.add_argument("--output-dir", default="benchmarks")
    parser.add_argument("--images-dir", default="benchmarks/images")
    parser.add_argument("--local-infographic-dir", default="benchmarks/raw/infographic_vqa")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    images_dir = Path(args.images_dir)
    local_infographic_dir = Path(args.local_infographic_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_path = output_dir / "eval_set.jsonl"
    corpus_path = output_dir / "corpus.jsonl"

    kinds = ["docvqa", "infographicvqa"] if args.dataset == "all" else [args.dataset]
    used = {}

    for kind in kinds:
        if kind == "docvqa":
            dataset_name = _select_docvqa_name()
            resolved_split = _select_split(dataset_name, args.split)
            stats = _process_docvqa(
                dataset_name=dataset_name,
                split=resolved_split,
                images_root=images_dir,
                eval_path=eval_path,
                corpus_path=corpus_path,
                checkpoint_every=args.checkpoint_every,
                max_samples=args.max_samples,
            )
            used[kind] = stats
        else:
            stats = _process_infographic_local(
                raw_dir=local_infographic_dir,
                split=args.split,
                images_root=images_dir,
                eval_path=eval_path,
                corpus_path=corpus_path,
                checkpoint_every=args.checkpoint_every,
                max_samples=args.max_samples,
            )
            used[kind] = stats

    summary = {
        "requested_dataset": args.dataset,
        "requested_split": args.split,
        "used": used,
        "eval_samples": _count_jsonl_rows(eval_path),
        "corpus_entries": _count_jsonl_rows(corpus_path),
        "eval_path": str(eval_path.resolve()),
        "corpus_path": str(corpus_path.resolve()),
        "resume_supported": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
