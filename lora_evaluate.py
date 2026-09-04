#!/usr/bin/env python3
"""Evaluate a completed GSAP-ERE LoRA adapter without retraining or a base pass."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from lora_pilot import (
    BARE_TRAINING_FORMAT,
    MODEL_REVISIONS,
    REPO_ROOT,
    VOCABULARY,
    evaluate_variant,
    git_commit,
    initialize_output_dir,
    load_checkpoint_adapter,
    load_model_and_tokenizer,
    package_version,
    scorer_python,
    sha256_file,
    utc_now,
)
from paper_llm.inference import SentenceRecord, gpu_inventory, load_records


DEFAULT_MODEL_ID = "Qwen/Qwen3-14B-Base"
DEFAULT_TARGET_DATA = REPO_ROOT / "data" / "test.jsonl"


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def load_target_documents(
    target_data: Path,
    document_ids: list[str],
) -> dict[str, list[SentenceRecord]]:
    if len(document_ids) != 3:
        raise ValueError(f"expected exactly 3 --document-id values, found {len(document_ids)}")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("--document-id values must be unique")
    unsafe = [document_id for document_id in document_ids if Path(document_id).name != document_id]
    if unsafe:
        raise ValueError(f"document IDs must be plain filenames: {unsafe}")
    vocabulary = json.loads(VOCABULARY.read_text(encoding="utf-8"))
    records = load_records(target_data, vocabulary)
    requested = set(document_ids)
    unexpected = requested - {record.doc_id for record in records}
    if unexpected:
        raise ValueError(
            f"documents absent from {target_data}: {sorted(unexpected)}"
        )
    result: dict[str, list[SentenceRecord]] = {}
    for document_id in document_ids:
        targets = sorted(
            (record for record in records if record.doc_id == document_id),
            key=lambda record: record.sentence_id,
        )
        if not targets:
            raise RuntimeError(f"document contains no sentences: {document_id}")
        sentence_ids = [record.sentence_id for record in targets]
        if sentence_ids != list(range(len(targets))):
            raise RuntimeError(
                f"non-contiguous sentence IDs for {document_id}: {sentence_ids[:10]}"
            )
        result[document_id] = targets
    return result


def validate_source_run(source_run: Path, model_id: str) -> dict[str, Any]:
    if not (source_run / "COMPLETE").is_file():
        raise RuntimeError(f"source run is not complete: {source_run}")
    run_path = source_run / "run.json"
    if not run_path.is_file():
        raise RuntimeError(f"source run has no run.json: {source_run}")
    source = json.loads(run_path.read_text(encoding="utf-8"))
    expected_revision = MODEL_REVISIONS[model_id]
    if source.get("status") != "complete":
        raise RuntimeError(f"source run status is not complete: {source.get('status')!r}")
    if source.get("model", {}).get("id") != model_id:
        raise RuntimeError("source adapter model ID does not match --model-id")
    if source.get("model", {}).get("revision") != expected_revision:
        raise RuntimeError("source adapter model revision does not match pinned revision")
    if source.get("training_format") != BARE_TRAINING_FORMAT:
        raise RuntimeError("source adapter was not trained in bare format")
    source_vocabulary_sha256 = source.get("inputs", {}).get("vocabulary_sha256")
    if source_vocabulary_sha256 != sha256_file(VOCABULARY):
        raise RuntimeError("current vocabulary does not match the source training run")
    adapter_dir = source_run / "adapter"
    expected_files = source.get("adapter_files")
    if not isinstance(expected_files, list) or not expected_files:
        raise RuntimeError("source run has no adapter-file manifest")
    for entry in expected_files:
        path = adapter_dir / entry["path"]
        if not path.is_file():
            raise RuntimeError(f"missing source adapter file: {path}")
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"source adapter file failed integrity check: {path}")
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-data", type=Path, default=DEFAULT_TARGET_DATA)
    parser.add_argument("--document-id", action="append", required=True)
    parser.add_argument("--model-id", choices=tuple(MODEL_REVISIONS), default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(args: argparse.Namespace, output_dir: Path) -> None:
    started_at = utc_now()
    wall_start = time.monotonic()
    source_run = args.source_run.resolve()
    target_data = args.target_data.resolve()
    source = validate_source_run(source_run, args.model_id)
    documents = load_target_documents(target_data, args.document_id)
    model_revision = MODEL_REVISIONS[args.model_id]
    state: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "saved bare QLoRA adapter evaluation",
        "status": "loading_model",
        "git_commit": git_commit(),
        "started_at": started_at,
        "source_run": {
            "path": str(source_run),
            "git_commit": source.get("git_commit"),
            "slurm_job_id": source.get("system", {}).get("slurm_job_id"),
            "adapter_files": source["adapter_files"],
        },
        "model": {
            "id": args.model_id,
            "revision": model_revision,
            "quantization": source["model"]["quantization"],
            "adapter": "loaded from source_run; no training in this job",
        },
        "evaluation": {
            "base_variant_evaluated": False,
            "training_performed": False,
            "prompt_format": "bare x-only content",
            "target_data": str(target_data),
            "target_data_sha256": sha256_file(target_data),
            "vocabulary_sha256": sha256_file(VOCABULARY),
            "planned_document_ids": args.document_id,
            "planned_sentence_counts": {
                document_id: len(targets)
                for document_id, targets in documents.items()
            },
            "completed_document_ids": [],
            "generation": "greedy Hugging Face generation",
            "relation_candidates": "all ordered pairs of predicted entities within each sentence",
        },
    }
    atomic_write_json(output_dir / "run.json", state)

    torch, tokenizer, model, target_report = load_model_and_tokenizer(
        args.seed,
        args.model_id,
        model_revision,
    )
    load_checkpoint_adapter(model, source_run)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    atomic_write_json(output_dir / "targeted_modules.json", target_report)
    state["status"] = "evaluating"
    state["model_loaded_at"] = utc_now()
    atomic_write_json(output_dir / "run.json", state)

    document_results: dict[str, Any] = {}
    prediction_paths: list[Path] = []
    for document_id, targets in documents.items():
        document_start = time.monotonic()
        result = evaluate_variant(
            name=f"documents/{document_id}",
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            targets=targets,
            training=[],
            similarities=None,
            pair_examples=[],
            pairs_by_signature={},
            output_dir=output_dir,
            bare=True,
            model_id=args.model_id,
            gold_data=target_data,
        )
        document_dir = output_dir / "documents" / document_id
        document_summary = {
            "document_id": document_id,
            "sentence_count": len(targets),
            "result": result,
            "wall_seconds": round(time.monotonic() - document_start, 6),
            "finished_at": utc_now(),
            "artifacts": {
                name: {
                    "bytes": (document_dir / name).stat().st_size,
                    "sha256": sha256_file(document_dir / name),
                }
                for name in ("prediction.json", "retrieval.json", "scores.json", "trace.jsonl")
            },
        }
        atomic_write_json(document_dir / "result.json", document_summary)
        atomic_write_json(
            document_dir / "COMPLETE",
            {
                "document_id": document_id,
                "finished_at": document_summary["finished_at"],
                "result_sha256": sha256_file(document_dir / "result.json"),
            },
        )
        document_results[document_id] = document_summary
        prediction_paths.append(document_dir / "prediction.json")
        state["evaluation"]["completed_document_ids"] = list(document_results)
        state["document_results"] = document_results
        atomic_write_json(output_dir / "run.json", state)
        print(
            f"[complete {len(document_results)}/{len(documents)}] {document_id}",
            flush=True,
        )

    aggregate_path = output_dir / "aggregate_scores.json"
    subprocess.run(
        [
            scorer_python(),
            str(REPO_ROOT / "paper_llm" / "evaluate.py"),
            "--gold",
            str(target_data),
            "--output",
            str(aggregate_path),
            *[str(path) for path in prediction_paths],
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    state.update(
        {
            "status": "complete",
            "aggregate_scores": json.loads(aggregate_path.read_text(encoding="utf-8")),
            "finished_at": utc_now(),
            "wall_seconds": round(time.monotonic() - wall_start, 6),
            "software": {
                name: package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "bitsandbytes",
                    "accelerate",
                )
            },
            "system": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "gpus": gpu_inventory(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            },
        }
    )
    atomic_write_json(output_dir / "run.json", state)
    atomic_write_json(
        output_dir / "EVALUATION_COMPLETE",
        {
            "finished_at": state["finished_at"],
            "git_commit": state["git_commit"],
            "document_ids": args.document_id,
            "run_sha256": sha256_file(output_dir / "run.json"),
        },
    )
    print(json.dumps(state["aggregate_scores"], indent=2), flush=True)


def main() -> None:
    args = parse_args()
    output_dir = initialize_output_dir(args.output_dir)
    try:
        run(args, output_dir)
    except BaseException as error:
        atomic_write_json(
            output_dir / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": utc_now(),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
