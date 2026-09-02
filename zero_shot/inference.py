#!/usr/bin/env python3
"""Run one GSAP-ERE prompt through an Ollama model and record provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOCUMENT_ID = "00016_2106_09462.txt"
ENTITY_TYPES = [
    "MLModel",
    "MLModelGeneric",
    "ModelArchitecture",
    "Method",
    "Dataset",
    "DatasetGeneric",
    "DataSource",
    "ReferenceLink",
    "Task",
    "URL",
]
RELATION_TYPES = [
    "appliedTo",
    "architecture",
    "benchmarkFor",
    "citation",
    "coreference",
    "evaluatedOn",
    "generatedBy",
    "hasInstanceType",
    "isBasedOn",
    "isComparedTo",
    "isHyponymOf",
    "isPartOf",
    "size",
    "sourcedFrom",
    "trainedOn",
    "transformedFrom",
    "usedFor",
    "url",
]

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "document_id": {"type": "string", "const": DOCUMENT_ID},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "segment_id": {"type": "integer", "minimum": 0},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 0},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                    "text": {"type": "string", "minLength": 1},
                },
                "required": ["id", "segment_id", "start", "end", "type", "text"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "head": {"type": "string", "minLength": 1},
                    "type": {"type": "string", "enum": RELATION_TYPES},
                    "tail": {"type": "string", "minLength": 1},
                },
                "required": ["head", "type", "tail"],
            },
        },
    },
    "required": ["document_id", "entities", "relations"],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def api_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code} for {path}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"cannot reach Ollama at {base_url}: {error.reason}") from error


def git_commit(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def gpu_inventory() -> list[str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def model_record(tags: dict[str, Any], requested_model: str) -> dict[str, Any] | None:
    aliases = {requested_model, f"{requested_model}:latest"}
    if ":" not in requested_model:
        aliases.add(f"{requested_model}:latest")
    for model in tags.get("models", []):
        names = {model.get("name"), model.get("model")}
        if aliases & names:
            return {
                key: model.get(key)
                for key in ("name", "model", "digest", "size", "modified_at", "details")
                if model.get(key) is not None
            }
    return None


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the zero-shot GSAP-ERE prompt through Ollama."
    )
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--prompt", type=Path, default=script_dir / "prompt.md")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "qwen3.8-27b-ollama-john",
    )
    parser.add_argument("--think", choices=("false", "low", "medium", "high"), default="high")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-ctx", type=int, default=65536)
    parser.add_argument("--num-predict", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_bytes = args.prompt.read_bytes()
    prompt = prompt_bytes.decode("utf-8")
    output_dir = args.output_dir.resolve()
    prediction_path = output_dir / "prediction.json"
    metadata_path = output_dir / "run.json"

    existing = [path for path in (prediction_path, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to overwrite existing result files: {joined}")
    output_dir.mkdir(parents=True, exist_ok=True)

    server_version = api_json(args.base_url, "/api/version", timeout=30)
    tags_before = api_json(args.base_url, "/api/tags", timeout=30)
    options = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "seed": args.seed,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
    }
    think: bool | str = False if args.think == "false" else args.think
    request_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "think": think,
        "keep_alive": 0,
        "options": options,
    }

    started_at = datetime.now(timezone.utc)
    wall_start = time.monotonic()
    response = api_json(
        args.base_url, "/api/chat", payload=request_payload, timeout=args.timeout
    )
    wall_seconds = time.monotonic() - wall_start
    finished_at = datetime.now(timezone.utc)

    message = response.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("Ollama response has no string message.content")
    content = message["content"].strip()
    thinking = message.get("thinking", "")
    if not isinstance(thinking, str):
        thinking = ""

    prediction_bytes = (content + "\n").encode("utf-8")
    prediction_path.write_bytes(prediction_bytes)
    try:
        parsed_prediction = json.loads(content)
        prediction_json_valid = isinstance(parsed_prediction, dict)
        prediction_json_error = None
    except json.JSONDecodeError as error:
        prediction_json_valid = False
        prediction_json_error = str(error)

    response_metrics = {
        key: response.get(key)
        for key in (
            "created_at",
            "done",
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )
        if response.get(key) is not None
    }
    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    if isinstance(eval_count, int) and isinstance(eval_duration, int) and eval_duration:
        response_metrics["generation_tokens_per_second"] = round(
            eval_count / (eval_duration / 1_000_000_000), 3
        )

    repo_root = Path(__file__).resolve().parents[1]
    metadata = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "wall_seconds": round(wall_seconds, 3),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": gpu_inventory(),
        "git_commit": git_commit(repo_root),
        "prompt": {
            "path": str(args.prompt.resolve()),
            "bytes": len(prompt_bytes),
            "sha256": sha256_bytes(prompt_bytes),
        },
        "prediction": {
            "path": str(prediction_path),
            "bytes": len(prediction_bytes),
            "sha256": sha256_bytes(prediction_bytes),
            "json_object_valid": prediction_json_valid,
            "json_error": prediction_json_error,
        },
        "backend": {
            "name": "Ollama (llama.cpp engine)",
            "base_url": args.base_url,
            "version": server_version.get("version"),
        },
        "model": {
            "requested": args.model,
            "returned": response.get("model"),
            "installed_record": model_record(tags_before, args.model),
        },
        "generation": {
            "think": think,
            "structured_output": "JSON Schema",
            "stream": False,
            "options": options,
        },
        "response": response_metrics,
        "thinking": {
            "characters": len(thinking),
            "sha256": sha256_bytes(thinking.encode("utf-8")) if thinking else None,
            "stored": False,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"prediction: {prediction_path}")
    print(f"metadata:   {metadata_path}")
    print(f"JSON object valid: {prediction_json_valid}")
    if response_metrics.get("generation_tokens_per_second") is not None:
        print(
            "generation tokens/s: "
            f"{response_metrics['generation_tokens_per_second']}"
        )


if __name__ == "__main__":
    main()
