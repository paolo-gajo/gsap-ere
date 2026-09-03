#!/usr/bin/env python3
"""Move the existing John runs into the flag-based result layout."""

from __future__ import annotations

import json
from pathlib import Path


MODEL = "qwen3.8:27b"
MODEL_SLUG = "qwen3.8-27b"
DOCUMENT = "00016_2106_09462"

LEGACY_RUNS = {
    f"qwen3.8-27b-ollama-john-{DOCUMENT}": (10, 1, 0, "indices", 0),
    f"qwen3.8-27b-ollama-john-thinking-{DOCUMENT}": (10, 1, 0, "indices", 1),
    f"qwen3.8-27b-ollama-john-full-article-{DOCUMENT}": (10, 1, 1, "indices", 0),
    f"qwen3.8-27b-ollama-john-thinking-full-article-{DOCUMENT}": (10, 1, 1, "indices", 1),
    f"qwen3.8-27b-ollama-john-re-5shot-{DOCUMENT}": (10, 5, 0, "indices", 0),
    f"qwen3.8-27b-ollama-john-thinking-re-5shot-{DOCUMENT}": (10, 5, 0, "indices", 1),
    f"qwen3.8-27b-ollama-john-re-10shot-{DOCUMENT}": (10, 10, 0, "indices", 0),
    f"qwen3.8-27b-ollama-john-thinking-re-10shot-{DOCUMENT}": (10, 10, 0, "indices", 1),
    f"qwen3.8-27b-ollama-john-zero-icl-{DOCUMENT}": (0, 0, 0, "indices", 0),
    f"qwen3.8-27b-ollama-john-thinking-zero-icl-{DOCUMENT}": (0, 0, 0, "indices", 1),
    f"qwen3.8-27b-ollama-john-zero-icl-ner-inline-{DOCUMENT}": (0, 0, 0, "inline", 0),
    f"qwen3.8-27b-ollama-john-thinking-zero-icl-ner-inline-{DOCUMENT}": (0, 0, 0, "inline", 1),
}


def result_directory(
    root: Path,
    ner_examples: int,
    re_examples: int,
    full_context: int,
    ner_output: str,
    thinking: int,
) -> Path:
    return (
        root
        / f"model={MODEL_SLUG}"
        / f"ner-examples={ner_examples}"
        / f"re-examples={re_examples}"
        / f"full-context={full_context}"
        / f"ner-output={ner_output}"
        / f"thinking={thinking}"
        / DOCUMENT
    )


def update_run_json(directory: Path, settings: tuple[int, int, int, str, int]) -> None:
    run_path = directory / "run.json"
    if not run_path.exists():
        return

    ner_examples, re_examples, full_context, ner_output, thinking = settings
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["configuration"] = {
        "model": MODEL,
        "ner_examples": ner_examples,
        "re_examples": re_examples,
        "full_context": bool(full_context),
        "ner_output": ner_output,
        "thinking": bool(thinking),
    }

    repo_root = Path(__file__).resolve().parent.parent
    relative_directory = directory.relative_to(repo_root)
    for artifact in run.get("artifacts", {}).values():
        artifact["path"] = str(relative_directory / Path(artifact["path"]).name)

    run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent / "results"
    for legacy_name, settings in LEGACY_RUNS.items():
        source = root / legacy_name
        destination = result_directory(root, *settings)
        if source.exists():
            if destination.exists():
                raise SystemExit(f"refusing to merge {source} into existing {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            print(f"moved {source.name} -> {destination.relative_to(root)}")
        if destination.exists():
            update_run_json(destination, settings)


if __name__ == "__main__":
    main()
