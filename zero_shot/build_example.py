#!/usr/bin/env python3
"""Build the annotation-free article and pasteable zero-shot prompt."""

from __future__ import annotations

import json
from pathlib import Path


DOCUMENT_ID = "00016_2106_09462.txt"
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.jsonl"
VOCABULARY_PATH = ROOT / "vocabulary.json"
PROMPT_TEMPLATE_PATH = OUT / "prompt_template.md"

CLOSE_PUNCTUATION = {",", ".", ";", ":", "!", "?", "%", ")", "]", "}", "'s"}
OPEN_PUNCTUATION = {"(", "[", "{"}
JOIN_BOTH_SIDES = {"-", "/", ":/"}


def load_document() -> dict:
    matches = []
    with DATA_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            document = json.loads(line)
            if document["doc_id"] == DOCUMENT_ID:
                matches.append(document)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {DOCUMENT_ID!r} record in {DATA_PATH}, found {len(matches)}"
        )
    return matches[0]


def detokenize(tokens: list[str]) -> str:
    """Create a readable rendering without changing the authoritative tokens."""
    rendered = ""
    previous = ""
    before_previous = ""
    quote_is_open = False

    for token in tokens:
        if not rendered:
            needs_space = False
        elif token == '"':
            needs_space = not quote_is_open and previous not in OPEN_PUNCTUATION
            quote_is_open = not quote_is_open
        elif previous == '"' and quote_is_open:
            needs_space = False
        elif previous == "," and before_previous.isdigit() and token.isdigit():
            needs_space = False
        elif previous == "∼" or (
            previous.isdigit()
            and len(token) > 1
            and token[0] in {"−", "+"}
            and token[1:].isdigit()
        ):
            needs_space = False
        elif token in CLOSE_PUNCTUATION or previous in OPEN_PUNCTUATION:
            needs_space = False
        elif token in JOIN_BOTH_SIDES or previous in JOIN_BOTH_SIDES:
            needs_space = False
        else:
            needs_space = True

        if needs_space:
            rendered += " "
        rendered += token
        before_previous = previous
        previous = token

    return rendered


def build_model_input(document: dict, vocabulary: list[str]) -> dict:
    segments = []
    for segment_id, token_ids in enumerate(document["sentences"]):
        tokens = [vocabulary[token_id] for token_id in token_ids]
        segments.append(
            {
                "segment_id": segment_id,
                "text": detokenize(tokens),
                "indexed_tokens": [[index, token] for index, token in enumerate(tokens)],
            }
        )
    return {
        "document_id": document["doc_id"],
        "indexing": "zero-based token indices local to each segment; end is inclusive",
        "segments": segments,
    }


def main() -> None:
    document = load_document()
    vocabulary = json.loads(VOCABULARY_PATH.read_text(encoding="utf-8"))
    model_input = build_model_input(document, vocabulary)

    article = "\n\n".join(segment["text"] for segment in model_input["segments"])
    (OUT / "article.txt").write_text(article + "\n", encoding="utf-8")

    serialized_input = json.dumps(model_input, ensure_ascii=False, indent=2)
    (OUT / "model_input.json").write_text(serialized_input + "\n", encoding="utf-8")

    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    placeholder = "{{MODEL_INPUT_JSON}}"
    if template.count(placeholder) != 1:
        raise RuntimeError(f"expected exactly one {placeholder} in {PROMPT_TEMPLATE_PATH}")
    prompt = template.replace(placeholder, serialized_input)
    (OUT / "prompt.md").write_text(prompt.rstrip() + "\n", encoding="utf-8")

    token_count = sum(len(segment["indexed_tokens"]) for segment in model_input["segments"])
    print(
        f"wrote article.txt, model_input.json, and prompt.md "
        f"({len(model_input['segments'])} segments; {token_count} tokens)"
    )


if __name__ == "__main__":
    main()
