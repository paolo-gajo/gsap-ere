#!/usr/bin/env python3
"""Score one zero-shot GSAP-ERE JSON prediction with exact micro metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DOCUMENT_ID = "00016_2106_09462.txt"

ENTITY_TYPES = frozenset(
    {
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
    }
)

# `processed` and `versionOf` occur in the released data although the supplied
# annotation guideline and paper define only the other 18 labels.
RELATION_TYPES = frozenset(
    {
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
        "processed",
        "size",
        "sourcedFrom",
        "trainedOn",
        "transformedFrom",
        "url",
        "usedFor",
        "versionOf",
    }
)

SYMMETRIC_RELATIONS = frozenset({"coreference", "isComparedTo"})

Span = tuple[int, int, int]  # segment_id, inclusive start, inclusive end
Relation = tuple[int, int, int, int, int, int, str]


@dataclass(frozen=True)
class Gold:
    document_id: str
    tokens: tuple[tuple[str, ...], ...]
    span_types: dict[Span, frozenset[str]]
    relations: frozenset[Relation]
    entity_annotation_rows: int


@dataclass(frozen=True)
class Prediction:
    span_types: dict[Span, str]
    relations: dict[Relation, tuple[str, str]]
    warnings: tuple[str, ...]


def canonical_relation(head: Span, tail: Span, label: str) -> Relation:
    if label in SYMMETRIC_RELATIONS and tail < head:
        head, tail = tail, head
    return (*head, *tail, label)


def load_gold(
    gold_path: Path, vocabulary_path: Path, document_id: str
) -> Gold:
    documents = []
    with gold_path.open(encoding="utf-8") as stream:
        for line in stream:
            document = json.loads(line)
            if document["doc_id"] == document_id:
                documents.append(document)
    if len(documents) != 1:
        raise ValueError(
            f"expected one {document_id!r} record in {gold_path}, found {len(documents)}"
        )

    document = documents[0]
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    tokens = tuple(
        tuple(vocabulary[token_id] for token_id in token_ids)
        for token_ids in document["sentences"]
    )

    span_types_mutable: dict[Span, set[str]] = {}
    relation_set: set[Relation] = set()
    annotation_rows = 0
    offset = 0

    for segment_id, (segment_tokens, entities, relations) in enumerate(
        zip(document["sentences"], document["ner"], document["relations"])
    ):
        for global_start, global_end, label in entities:
            span = (segment_id, global_start - offset, global_end - offset)
            if not (0 <= span[1] <= span[2] < len(segment_tokens)):
                raise ValueError(f"invalid gold entity span: {span}")
            span_types_mutable.setdefault(span, set()).add(label)
            annotation_rows += 1

        for head_start, head_end, tail_start, tail_end, label in relations:
            head = (segment_id, head_start - offset, head_end - offset)
            tail = (segment_id, tail_start - offset, tail_end - offset)
            relation_set.add(canonical_relation(head, tail, label))

        offset += len(segment_tokens)

    span_types = {
        span: frozenset(labels) for span, labels in span_types_mutable.items()
    }
    return Gold(
        document_id=document_id,
        tokens=tokens,
        span_types=span_types,
        relations=frozenset(relation_set),
        entity_annotation_rows=annotation_rows,
    )


def strip_one_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped
    if lines[0].strip().lower() not in {"```", "```json"}:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def require_integer(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    return value


def warn_extra_fields(
    value: dict[str, Any], expected: set[str], context: str, warnings: list[str]
) -> None:
    extras = sorted(set(value) - expected)
    if extras:
        warnings.append(f"{context} has ignored extra fields: {', '.join(extras)}")


def load_prediction(prediction_path: Path, gold: Gold) -> Prediction:
    raw_text = prediction_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(strip_one_json_fence(raw_text))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid prediction JSON: {error}") from error

    payload = require_object(payload, "top-level prediction")
    warnings: list[str] = []
    required_top = {"document_id", "entities", "relations"}
    missing_top = sorted(required_top - set(payload))
    if missing_top:
        raise ValueError(f"top-level prediction is missing: {', '.join(missing_top)}")
    warn_extra_fields(payload, required_top, "top-level prediction", warnings)

    document_id = require_string(payload["document_id"], "document_id")
    if document_id != gold.document_id:
        raise ValueError(
            f"document_id is {document_id!r}; expected {gold.document_id!r}"
        )

    entity_rows = require_list(payload["entities"], "entities")
    entities_by_id: dict[str, tuple[Span, str]] = {}
    predicted_span_types: dict[Span, str] = {}
    entity_fields = {"id", "segment_id", "start", "end", "type", "text"}

    for index, value in enumerate(entity_rows):
        context = f"entities[{index}]"
        entity = require_object(value, context)
        missing = sorted(entity_fields - set(entity))
        if missing:
            raise ValueError(f"{context} is missing: {', '.join(missing)}")
        warn_extra_fields(entity, entity_fields, context, warnings)

        entity_id = require_string(entity["id"], f"{context}.id")
        if entity_id in entities_by_id:
            raise ValueError(f"duplicate entity id: {entity_id!r}")

        segment_id = require_integer(entity["segment_id"], f"{context}.segment_id")
        start = require_integer(entity["start"], f"{context}.start")
        end = require_integer(entity["end"], f"{context}.end")
        if not 0 <= segment_id < len(gold.tokens):
            raise ValueError(f"{context}.segment_id is out of range: {segment_id}")
        if not 0 <= start <= end < len(gold.tokens[segment_id]):
            raise ValueError(
                f"{context} has out-of-range span ({start}, {end}) for "
                f"segment {segment_id} of length {len(gold.tokens[segment_id])}"
            )
        span = (segment_id, start, end)
        if span in predicted_span_types:
            raise ValueError(
                f"duplicate entity span {span}; output only one type per unique span"
            )

        label = require_string(entity["type"], f"{context}.type")
        if label not in ENTITY_TYPES:
            warnings.append(f"{context} has unknown entity type {label!r}; scored as FP")
        supplied_text = require_string(entity["text"], f"{context}.text")
        expected_text = " ".join(gold.tokens[segment_id][start : end + 1])
        if supplied_text != expected_text:
            warnings.append(
                f"{context}.text mismatch: expected {expected_text!r}; coordinates retained"
            )

        entities_by_id[entity_id] = (span, label)
        predicted_span_types[span] = label

    relation_rows = require_list(payload["relations"], "relations")
    predicted_relations: dict[Relation, tuple[str, str]] = {}
    relation_fields = {"head", "type", "tail"}

    for index, value in enumerate(relation_rows):
        context = f"relations[{index}]"
        relation = require_object(value, context)
        missing = sorted(relation_fields - set(relation))
        if missing:
            raise ValueError(f"{context} is missing: {', '.join(missing)}")
        warn_extra_fields(relation, relation_fields, context, warnings)

        head_id = require_string(relation["head"], f"{context}.head")
        tail_id = require_string(relation["tail"], f"{context}.tail")
        if head_id not in entities_by_id:
            raise ValueError(f"{context}.head refers to unknown entity id {head_id!r}")
        if tail_id not in entities_by_id:
            raise ValueError(f"{context}.tail refers to unknown entity id {tail_id!r}")
        label = require_string(relation["type"], f"{context}.type")
        if label not in RELATION_TYPES:
            warnings.append(f"{context} has unknown relation type {label!r}; scored as FP")

        head_span, head_type = entities_by_id[head_id]
        tail_span, tail_type = entities_by_id[tail_id]
        if head_span[0] != tail_span[0]:
            warnings.append(f"{context} crosses segments; scored as FP")
        if label in SYMMETRIC_RELATIONS and tail_span < head_span:
            head_span, tail_span = tail_span, head_span
            head_type, tail_type = tail_type, head_type
        key = (*head_span, *tail_span, label)
        if key in predicted_relations:
            warnings.append(f"{context} duplicates an earlier relation; counted once")
            continue
        predicted_relations[key] = (head_type, tail_type)

    return Prediction(
        span_types=predicted_span_types,
        relations=predicted_relations,
        warnings=tuple(warnings),
    )


def prf(tp: int, gold_count: int, predicted_count: int) -> dict[str, int | float]:
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / gold_count if gold_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_percent": round(100 * precision, 2),
        "recall_percent": round(100 * recall, 2),
        "f1_percent": round(100 * f1, 2),
        "tp": tp,
        "fp": predicted_count - tp,
        "fn": gold_count - tp,
        "gold": gold_count,
        "predicted": predicted_count,
    }


def score(gold: Gold, prediction: Prediction) -> dict[str, Any]:
    ner_tp = sum(
        predicted_type in gold.span_types.get(span, frozenset())
        for span, predicted_type in prediction.span_types.items()
    )
    ner = prf(ner_tp, len(gold.span_types), len(prediction.span_types))

    predicted_relation_keys = set(prediction.relations)
    rel_tp = len(gold.relations & predicted_relation_keys)
    rel = prf(rel_tp, len(gold.relations), len(predicted_relation_keys))

    rel_plus_tp = 0
    for relation, (head_type, tail_type) in prediction.relations.items():
        if relation not in gold.relations:
            continue
        head_span = relation[0:3]
        tail_span = relation[3:6]
        if (
            head_type in gold.span_types.get(head_span, frozenset())
            and tail_type in gold.span_types.get(tail_span, frozenset())
        ):
            rel_plus_tp += 1
    rel_plus = prf(
        rel_plus_tp, len(gold.relations), len(predicted_relation_keys)
    )

    return {
        "document_id": gold.document_id,
        "averaging": "micro",
        "matching": "exact token spans and case-sensitive labels",
        "ner": ner,
        "rel": rel,
        "rel_plus": rel_plus,
        "gold_counts": {
            "entity_annotation_rows": gold.entity_annotation_rows,
            "entity_unique_spans": len(gold.span_types),
            "relations": len(gold.relations),
        },
        "warnings": list(prediction.warnings),
    }


def evaluate_prediction(
    prediction_path: Path,
    gold_path: Path,
    vocabulary_path: Path,
    document_id: str = DEFAULT_DOCUMENT_ID,
) -> dict[str, Any]:
    gold = load_gold(gold_path, vocabulary_path, document_id)
    prediction = load_prediction(prediction_path, gold)
    return score(gold, prediction)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Score exact micro NER, REL, and REL+ for one GSAP-ERE response."
    )
    parser.add_argument("prediction", type=Path, help="model response JSON file")
    parser.add_argument(
        "--gold",
        type=Path,
        default=root / "data" / "train.jsonl",
        help="gold GSAP-ERE JSONL (default: data/train.jsonl)",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=root / "vocabulary.json",
        help="token vocabulary JSON (default: vocabulary.json)",
    )
    parser.add_argument(
        "--document-id",
        default=DEFAULT_DOCUMENT_ID,
        help=f"document to score (default: {DEFAULT_DOCUMENT_ID})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = evaluate_prediction(
            args.prediction, args.gold, args.vocabulary, args.document_id
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
