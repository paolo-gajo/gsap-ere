#!/usr/bin/env python3
"""Score paper-pipeline JSON with the authors' gsapere evaluator."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from gsapere.evaluation.hgere import evaluate as gsapere_evaluate


SYMMETRIC_RELATIONS = ("coreference", "isComparedTo")


def load_document(path: Path, document_id: str) -> dict[str, Any]:
    matches = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            document = json.loads(line)
            if document["doc_id"] == document_id:
                matches.append(document)
    if len(matches) != 1:
        raise ValueError(
            f"expected one {document_id!r} record in {path}, found {len(matches)}"
        )
    return matches[0]


def percent(value: float) -> float:
    return round(value * 100, 2)


def metric_summary(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "precision": metrics[f"{prefix}_precision"],
        "recall": metrics[f"{prefix}_recall"],
        "f1": metrics[f"{prefix}_f1"],
        "precision_percent": percent(metrics[f"{prefix}_precision"]),
        "recall_percent": percent(metrics[f"{prefix}_recall"]),
        "f1_percent": percent(metrics[f"{prefix}_f1"]),
        "tp": metrics[f"{prefix}_tp"],
        "fp": metrics[f"{prefix}_fp"],
        "fn": metrics[f"{prefix}_fn"],
        "gold": metrics[f"{prefix}_n_gold"],
        "predicted": metrics[f"{prefix}_n_pred"],
    }


def build_scorer_document(gold: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    sentence_count = len(gold["sentences"])
    offsets = []
    offset = 0
    for sentence in gold["sentences"]:
        offsets.append(offset)
        offset += len(sentence)

    predicted_ner: list[list[list[Any]]] = [[] for _ in range(sentence_count)]
    entity_by_id: dict[str, tuple[int, int, int, str]] = {}
    seen_spans: set[tuple[int, int, int]] = set()
    for index, entity in enumerate(prediction.get("entities", [])):
        entity_id = entity.get("id")
        sentence_id = entity.get("segment_id")
        start = entity.get("start")
        end = entity.get("end")
        label = entity.get("type")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError(f"entities[{index}] has invalid id")
        if entity_id in entity_by_id:
            raise ValueError(f"duplicate entity id {entity_id!r}")
        if not isinstance(sentence_id, int) or not 0 <= sentence_id < sentence_count:
            raise ValueError(f"entities[{index}] has invalid segment_id")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"entities[{index}] has non-integer span")
        if not 0 <= start <= end < len(gold["sentences"][sentence_id]):
            raise ValueError(f"entities[{index}] has out-of-range span")
        if not isinstance(label, str) or not label:
            raise ValueError(f"entities[{index}] has invalid type")
        span_key = (sentence_id, start, end)
        if span_key in seen_spans:
            raise ValueError(f"duplicate predicted span {span_key}")
        seen_spans.add(span_key)
        global_start = offsets[sentence_id] + start
        global_end = offsets[sentence_id] + end
        predicted_ner[sentence_id].append([global_start, global_end, label])
        entity_by_id[entity_id] = (
            sentence_id,
            global_start,
            global_end,
            label,
        )

    predicted_rel: list[list[list[Any]]] = [[] for _ in range(sentence_count)]
    seen_relations: set[tuple[Any, ...]] = set()
    for index, relation in enumerate(prediction.get("relations", [])):
        head_id = relation.get("head")
        tail_id = relation.get("tail")
        label = relation.get("type")
        if head_id not in entity_by_id or tail_id not in entity_by_id:
            raise ValueError(f"relations[{index}] references an unknown entity")
        if not isinstance(label, str) or not label:
            raise ValueError(f"relations[{index}] has invalid type")
        head_sentence, head_start, head_end, _ = entity_by_id[head_id]
        tail_sentence, tail_start, tail_end, _ = entity_by_id[tail_id]
        if head_sentence != tail_sentence:
            raise ValueError(f"relations[{index}] crosses sentence boundaries")
        row = (head_start, head_end, tail_start, tail_end, label)
        if row in seen_relations:
            continue
        seen_relations.add(row)
        predicted_rel[head_sentence].append(list(row))

    return {
        "doc_id": gold["doc_id"],
        "ner": gold["ner"],
        "relations": gold["relations"],
        "predicted_ner": predicted_ner,
        "predicted_rel": predicted_rel,
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", nargs="+", type=Path)
    parser.add_argument("--gold", type=Path, default=repo_root / "data" / "train.jsonl")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document_ids = []
    scorer_documents = []
    for prediction_path in args.prediction:
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        document_id = prediction.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"{prediction_path} has no valid document_id")
        gold = load_document(args.gold, document_id)
        document_ids.append(document_id)
        scorer_documents.append(build_scorer_document(gold, prediction))

    metrics = gsapere_evaluate(
        scorer_documents, sym_labels=SYMMETRIC_RELATIONS
    )
    document_metadata = (
        {"document_id": document_ids[0]}
        if len(document_ids) == 1
        else {"document_ids": document_ids}
    )
    result = {
        **document_metadata,
        "averaging": "micro",
        "matching": "exact token spans and case-sensitive labels",
        "scorer": {
            "package": "gsapere",
            "version": importlib.metadata.version("gsapere"),
            "function": "gsapere.evaluation.hgere.evaluate",
            "symmetric_relations": list(SYMMETRIC_RELATIONS),
        },
        "ner": metric_summary(metrics, "ner"),
        "rel": metric_summary(metrics, "re"),
        "rel_plus": metric_summary(metrics, "re+"),
        "upstream_metrics": metrics,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
