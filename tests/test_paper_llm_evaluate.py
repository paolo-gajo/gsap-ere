from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


if importlib.util.find_spec("gsapere") is None:
    gsapere = ModuleType("gsapere")
    evaluation = ModuleType("gsapere.evaluation")
    hgere = ModuleType("gsapere.evaluation.hgere")
    hgere.evaluate = None
    gsapere.evaluation = evaluation
    evaluation.hgere = hgere
    sys.modules["gsapere"] = gsapere
    sys.modules["gsapere.evaluation"] = evaluation
    sys.modules["gsapere.evaluation.hgere"] = hgere

from paper_llm import evaluate


def fake_metrics() -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for prefix in ("ner", "re", "re+"):
        metrics.update(
            {
                f"{prefix}_precision": 0.5,
                f"{prefix}_recall": 0.25,
                f"{prefix}_f1": 1 / 3,
                f"{prefix}_tp": 1,
                f"{prefix}_fp": 1,
                f"{prefix}_fn": 3,
                f"{prefix}_n_gold": 4,
                f"{prefix}_n_pred": 2,
            }
        )
    return metrics


class EvaluateTests(unittest.TestCase):
    def run_evaluate(self, document_ids: list[str]):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.jsonl"
            gold_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "doc_id": document_id,
                            "sentences": [[document_id]],
                            "ner": [[]],
                            "relations": [[]],
                        }
                    )
                    + "\n"
                    for document_id in document_ids
                ),
                encoding="utf-8",
            )
            prediction_paths = []
            for document_id in document_ids:
                prediction_path = root / f"{document_id}.json"
                prediction_path.write_text(
                    json.dumps(
                        {
                            "document_id": document_id,
                            "entities": [],
                            "relations": [],
                        }
                    ),
                    encoding="utf-8",
                )
                prediction_paths.append(prediction_path)
            output_path = root / "scores.json"
            argv = [
                "evaluate.py",
                *(str(path) for path in prediction_paths),
                "--gold",
                str(gold_path),
                "--output",
                str(output_path),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    evaluate.importlib.metadata,
                    "version",
                    return_value="test-version",
                ),
                patch.object(
                    evaluate,
                    "gsapere_evaluate",
                    return_value=fake_metrics(),
                ) as scorer,
            ):
                evaluate.main()
            return json.loads(output_path.read_text(encoding="utf-8")), scorer

    def test_single_prediction_retains_document_id(self):
        result, scorer = self.run_evaluate(["doc-1"])

        self.assertEqual(result["document_id"], "doc-1")
        self.assertNotIn("document_ids", result)
        scorer.assert_called_once()
        self.assertEqual(
            [document["doc_id"] for document in scorer.call_args.args[0]],
            ["doc-1"],
        )

    def test_multiple_predictions_are_scored_together(self):
        result, scorer = self.run_evaluate(["doc-1", "doc-2", "doc-3"])

        self.assertEqual(result["document_ids"], ["doc-1", "doc-2", "doc-3"])
        self.assertNotIn("document_id", result)
        scorer.assert_called_once()
        self.assertEqual(
            [document["doc_id"] for document in scorer.call_args.args[0]],
            ["doc-1", "doc-2", "doc-3"],
        )
        self.assertEqual(
            scorer.call_args.kwargs,
            {"sym_labels": evaluate.SYMMETRIC_RELATIONS},
        )


if __name__ == "__main__":
    unittest.main()
