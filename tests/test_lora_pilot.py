from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lora_pilot
from paper_llm.inference import Entity


class FakeTokenizer:
    eos_token_id = 100_000

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        self.chat_template_calls = getattr(self, "chat_template_calls", 0) + 1
        self.assertions = (tokenize, add_generation_prompt, enable_thinking)
        return f"USER:{messages[0]['content']}\nASSISTANT:<think></think>\n"

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("special tokens must not be added twice")
        return [ord(character) for character in text]


class LoRAPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets, cls.eligible = lora_pilot.load_pilot_records()
        cls.examples = lora_pilot.sample_pilot_examples(
            cls.eligible, lora_pilot.DEFAULT_SEED
        )

    def test_target_is_excluded_wholesale(self):
        self.assertEqual(len(self.targets), 44)
        self.assertEqual(
            [record.sentence_id for record in self.targets], list(range(44))
        )
        self.assertNotIn(
            lora_pilot.TARGET_DOCUMENT_ID,
            {record.doc_id for record in self.eligible},
        )
        self.assertTrue(
            all(
                self.eligible[example.record_index].doc_id
                != lora_pilot.TARGET_DOCUMENT_ID
                for example in self.examples
            )
        )

    def test_sampling_counts_balance_and_uniqueness(self):
        self.assertEqual(len(self.examples), 200)
        self.assertEqual(
            Counter(example.task for example in self.examples),
            {"ner": 100, "re": 100},
        )
        ner_indices = [
            example.record_index for example in self.examples if example.task == "ner"
        ]
        re_examples = [example for example in self.examples if example.task == "re"]
        self.assertEqual(len(set(ner_indices)), 100)
        self.assertEqual(len({example.record_index for example in re_examples}), 100)
        self.assertEqual(
            Counter(
                "NIL" if example.pair.label == "NIL" else "positive"
                for example in re_examples
            ),
            {"positive": 50, "NIL": 50},
        )

    def test_sampling_is_deterministic(self):
        repeated = lora_pilot.sample_pilot_examples(
            self.eligible, lora_pilot.DEFAULT_SEED
        )
        self.assertEqual(self.examples, repeated)

    def test_gold_completions_are_json_and_training_prompts_are_zero_shot(self):
        ner_template = lora_pilot.NER_TEMPLATE.read_text(encoding="utf-8")
        re_template = lora_pilot.RE_TEMPLATE.read_text(encoding="utf-8")
        for example in self.examples:
            completion = lora_pilot.gold_completion(example, self.eligible)
            self.assertIsInstance(json.loads(completion), dict)
            prompt = lora_pilot.training_prompt(
                example, self.eligible, ner_template, re_template
            )
            self.assertNotIn("# Few-Shot Examples", prompt)
            self.assertNotIn("# Full Article Context", prompt)

    def test_completion_only_masking(self):
        tokenizer = FakeTokenizer()
        source = next(example for example in self.examples if example.task == "re")
        record = self.eligible[source.record_index]
        example = lora_pilot.PreparedExample(
            sample_index=1,
            task="re",
            stratum=lora_pilot.example_stratum(source),
            record_key=record.key,
            source=source,
        )
        encoded = lora_pilot.encode_training_example(
            tokenizer,
            example,
            self.eligible,
            lora_pilot.NER_TEMPLATE.read_text(encoding="utf-8"),
            lora_pilot.RE_TEMPLATE.read_text(encoding="utf-8"),
        )
        self.assertEqual(tokenizer.assertions, (False, True, False))
        self.assertEqual(tokenizer.chat_template_calls, 1)
        self.assertTrue(
            all(
                value == -100
                for value in encoded["labels"][: encoded["prompt_tokens"]]
            )
        )
        self.assertEqual(
            encoded["labels"][encoded["prompt_tokens"] :],
            encoded["input_ids"][encoded["prompt_tokens"] :],
        )
        self.assertEqual(encoded["input_ids"][-1], tokenizer.eos_token_id)

    def test_bare_prompts_are_exactly_x(self):
        ner_source = next(example for example in self.examples if example.task == "ner")
        ner_record = self.eligible[ner_source.record_index]
        self.assertEqual(
            lora_pilot.training_prompt(
                ner_source,
                self.eligible,
                "unused NER template",
                "unused RE template",
                bare=True,
            ),
            json.dumps(lora_pilot.indexed_tokens(ner_record), ensure_ascii=False),
        )

        re_source = next(example for example in self.examples if example.task == "re")
        re_record = self.eligible[re_source.record_index]
        self.assertEqual(
            lora_pilot.training_prompt(
                re_source,
                self.eligible,
                "unused NER template",
                "unused RE template",
                bare=True,
            ),
            json.dumps(
                lora_pilot.pair_input(
                    re_record,
                    re_source.pair.subject,
                    re_source.pair.object,
                ),
                ensure_ascii=False,
            ),
        )

    def test_bare_encoding_has_no_chat_wrapper(self):
        tokenizer = FakeTokenizer()
        source = next(example for example in self.examples if example.task == "re")
        record = self.eligible[source.record_index]
        example = lora_pilot.PreparedExample(
            sample_index=1,
            task="re",
            stratum=lora_pilot.example_stratum(source),
            record_key=record.key,
            source=source,
        )
        prompt = json.dumps(
            lora_pilot.pair_input(
                record,
                source.pair.subject,
                source.pair.object,
            ),
            ensure_ascii=False,
        )
        completion = lora_pilot.gold_completion(source, self.eligible)
        encoded = lora_pilot.encode_training_example(
            tokenizer,
            example,
            self.eligible,
            "unused NER template",
            "unused RE template",
            bare=True,
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        self.assertFalse(hasattr(tokenizer, "chat_template_calls"))
        self.assertEqual(encoded["input_ids"], prompt_ids + completion_ids)
        self.assertEqual(encoded["labels"], [-100] * len(prompt_ids) + completion_ids)

    def test_cli_accepts_bare_and_supported_model(self):
        with patch(
            "sys.argv",
            [
                "lora_pilot.py",
                "--output-dir",
                "/tmp/output",
                "--bare",
                "--model-id",
                "Qwen/Qwen3-14B-Base",
            ],
        ):
            args = lora_pilot.parse_args()
        self.assertTrue(args.bare)
        self.assertEqual(args.model_id, "Qwen/Qwen3-14B-Base")

    def test_bare_prepare_only_records_format_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            output_dir.mkdir()
            lora_pilot.run(
                SimpleNamespace(
                    resume_from=None,
                    prepare_only=True,
                    training_regime=lora_pilot.PILOT_REGIME,
                    seed=lora_pilot.DEFAULT_SEED,
                    bare=True,
                    model_id="Qwen/Qwen3-14B-Base",
                ),
                output_dir,
            )
            run = json.loads((output_dir / "run.json").read_text())
            manifest = json.loads((output_dir / "sample_manifest.json").read_text())
            self.assertEqual(run["status"], "prepared")
            self.assertEqual(run["training_format"], lora_pilot.BARE_TRAINING_FORMAT)
            self.assertTrue(run["training"]["bare"])
            self.assertEqual(run["model"]["id"], "Qwen/Qwen3-14B-Base")
            self.assertEqual(
                run["model"]["revision"],
                lora_pilot.MODEL_REVISIONS["Qwen/Qwen3-14B-Base"],
            )
            self.assertEqual(
                manifest["training_format"], lora_pilot.BARE_TRAINING_FORMAT
            )
            with (output_dir / "training_records.jsonl").open() as stream:
                self.assertTrue(
                    all(
                        json.loads(line)["training_format"]
                        == lora_pilot.BARE_TRAINING_FORMAT
                        for line in stream
                    )
                )

    def test_pilot_schedule_is_400_true_batch8_updates(self):
        prepared = []
        for sample_index, source in enumerate(self.examples, start=1):
            record = self.eligible[source.record_index]
            prepared.append(
                lora_pilot.PreparedExample(
                    sample_index=sample_index,
                    task=source.task,
                    stratum=lora_pilot.example_stratum(source),
                    record_key=record.key,
                    source=source,
                )
            )
        schedule = lora_pilot.build_training_schedule(
            prepared, lora_pilot.DEFAULT_SEED, lora_pilot.PILOT_REGIME
        )
        self.assertEqual(len(schedule), 400)
        self.assertEqual(Counter(batch.task for batch in schedule), {"ner": 200, "re": 200})
        self.assertTrue(all(len(batch.rows) == 8 for batch in schedule))
        self.assertTrue(
            all(
                len({row.sample_index for row in batch.rows}) == len(batch.rows)
                for batch in schedule
            )
        )
        self.assertEqual(
            set(
                Counter(
                    row.sample_index for batch in schedule for row in batch.rows
                ).values()
            ),
            {16},
        )
        self.assertEqual(
            Counter(row.stratum for batch in schedule for row in batch.rows),
            {"ner": 1600, "positive": 800, "nil": 800},
        )

    def test_full_pass_counts_and_drops_terminal_remainder(self):
        sources = lora_pilot.full_pass_examples(
            self.eligible, lora_pilot.DEFAULT_SEED
        )
        prepared = [
            lora_pilot.PreparedExample(
                sample_index=index,
                task=source.task,
                stratum=lora_pilot.example_stratum(source),
                record_key=self.eligible[source.record_index].key,
                source=source,
            )
            for index, source in enumerate(sources, start=1)
        ]
        schedule = lora_pilot.build_training_schedule(
            prepared, lora_pilot.DEFAULT_SEED, lora_pilot.FULL_PASS_REGIME
        )
        self.assertEqual(Counter(source.task for source in sources), {"ner": 20712, "re": 189786})
        self.assertEqual(Counter(source.task for source in sources if source.task == "re" and source.pair.label != "NIL"), {"re": 30276})
        self.assertEqual(
            sum(
                source.task == "re" and source.pair.label == "NIL"
                for source in sources
            ),
            159510,
        )
        self.assertEqual(len(schedule), 26312)
        self.assertEqual(Counter(batch.task for batch in schedule), {"ner": 2589, "re": 23723})
        self.assertEqual(Counter(len(batch.rows) for batch in schedule), {8: 26312})
        self.assertTrue(
            all({row.task for row in batch.rows} == {batch.task} for batch in schedule)
        )
        full_presentations = Counter(
            row.sample_index for batch in schedule for row in batch.rows
        )
        self.assertEqual(
            len(full_presentations),
            210496,
        )
        self.assertEqual(set(full_presentations.values()), {1})
        dropped = [
            row for row in prepared if row.sample_index not in full_presentations
        ]
        self.assertEqual(
            [(row.sample_index, row.stratum) for row in dropped],
            [(57757, "nil"), (118283, "nil")],
        )

    def test_padding_and_outer_json_fence(self):
        rows = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 3],
            },
            {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "labels": [-100, 5],
            },
        ]
        padded = lora_pilot.padded_training_rows(rows, pad_token_id=99)
        self.assertEqual(padded["input_ids"][1], [4, 5, 99])
        self.assertEqual(padded["attention_mask"][1], [1, 1, 0])
        self.assertEqual(padded["labels"][1], [-100, 5, -100])
        parsed, error = lora_pilot.parse_generated_json(
            '```json\n{"entities": []}\n```'
        )
        self.assertEqual(parsed, {"entities": []})
        self.assertIsNone(error)
        parsed, error = lora_pilot.parse_generated_json(
            'result:\n```json\n{"entities": []}\n```'
        )
        self.assertIsNone(parsed)
        self.assertIsNotNone(error)

    def test_checkpoint_write_is_complete_and_retains_two(self):
        class FakeCuda:
            @staticmethod
            def get_rng_state_all():
                return [b"cuda-rng"]

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def get_rng_state():
                return b"cpu-rng"

            @staticmethod
            def save(value, path):
                with Path(path).open("wb") as stream:
                    pickle.dump(value, stream)

        class FakeModel:
            @staticmethod
            def save_pretrained(path, *, safe_serialization):
                self = Path(path)
                self.mkdir(parents=True)
                (self / "adapter_config.json").write_text("{}")
                (self / "adapter_model.safetensors").write_bytes(b"weights")

        class FakeStateful:
            @staticmethod
            def state_dict():
                return {"state": "ok"}

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            for step in (1000, 2000, 3000):
                lora_pilot.save_training_checkpoint(
                    FakeTorch,
                    FakeModel,
                    FakeStateful,
                    FakeStateful,
                    output_dir,
                    lora_pilot.FULL_PASS_REGIME,
                    lora_pilot.TEMPLATED_TRAINING_FORMAT,
                    lora_pilot.MODEL_ID,
                    lora_pilot.MODEL_REVISION,
                    "schedule-sha256",
                    step,
                    [],
                    Counter(),
                    Counter(),
                    Counter(),
                )
            checkpoint_root = output_dir / "checkpoints"
            self.assertEqual(
                (checkpoint_root / "LATEST").read_text().strip(),
                "step-00003000",
            )
            self.assertEqual(
                sorted(path.name for path in checkpoint_root.glob("step-*")),
                ["step-00002000", "step-00003000"],
            )
            self.assertTrue(
                (checkpoint_root / "step-00003000" / "COMPLETE").is_file()
            )

    def test_prediction_shape_and_symmetric_canonicalization(self):
        target = self.targets[0]
        entities = [Entity(0, 0, "Method"), Entity(1, 1, "Method")]
        prediction = lora_pilot.build_prediction(
            [target], [entities], [(0, 1, 0, "coreference")]
        )
        self.assertEqual(prediction["document_id"], lora_pilot.TARGET_DOCUMENT_ID)
        self.assertEqual(len(prediction["entities"]), 2)
        self.assertEqual(
            prediction["relations"],
            [{"head": "s0e0", "type": "coreference", "tail": "s0e1"}],
        )


if __name__ == "__main__":
    unittest.main()
