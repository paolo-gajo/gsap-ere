from __future__ import annotations

import json
import unittest
from collections import Counter

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
        example = lora_pilot.PreparedExample(
            optimizer_step=1,
            task="re",
            record_key="example:0",
            prompt="classify this pair",
            completion='{"label":"NIL"}',
        )
        encoded = lora_pilot.encode_training_example(tokenizer, example)
        self.assertEqual(tokenizer.assertions, (False, True, False))
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
