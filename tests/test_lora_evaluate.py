from __future__ import annotations

import unittest

import lora_evaluate


class LoRAEvaluateTests(unittest.TestCase):
    def test_loads_exactly_three_requested_test_documents_in_order(self):
        document_ids = [
            "10018_2208_04405.txt",
            "00050_1406_2661.txt",
            "10047_2005_06182.txt",
        ]
        documents = lora_evaluate.load_target_documents(
            lora_evaluate.DEFAULT_TARGET_DATA,
            document_ids,
        )
        self.assertEqual(list(documents), document_ids)
        self.assertEqual(
            {document_id: len(records) for document_id, records in documents.items()},
            {
                "10018_2208_04405.txt": 193,
                "00050_1406_2661.txt": 113,
                "10047_2005_06182.txt": 86,
            },
        )

    def test_rejects_wrong_count_duplicates_and_paths(self):
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            lora_evaluate.load_target_documents(
                lora_evaluate.DEFAULT_TARGET_DATA,
                ["10018_2208_04405.txt"],
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            lora_evaluate.load_target_documents(
                lora_evaluate.DEFAULT_TARGET_DATA,
                [
                    "10018_2208_04405.txt",
                    "10018_2208_04405.txt",
                    "10047_2005_06182.txt",
                ],
            )
        with self.assertRaisesRegex(ValueError, "plain filenames"):
            lora_evaluate.load_target_documents(
                lora_evaluate.DEFAULT_TARGET_DATA,
                [
                    "../10018_2208_04405.txt",
                    "00050_1406_2661.txt",
                    "10047_2005_06182.txt",
                ],
            )


if __name__ == "__main__":
    unittest.main()
