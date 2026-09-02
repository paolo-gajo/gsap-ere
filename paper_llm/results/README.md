# Paper-method LLM results

Each directory here contains one sentence-level, two-stage LLM pipeline run:

- `prediction.json`: assembled entity and relation predictions;
- `scores.json`: exact metrics from `gsapere.evaluation.hgere.evaluate`;
- `run.json`: model, software, hardware, input hashes, and reconstruction notes;
- `retrieval.json`: every selected training demonstration and similarity; and
- `trace.jsonl`: exact prompts, raw responses, timings, and parsing warnings.

Thinking-enabled runs also store each response's separate `raw_thinking` text
in `trace.jsonl`.

Zero-ICL runs store empty NER example lists and `null` RE examples in
`retrieval.json`; no retriever or training demonstration pool is used.

These results are methodologically separate from `zero_shot/results/`.
