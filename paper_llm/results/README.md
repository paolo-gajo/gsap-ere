# Paper-method LLM results

Runs use one directory level per varying setting:

```text
model=<model>/
  ner-examples=<count>/
    re-examples=<count>/
      full-context=<0|1>/
        ner-output=<indices|inline>/
          thinking=<0|1>/
            <document-id>/
```

Each document directory contains one sentence-level, two-stage LLM pipeline run:

- `prediction.json`: assembled entity and relation predictions;
- `scores.json`: exact metrics from `gsapere.evaluation.hgere.evaluate`;
- `run.json`: all six settings, software, hardware, input hashes, and reconstruction notes;
- `retrieval.json`: every selected training demonstration and similarity; and
- `trace.jsonl`: exact prompts, raw responses, timings, and parsing warnings.

Thinking-enabled runs also store each response's separate `raw_thinking` text
in `trace.jsonl`.

Zero-ICL runs store empty NER example lists and `null` RE examples in
`retrieval.json`; no retriever or training demonstration pool is used.

Five-shot RE runs store all five demonstrations for each candidate under the
`examples` field in `retrieval.json`; `example` remains as a compatibility
alias for the first item.

These results are methodologically separate from `zero_shot/results/`.
