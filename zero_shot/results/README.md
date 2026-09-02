# Results

Each inference run has its own directory:

- `prediction.json`: the model's unedited final answer in the evaluator's input format.
- `scores.json`: exact micro NER, REL, and REL+ scores from `../evaluate.py`.
- `run.json`: available provenance, model, backend, hardware, prompt hash, generation settings, and timings.

The Qwen run is a custom one-prompt, full-article zero-shot experiment. It is not a reproduction of the paper's unreleased sentence-level, two-stage LLM pipeline.

Score any run from the repository root with:

    python3 zero_shot/evaluate.py zero_shot/results/<run-id>/prediction.json

Do not repair model predictions before scoring. Keep malformed responses as failed runs.
