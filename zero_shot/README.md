# Zero-shot GSAP-ERE smoke test

This directory contains one annotation-free GSAP-ERE document, a pasteable zero-shot prompt, and an exact-match scorer.

This is the custom full-article experiment. The paper's sentence-level,
few-shot LLM pipeline is reconstructed separately in `paper_llm/`.

## Files

- `article.txt`: readable plain text with all 44 retained dataset segments and no labels or relations.
- `model_input.json`: the same text with authoritative local token indices; no labels or relations.
- `prompt.md`: the complete prompt to paste into a model. It includes the task definitions, output schema, and `model_input.json`.
- `prompt_template.md`: source template used to build `prompt.md`.
- `prediction_template.json`: an empty but valid response, useful for testing the scorer.
- `evaluate.py`: dependency-free scorer for exact micro NER, REL, and REL+.
- `inference.py`: custom Ollama client that records the raw prediction and run provenance.
- `run_john_qwen.sh`: John-specific Qwen3.8 runner using Ollama's GPU-backed llama.cpp engine.
- `results/`: one directory per model run, containing predictions, scores, and provenance.

Regenerate the article and prompt from the repository data with:

    python3 zero_shot/build_example.py

## Run a model

1. Start a fresh chat and disable browsing, search, tools, memories, and project context where the product permits it.
2. Paste all of `prompt.md` as one user message.
3. Save the model's response unchanged as a JSON file. Do not manually repair spans, labels, or relations.
4. Record the product, exact model/version shown, date, settings, and raw response.
5. Use the same prompt and protocol for every model.

For APIs, use deterministic decoding where supported (for example temperature 0) and archive the exact request and response. Consumer chat products are not necessarily reproducible even with the same visible prompt.

## Score a response

For example:

    python3 zero_shot/evaluate.py zero_shot/results/<run-id>/prediction.json

The scorer accepts raw JSON and also tolerates one enclosing `json` Markdown fence. Other malformed output fails instead of being silently repaired.

The reported exact micro metrics are:

- **NER**: a predicted entity is correct when its segment, inclusive token span, and type match. The gold denominator is unique span positions. If a gold span has multiple valid labels, one matching predicted label is sufficient.
- **REL**: the ordered pair of exact endpoint spans and the relation label must match; endpoint entity types are ignored. This is called RE in the GSAP-ERE paper.
- **REL+**: REL must match and the predicted types of both endpoint entities must also be valid gold types.

`coreference` and `isComparedTo` are treated as symmetric; all other relation directions matter. Precision, recall, and F1 are computed from corpus-level TP, FP, and FN counts. For this document the gold denominators are 124 unique entity spans (125 annotation rows because one span has two labels) and 59 relations.

## Label-set discrepancy

The annotation guideline and paper define 10 entity types and 18 relation types. The released train/dev/test JSONL files also contain two undocumented relation labels: `processed` and `versionOf`. Neither occurs in this selected document. The prompt therefore exposes only the 18 defined relations, while the scorer recognizes all 20 labels so it remains faithful to the released data.

## Interpretation

This is a useful end-to-end smoke test, not a defensible model ranking. The document comes from the public training split, so a commercial model may have encountered it or related material. It is also only one document. A serious comparison should use many sequestered documents, keep the protocol fixed, retain invalid outputs as failures, and aggregate micro counts before computing F1.

## Qwen3.8 on John

The original GSAP-ERE LLM prompting implementation is not present in the authors' GitHub repository, dataset deposit, paper source, or current `gsapere` package. The runner here is therefore explicitly custom and evaluates the same full-article prompt used for the other preliminary result.

After installing Ollama and pulling the repository on John, run:

    bash zero_shot/run_john_qwen.sh

Defaults: `qwen3.8:27b`, high thinking effort, temperature 0, seed 0, 65,536-token context, JSON-Schema-constrained output, flash attention, and an 8-bit KV cache. Override `MODEL`, `RESULT_DIR`, or `OLLAMA_BASE_URL` through environment variables. Set `OVERWRITE=1` to replace an existing Qwen result.
