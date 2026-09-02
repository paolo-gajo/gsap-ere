# GSAP-ERE paper LLM pipeline

This directory is separate from `zero_shot/`.

- `zero_shot/` is the custom full-article, zero-shot benchmark used for the
  hosted-model comparison.
- `paper_llm/` reconstructs only the LLM prompting pipeline described in the
  GSAP-ERE paper. It does not use PL-Marker, HGERE, their checkpoints, or any
  supervised training code.

## What follows the paper

The runner uses:

- sentence-level inference;
- a two-stage pipeline: NER, then RE over every ordered pair of predicted
  entities;
- 10 NER demonstrations and one RE demonstration;
- dynamic `similar+diverse` demonstration selection from the training split;
- `sentence-transformers/multi-qa-mpnet-base-cos-v1` cosine similarity;
- Ollama with a quantized model;
- temperature zero; and
- the `gsapere==0.2.4` authors' package for exact NER, RE, and RE+ scoring.

## Reconstruction boundary

The paper points to LLM code and Appendix A.3 prompts that are not present in
the public project repository, project page, dataset deposit, PyPI package, or
arXiv source. The exact prompt text and exact unique-label re-ranking algorithm
therefore cannot be copied.

The reconstruction keeps the paper's five named prompt sections. For NER it
greedily re-ranks the 100 most similar training sentences to maximize newly
covered entity labels, breaking ties by cosine similarity. For RE, where
`k = 1` makes set diversity inapplicable, it retrieves the most similar
training-sentence pair with the same ordered entity-type signature when one is
available. These choices and every selected example are saved in each run.

The selected article (`00016_2106_09462.txt`) belongs to the training split.
It is excluded wholesale from the demonstration pool so its gold annotations
cannot leak into prompts. Consequently this one-article Qwen3.8 run is not a
reproduction of the paper's test-set Qwen2.5/LLaMA numbers.

## John

From the repository root:

```bash
bash paper_llm/run_john_qwen.sh
```

The script bootstraps an isolated environment, starts Ollama, resumes the model
download if needed, runs the pipeline, and writes the authors' metrics to:

```text
paper_llm/results/qwen3.8-27b-ollama-john-00016_2106_09462/
```

Inference is resumable at the individual NER or RE prompt level.

To run the same pipeline with Qwen3.8 thinking enabled:

```bash
THINK=1 bash paper_llm/run_john_qwen.sh
```

This writes to the separate
`paper_llm/results/qwen3.8-27b-ollama-john-thinking-00016_2106_09462/`
directory. Thinking text is retained in `trace.jsonl`. The generation caps are
raised from 2,048/64 to 4,096/4,096 tokens for NER/RE so the thinking tokens do
not consume the short structured-answer allowance.
