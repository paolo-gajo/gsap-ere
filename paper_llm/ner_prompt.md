# Task Introduction

Perform sentence-level named entity recognition for the GSAP-ERE scholarly
information extraction task. Extract every qualifying contiguous token span.

# Label Definitions

{{LABEL_DEFINITIONS}}

# Step-by-Step Instructions

1. Read the complete tokenized sentence.
2. Find every explicit named or generic GSAP-ERE entity mention.
3. Choose its exact inclusive zero-based token boundaries.
4. Assign exactly one best label to each unique span. Nested spans are allowed.
5. Return one JSON object with an `entities` array and no other text. Each item
   must contain integer `start`, integer `end`, and string `type`. Return an
   empty array when the sentence contains no entity.

# Full Article Context

{{FULL_ARTICLE_CONTEXT}}

# Few-Shot Examples

{{FEW_SHOT_EXAMPLES}}

# Main Input

Sentence tokens:

{{MAIN_INPUT}}

Return only `{"entities": [...]}`.
