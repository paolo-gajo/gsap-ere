# Task Introduction

Perform sentence-level relation classification for one ordered pair of
GSAP-ERE entity mentions. Assign exactly one relation label or `NIL`.

# Label Definitions

{{LABEL_DEFINITIONS}}

# Step-by-Step Instructions

1. Read the complete sentence and the marked ordered pair.
2. Treat `subject` as the directed source and `object` as the directed target.
3. Select a relation only when the sentence states it for this exact pair.
4. Use `NIL` when no listed relation holds in this direction.
5. Return one JSON object with only a `label` field and no other text.
{{OPTIONAL_SECTIONS}}
# Main Input

{{MAIN_INPUT}}

Return only `{"label": "RELATION_OR_NIL"}`.
