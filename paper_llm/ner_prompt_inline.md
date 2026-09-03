# Task Introduction

Perform sentence-level named entity recognition for the GSAP-ERE scholarly
information extraction task. Extract every qualifying contiguous text span.

# Label Definitions

{{LABEL_DEFINITIONS}}

# Step-by-Step Instructions

1. Read the complete sentence.
2. Find every explicit named or generic GSAP-ERE entity mention.
3. Copy the sentence exactly and enclose each entity with paired markers.
4. Start an entity with `[[ID:TYPE]]` and end it with `[[/ID]]`. Use a unique
   identifier such as `e1` for each entity. Identifiers are not text positions.
5. Put markers directly against the first and last character of the entity. Do
   not change, delete, reorder, or add any text outside the markers.
6. Assign exactly one best label to each unique span. Nested and crossing spans
   are allowed; pair their boundaries by identifier.
7. Return only the marked sentence, without JSON, a code fence, or an
   explanation. If there is no entity, return the unchanged sentence.
{{OPTIONAL_SECTIONS}}
# Main Input

Sentence:

{{MAIN_INPUT}}

Return only the marked sentence.
