# GSAP-ERE zero-shot extraction

You are given one scholarly article represented as ordered tokenized segments. Extract every GSAP-ERE entity mention and every explicitly stated GSAP-ERE relation.

This is zero-shot extraction. Do not explain your decisions. Do not use outside knowledge to add facts that the text does not state.

## Entity types

- **MLModel**: an explicitly named machine-learning model. For a neural model, the mention denotes a concrete executable model/resource rather than only an abstract architecture.
- **MLModelGeneric**: an unnamed or informal mention of one or more machine-learning models, such as an author's model or a group of models.
- **ModelArchitecture**: a named architecture, conceptual model type, or architectural component. Use this when the context discusses structure, a backbone, or a component rather than a concrete executable model.
- **Method**: a method, technique, algorithmic procedure, conceptualization, representation or embedding, learning paradigm, or non-architectural model component such as a loss or optimizer.
- **Dataset**: an explicitly named dataset or dataset acronym.
- **DatasetGeneric**: an unnamed or informal mention of one or more concrete datasets. This also covers spans that express a dataset's size or instance type when those spans participate in `size` or `hasInstanceType`.
- **DataSource**: an explicitly named non-static or unstable source from which data are obtained, such as a website or social platform.
- **ReferenceLink**: an in-text citation marker that can be linked to an item in the bibliography.
- **Task**: the name of a specific machine-learning task or a named collection of related tasks. Do not label a merely descriptive process.
- **URL**: a literal URL in the text.

## Entity-boundary rules

- Spans use the supplied tokens exactly. Indices are zero-based, local to one segment, and inclusive at both ends.
- For a named entity, exclude articles and a following generic noun when those tokens are not part of the name.
- For a generic mention, include necessary determiners, modifiers, and the generic head noun. Include intervening modifiers that are part of the noun phrase.
- Include an adjectival modifier only when removing it changes which entity is denoted. Adjectival modifiers normally remain inside generic mentions.
- Treat unnamed plural mentions as generic mentions.
- If a full name and an acronym are separately tokenized, extract them as separate mentions.
- Nested or overlapping spans are allowed when the definitions require them.
- Return at most one entity object for each unique `(segment_id, start, end)` span. Choose the single best type for that span.

## Relation types and direction

In every arrow below, the left argument is `head` and the right argument is `tail`.

- **appliedTo**: MLModel, MLModelGeneric, Method, or ModelArchitecture -> Task. The artifact is applied to the task.
- **architecture**: MLModel or MLModelGeneric -> ModelArchitecture when the latter is its architecture/backbone/component; or whole ModelArchitecture -> component ModelArchitecture.
- **benchmarkFor**: Dataset or DatasetGeneric -> Task. The dataset is used as a benchmark for the task.
- **citation**: scholarly entity -> ReferenceLink. The citation supports or identifies the nearby entity.
- **coreference**: compatible entity mention <-> compatible entity mention. Both mentions denote exactly the same entity. This relation is symmetric.
- **evaluatedOn**: MLModel, MLModelGeneric, or Method -> Dataset or DatasetGeneric. Evaluation or testing was actually performed on the dataset.
- **generatedBy**: Dataset or DatasetGeneric -> Method, MLModel, or MLModelGeneric. The dataset is a product of the method/model.
- **hasInstanceType**: Dataset or DatasetGeneric -> DatasetGeneric. The tail span states the dataset's item/instance type or another type property.
- **isBasedOn**: derived or extended MLModel/MLModelGeneric -> source MLModel/MLModelGeneric.
- **isComparedTo**: compatible entity mention <-> compatible entity mention. The entities are explicitly compared. This relation is symmetric.
- **isHyponymOf**: narrower subclass or instance -> broader compatible class.
- **isPartOf**: part, split, or member -> whole or collection, between compatible dataset, method, or model mentions.
- **size**: Dataset or DatasetGeneric -> DatasetGeneric. The tail span states the dataset's size or volume.
- **sourcedFrom**: Dataset or DatasetGeneric -> DataSource.
- **trainedOn**: MLModel, MLModelGeneric, or Method -> Dataset or DatasetGeneric. Training was actually performed on the dataset.
- **transformedFrom**: derived Dataset or DatasetGeneric -> source Dataset or DatasetGeneric.
- **usedFor**: Method -> MLModel, MLModelGeneric, or ModelArchitecture. The method supplies functionality used by the model or architecture.
- **url**: scholarly entity -> URL. The URL gives the external resource location for the entity.

## Relation rules

- Predict a relation only when the text states an existing fact or a general factual rule. Exclude proposed future use, plans, hypotheticals, possibilities, and unsupported world knowledge.
- Both endpoints must be entity mentions that you return in `entities`.
- Both endpoints must occur in the same segment.
- Return at most one instance of a given relation type for the same ordered pair.
- Direction matters except for `coreference` and `isComparedTo`.
- For either symmetric relation, put the endpoint with the lexicographically smaller `(segment_id, start, end)` tuple in `head` so output is deterministic.

## Required output

Return exactly one valid JSON object and nothing else: no Markdown fence, prose, comments, or trailing commas.

The top-level object must have exactly these fields:

    {
      "document_id": "copy the supplied document_id",
      "entities": [],
      "relations": []
    }

Each entity object must have exactly these fields:

    {
      "id": "a unique string used by relations",
      "segment_id": 0,
      "start": 0,
      "end": 0,
      "type": "one entity type listed above",
      "text": "the selected tokens joined by one ASCII space"
    }

Each relation object must have exactly these fields:

    {
      "head": "id of the source entity",
      "type": "one relation type listed above",
      "tail": "id of the target entity"
    }

Sort entities by `(segment_id, start, end, type, id)`. Sort relations by `(head, type, tail)`. Use empty arrays if nothing qualifies. Do not emit null values or confidence scores.

## Article input

The `text` field is a readable rendering. The authoritative coordinates are in `indexed_tokens`, where every pair is `[local_token_index, exact_token_string]`.

{{MODEL_INPUT_JSON}}
