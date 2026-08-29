import json
import random
from itertools import product
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, BertModel, BertPreTrainedModel


MODEL_NAME = "google-bert/bert-base-cased"
DATA_PATH = Path("data/train.jsonl")
VOCAB_PATH = Path("vocabulary.json")
OUTPUT_PATH = Path("checkpoints/bert-ner-re")
EPOCHS = 3
LEARNING_RATE = 2e-5
NEGATIVES_PER_POSITIVE = 5
MIN_NEGATIVES = 32


class JointBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.entity_classifier = nn.Linear(
            config.hidden_size * 2, config.num_entity_labels
        )
        self.relation_classifier = nn.Linear(
            config.hidden_size * 4, config.num_relation_labels
        )
        self.post_init()

    def span_vectors(self, hidden, spans):
        return torch.cat(
            (hidden[0, spans[:, 0]], hidden[0, spans[:, 1]]), dim=-1
        )

    def forward(self, input_ids, attention_mask, entity_spans, relation_pairs):
        hidden = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        entity_vectors = self.span_vectors(hidden, entity_spans)
        entity_logits = self.entity_classifier(self.dropout(entity_vectors))

        if relation_pairs.numel():
            heads = self.span_vectors(hidden, relation_pairs[:, :2])
            tails = self.span_vectors(hidden, relation_pairs[:, 2:])
            relation_logits = self.relation_classifier(
                self.dropout(torch.cat((heads, tails), dim=-1))
            )
        else:
            relation_logits = hidden.new_empty((0, self.config.num_relation_labels))

        return entity_logits, relation_logits


def load_examples():
    vocabulary = json.loads(VOCAB_PATH.read_text())
    examples = []
    entity_labels = set()
    relation_labels = set()
    max_entity_width = 1

    with DATA_PATH.open() as stream:
        for line in stream:
            document = json.loads(line)
            offset = 0
            for token_ids, entities, relations in zip(
                document["sentences"], document["ner"], document["relations"]
            ):
                local_entities = []
                for start, end, label in entities:
                    start, end = start - offset, end - offset
                    local_entities.append((start, end, label))
                    entity_labels.add(label)
                    max_entity_width = max(max_entity_width, end - start + 1)

                local_relations = []
                for hs, he, ts, te, label in relations:
                    local_relations.append(
                        (hs - offset, he - offset, ts - offset, te - offset, label)
                    )
                    relation_labels.add(label)

                examples.append(
                    {
                        "words": [vocabulary[token_id] for token_id in token_ids],
                        "entities": local_entities,
                        "relations": local_relations,
                    }
                )
                offset += len(token_ids)

    return (
        examples,
        sorted(entity_labels),
        sorted(relation_labels),
        max_entity_width,
    )


def wordpiece_boundaries(encoding, word_count):
    starts = [-1] * word_count
    ends = [-1] * word_count
    for piece, word in enumerate(encoding.word_ids()):
        if word is None:
            continue
        if starts[word] == -1:
            starts[word] = piece
        ends[word] = piece
    if any(position == -1 for position in starts):
        raise ValueError("BERT input exceeded its maximum sequence length")
    return starts, ends


def sample_negatives(negatives, positive_count):
    count = min(
        len(negatives), max(MIN_NEGATIVES, positive_count * NEGATIVES_PER_POSITIVE)
    )
    return random.sample(negatives, count)


def make_entity_batch(example, starts, ends, label_to_id, max_width, device):
    gold = {}
    for start, end, label in example["entities"]:
        gold.setdefault((start, end), set()).add(label_to_id[label])

    candidates = [
        (start, end)
        for start in range(len(example["words"]))
        for end in range(start, min(len(example["words"]), start + max_width))
    ]
    positives = list(gold)
    negatives = [span for span in candidates if span not in gold]
    spans = positives + sample_negatives(negatives, len(positives))
    random.shuffle(spans)

    piece_spans = torch.tensor(
        [(starts[start], ends[end]) for start, end in spans],
        dtype=torch.long,
        device=device,
    )
    targets = torch.zeros(
        (len(spans), len(label_to_id)), dtype=torch.float, device=device
    )
    for row, span in enumerate(spans):
        for label in gold.get(span, ()):
            targets[row, label] = 1.0
    return piece_spans, targets


def make_relation_batch(example, starts, ends, label_to_id, device):
    entity_spans = sorted({(start, end) for start, end, _ in example["entities"]})
    gold = {}
    for hs, he, ts, te, label in example["relations"]:
        gold.setdefault((hs, he, ts, te), set()).add(label_to_id[label])
    candidates = [(*head, *tail) for head, tail in product(entity_spans, repeat=2)]
    positives = list(gold)
    negatives = [pair for pair in candidates if pair not in gold]
    pairs = positives + sample_negatives(negatives, len(positives))
    random.shuffle(pairs)

    piece_pairs = torch.tensor(
        [
            (starts[hs], ends[he], starts[ts], ends[te])
            for hs, he, ts, te in pairs
        ],
        dtype=torch.long,
        device=device,
    ).reshape(-1, 4)
    targets = torch.zeros(
        (len(pairs), len(label_to_id)), dtype=torch.float, device=device
    )
    for row, pair in enumerate(pairs):
        for label in gold.get(pair, ()):
            targets[row, label] = 1.0
    return piece_pairs, targets


def main():
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    examples, entity_labels, relation_labels, max_width = load_examples()
    entity_to_id = {label: index for index, label in enumerate(entity_labels)}
    relation_to_id = {label: index for index, label in enumerate(relation_labels)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_entity_labels = len(entity_labels)
    config.num_relation_labels = len(relation_labels)
    config.entity_labels = entity_labels
    config.relation_labels = relation_labels
    config.max_entity_width = max_width

    model = JointBert.from_pretrained(MODEL_NAME, config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        random.shuffle(examples)
        total_loss = 0.0
        for example in examples:
            encoding = tokenizer(
                example["words"], is_split_into_words=True, return_tensors="pt"
            )
            if encoding["input_ids"].shape[1] > config.max_position_embeddings:
                raise ValueError("BERT input exceeded its maximum sequence length")
            starts, ends = wordpiece_boundaries(encoding, len(example["words"]))
            entity_spans, entity_targets = make_entity_batch(
                example, starts, ends, entity_to_id, max_width, device
            )
            relation_pairs, relation_targets = make_relation_batch(
                example, starts, ends, relation_to_id, device
            )

            optimizer.zero_grad(set_to_none=True)
            entity_logits, relation_logits = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
                entity_spans=entity_spans,
                relation_pairs=relation_pairs,
            )
            loss = nn.functional.binary_cross_entropy_with_logits(
                entity_logits, entity_targets
            )
            if relation_targets.numel():
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    relation_logits, relation_targets
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"epoch={epoch + 1} loss={total_loss / len(examples):.4f}")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)


if __name__ == "__main__":
    main()
