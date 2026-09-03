#!/usr/bin/env python3
"""Run the fixed Qwen3.8-27B GSAP-ERE LoRA pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from paper_llm.inference import (
    DEFAULT_DOCUMENT_ID,
    DEFAULT_RETRIEVER,
    ENTITY_ORDER,
    RELATION_ORDER,
    SYMMETRIC_RELATIONS,
    Entity,
    PairExample,
    SentenceRecord,
    build_ner_prompt,
    build_pair_examples,
    build_re_prompt,
    entity_dict,
    gpu_inventory,
    load_records,
    sanitize_ner,
    sanitize_re,
    select_ner_examples,
    select_re_examples,
    sha256_file,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
RETRIEVER_ID = DEFAULT_RETRIEVER
RETRIEVER_REVISION = "d51b22a1dfa8184e9258074e56e2875e50612dca"
TARGET_DOCUMENT_ID = DEFAULT_DOCUMENT_ID

DEFAULT_SEED = 42
NER_STEPS = 100
RE_POSITIVE_STEPS = 50
RE_NIL_STEPS = 50
RE_STEPS = RE_POSITIVE_STEPS + RE_NIL_STEPS
TOTAL_STEPS = NER_STEPS + RE_STEPS

LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 16
MAX_TRAIN_TOKENS = 4096
NER_SHOTS = 10
RE_SHOTS = 1
RETRIEVAL_POOL_SIZE = 100
NER_MAX_NEW_TOKENS = 2048
RE_MAX_NEW_TOKENS = 64

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_DATA = REPO_ROOT / "data" / "train.jsonl"
VOCABULARY = REPO_ROOT / "vocabulary.json"
NER_TEMPLATE = REPO_ROOT / "paper_llm" / "ner_prompt.md"
RE_TEMPLATE = REPO_ROOT / "paper_llm" / "re_prompt.md"


@dataclass(frozen=True)
class PilotExample:
    task: str
    record_index: int
    pair: PairExample | None = None


@dataclass(frozen=True)
class PreparedExample:
    optimizer_step: int
    task: str
    record_key: str
    prompt: str
    completion: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def initialize_output_dir(path: Path) -> Path:
    output_dir = path.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_pilot_records() -> tuple[list[SentenceRecord], list[SentenceRecord]]:
    vocabulary = json.loads(VOCABULARY.read_text(encoding="utf-8"))
    records = load_records(TRAIN_DATA, vocabulary)
    targets = sorted(
        (record for record in records if record.doc_id == TARGET_DOCUMENT_ID),
        key=lambda record: record.sentence_id,
    )
    eligible = [record for record in records if record.doc_id != TARGET_DOCUMENT_ID]
    if len(targets) != 44:
        raise RuntimeError(
            f"expected 44 target sentences for {TARGET_DOCUMENT_ID}, found {len(targets)}"
        )
    if [record.sentence_id for record in targets] != list(range(44)):
        raise RuntimeError("target sentence IDs are not exactly 0 through 43")
    if any(record.doc_id == TARGET_DOCUMENT_ID for record in eligible):
        raise AssertionError("target document survived the training-data filter")
    return targets, eligible


def _sorted_pairs(pairs: Iterable[PairExample]) -> list[PairExample]:
    return sorted(
        pairs,
        key=lambda pair: (
            pair.subject.start,
            pair.subject.end,
            pair.subject.type,
            pair.object.start,
            pair.object.end,
            pair.object.type,
            pair.label,
        ),
    )


def sample_pilot_examples(
    eligible: list[SentenceRecord], seed: int
) -> list[PilotExample]:
    if len(eligible) < NER_STEPS:
        raise RuntimeError("not enough eligible sentences for NER sampling")

    ner_rng = random.Random(f"{seed}:ner")
    positive_rng = random.Random(f"{seed}:re-positive")
    nil_rng = random.Random(f"{seed}:re-nil")
    order_rng = random.Random(f"{seed}:order")

    ner_indices = ner_rng.sample(range(len(eligible)), NER_STEPS)
    pair_examples, _ = build_pair_examples(eligible)
    positive_by_record: dict[int, list[PairExample]] = defaultdict(list)
    nil_by_record: dict[int, list[PairExample]] = defaultdict(list)
    for pair in pair_examples:
        destination = nil_by_record if pair.label == "NIL" else positive_by_record
        destination[pair.record_index].append(pair)

    if len(positive_by_record) < RE_POSITIVE_STEPS:
        raise RuntimeError("not enough distinct positive RE sentences")
    positive_record_indices = positive_rng.sample(
        sorted(positive_by_record), RE_POSITIVE_STEPS
    )
    positive_record_set = set(positive_record_indices)
    nil_candidates = sorted(set(nil_by_record) - positive_record_set)
    if len(nil_candidates) < RE_NIL_STEPS:
        raise RuntimeError("not enough distinct NIL RE sentences after positive sampling")
    nil_record_indices = nil_rng.sample(nil_candidates, RE_NIL_STEPS)

    examples = [PilotExample("ner", index) for index in ner_indices]
    for index in positive_record_indices:
        pair = positive_rng.choice(_sorted_pairs(positive_by_record[index]))
        examples.append(PilotExample("re", index, pair))
    for index in nil_record_indices:
        pair = nil_rng.choice(_sorted_pairs(nil_by_record[index]))
        examples.append(PilotExample("re", index, pair))
    order_rng.shuffle(examples)

    task_counts = Counter(example.task for example in examples)
    re_labels = Counter(
        "NIL" if example.pair is not None and example.pair.label == "NIL" else "positive"
        for example in examples
        if example.task == "re"
    )
    re_record_indices = [
        example.record_index for example in examples if example.task == "re"
    ]
    if task_counts != {"ner": NER_STEPS, "re": RE_STEPS}:
        raise AssertionError(f"unexpected task counts: {task_counts}")
    if re_labels != {"positive": RE_POSITIVE_STEPS, "NIL": RE_NIL_STEPS}:
        raise AssertionError(f"unexpected RE balance: {re_labels}")
    if len(set(ner_indices)) != NER_STEPS:
        raise AssertionError("NER source sentences are not unique")
    if len(set(re_record_indices)) != RE_STEPS:
        raise AssertionError("RE source sentences are not unique")
    if any(eligible[example.record_index].doc_id == TARGET_DOCUMENT_ID for example in examples):
        raise AssertionError("target document leaked into sampled training examples")
    return examples


def gold_completion(example: PilotExample, eligible: list[SentenceRecord]) -> str:
    record = eligible[example.record_index]
    if example.task == "ner":
        payload = {
            "entities": [
                {"start": entity.start, "end": entity.end, "type": entity.type}
                for entity in record.entities
            ]
        }
    elif example.task == "re" and example.pair is not None:
        payload = {"label": example.pair.label}
    else:
        raise AssertionError(f"invalid pilot example: {example}")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def training_prompt(
    example: PilotExample,
    eligible: list[SentenceRecord],
    ner_template: str,
    re_template: str,
) -> str:
    record = eligible[example.record_index]
    if example.task == "ner":
        return build_ner_prompt(record, [], ner_template, output_format="indices")
    if example.task == "re" and example.pair is not None:
        return build_re_prompt(
            record,
            example.pair.subject,
            example.pair.object,
            [],
            re_template,
        )
    raise AssertionError(f"invalid pilot example: {example}")


def prepare_training_material(
    examples: list[PilotExample],
    eligible: list[SentenceRecord],
    output_dir: Path,
    seed: int,
) -> list[PreparedExample]:
    ner_template = NER_TEMPLATE.read_text(encoding="utf-8")
    re_template = RE_TEMPLATE.read_text(encoding="utf-8")
    prepared: list[PreparedExample] = []
    sample_rows = []
    training_path = output_dir / "training_records.jsonl"
    for optimizer_step, example in enumerate(examples, start=1):
        record = eligible[example.record_index]
        prompt = training_prompt(example, eligible, ner_template, re_template)
        completion = gold_completion(example, eligible)
        json.loads(completion)
        item = PreparedExample(
            optimizer_step=optimizer_step,
            task=example.task,
            record_key=record.key,
            prompt=prompt,
            completion=completion,
        )
        prepared.append(item)
        sample_row: dict[str, Any] = {
            "optimizer_step": optimizer_step,
            "task": example.task,
            "document_id": record.doc_id,
            "sentence_id": record.sentence_id,
            "record_key": record.key,
            "prompt_sha256": sha256_text(prompt),
            "completion": json.loads(completion),
        }
        if example.pair is not None:
            sample_row.update(
                {
                    "subject": entity_dict(example.pair.subject, record),
                    "object": entity_dict(example.pair.object, record),
                    "label": example.pair.label,
                }
            )
        sample_rows.append(sample_row)
        append_jsonl(
            training_path,
            {
                "optimizer_step": optimizer_step,
                "task": example.task,
                "record_key": record.key,
                "prompt": prompt,
                "completion": completion,
            },
        )

    if any(row["document_id"] == TARGET_DOCUMENT_ID for row in sample_rows):
        raise AssertionError("target document leaked into the sample manifest")
    write_json(
        output_dir / "sample_manifest.json",
        {
            "schema_version": 1,
            "seed": seed,
            "excluded_document_id": TARGET_DOCUMENT_ID,
            "sampling": {
                "ner": "100 distinct uniformly sampled eligible sentences",
                "re_positive": "50 distinct sentences with one uniformly sampled positive ordered pair",
                "re_nil": "50 further distinct sentences with one uniformly sampled NIL ordered pair",
                "task_order": "deterministically shuffled",
            },
            "counts": {
                "optimizer_steps": len(prepared),
                "ner": sum(item.task == "ner" for item in prepared),
                "re": sum(item.task == "re" for item in prepared),
                "re_positive": sum(row.get("label") not in {None, "NIL"} for row in sample_rows),
                "re_nil": sum(row.get("label") == "NIL" for row in sample_rows),
            },
            "inputs": {
                "train_data": str(TRAIN_DATA),
                "train_data_sha256": sha256_file(TRAIN_DATA),
                "vocabulary": str(VOCABULARY),
                "vocabulary_sha256": sha256_file(VOCABULARY),
                "ner_template": str(NER_TEMPLATE),
                "ner_template_sha256": sha256_file(NER_TEMPLATE),
                "re_template": str(RE_TEMPLATE),
                "re_template_sha256": sha256_file(RE_TEMPLATE),
            },
            "examples": sample_rows,
        },
    )
    return prepared


def chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_training_example(tokenizer: Any, example: PreparedExample) -> dict[str, Any]:
    rendered_prompt = chat_prompt(tokenizer, example.prompt)
    prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(example.completion, add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("tokenizer has no EOS token")
    if not completion_ids or completion_ids[-1] != eos_token_id:
        completion_ids.append(eos_token_id)
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > MAX_TRAIN_TOKENS:
        raise RuntimeError(
            f"training record {example.optimizer_step} has {len(input_ids)} tokens; "
            f"limit is {MAX_TRAIN_TOKENS} and truncation is forbidden"
        )
    return {
        "optimizer_step": example.optimizer_step,
        "task": example.task,
        "record_key": example.record_key,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
    }


def compute_similarities(
    training: list[SentenceRecord], targets: list[SentenceRecord]
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("sentence-transformers is required for evaluation retrieval") from error

    model = SentenceTransformer(
        RETRIEVER_ID,
        revision=RETRIEVER_REVISION,
        device="cpu",
    )
    training_embeddings = model.encode(
        [record.text for record in training],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    target_embeddings = model.encode(
        [record.text for record in targets],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return target_embeddings @ training_embeddings.T


def require_h100(torch: Any) -> dict[str, Any]:
    visible = torch.cuda.device_count()
    if visible != 1:
        raise RuntimeError(f"expected exactly one visible CUDA device, found {visible}")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name.upper():
        raise RuntimeError(f"expected an H100, found {name!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the allocated GPU does not report BF16 support")
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
    }


def load_model_and_tokenizer(seed: int) -> tuple[Any, Any, Any, dict[str, Any]]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    except ImportError as error:
        raise RuntimeError("torch, transformers, and peft are required for the full pilot") from error

    gpu = require_h100(torch)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.to(torch.device("cuda", 0))
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
    )
    model = get_peft_model(model, lora_config, revision=MODEL_REVISION)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    targeted_modules = sorted(
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and getattr(module, "lora_A")
    )
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not targeted_modules:
        raise RuntimeError("PEFT resolved zero LoRA target modules")
    if not trainable:
        raise RuntimeError("PEFT exposed zero trainable parameters")
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    if unexpected:
        raise RuntimeError(f"non-LoRA parameters are trainable: {unexpected[:10]}")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for _, parameter in trainable)
    target_report = {
        "selector": "all-linear",
        "peft_semantics": "all Linear/Transformers Conv1D modules except the output head",
        "module_count": len(targeted_modules),
        "module_names": targeted_modules,
        "trainable_parameter_count": trainable_parameters,
        "total_parameter_count": total_parameters,
        "trainable_percent": 100.0 * trainable_parameters / total_parameters,
        "trainable_parameter_names": [name for name, _ in trainable],
        "gpu": gpu,
    }
    return torch, tokenizer, model, target_report


def tensor_batch(torch: Any, row: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        "input_ids": torch.tensor([row["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            [row["attention_mask"]], dtype=torch.long, device=device
        ),
        "labels": torch.tensor([row["labels"]], dtype=torch.long, device=device),
    }


def train_adapter(
    torch: Any,
    model: Any,
    tokenized: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    if len(tokenized) != TOTAL_STEPS:
        raise AssertionError(f"expected {TOTAL_STEPS} tokenized records")
    device = next(model.parameters()).device
    model.train()
    longest = max(tokenized, key=lambda row: len(row["input_ids"]))
    smoke_batch = tensor_batch(torch, longest, device)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        smoke_loss = model(**smoke_batch).loss
    if not torch.isfinite(smoke_loss):
        raise RuntimeError(f"non-finite smoke loss: {smoke_loss.item()}")
    smoke_loss_value = float(smoke_loss.detach().cpu())
    smoke_loss.backward()
    smoke_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not smoke_gradients:
        raise RuntimeError("memory smoke produced no LoRA gradients")
    if not all(torch.isfinite(gradient).all() for gradient in smoke_gradients):
        raise RuntimeError("memory smoke produced non-finite gradients")
    model.zero_grad(set_to_none=True)
    del smoke_batch, smoke_gradients, smoke_loss
    torch.cuda.empty_cache()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        weight_decay=0.0,
        fused=True,
    )
    step_rows = []
    task_counts: Counter[str] = Counter()
    wall_start = time.monotonic()
    for row in tokenized:
        batch = tensor_batch(torch, row, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at optimizer step {row['optimizer_step']}: {loss.item()}"
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"non-finite gradient norm at optimizer step {row['optimizer_step']}"
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        task_counts[row["task"]] += 1
        step_row = {
            "optimizer_step": row["optimizer_step"],
            "task": row["task"],
            "record_key": row["record_key"],
            "loss": float(loss.detach().cpu()),
            "gradient_norm_before_clipping": float(grad_norm.detach().cpu()),
            "learning_rate": LEARNING_RATE,
        }
        step_rows.append(step_row)
        print(
            f"[train {row['optimizer_step']}/{TOTAL_STEPS}] {row['task']} "
            f"{row['record_key']} loss={step_row['loss']:.6f}",
            flush=True,
        )
        del batch, loss, grad_norm

    if task_counts != {"ner": NER_STEPS, "re": RE_STEPS}:
        raise AssertionError(f"wrong completed task counts: {task_counts}")
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    if adapter_config.get("revision") != MODEL_REVISION:
        raise RuntimeError(
            "saved adapter_config.json does not pin the requested base-model revision"
        )
    result = {
        "optimizer": "torch.optim.AdamW(fused=True)",
        "optimizer_steps": len(step_rows),
        "task_steps": dict(task_counts),
        "learning_rate": LEARNING_RATE,
        "scheduler": "constant (none)",
        "warmup_steps": 0,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.0,
        "max_gradient_norm": 1.0,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "smoke": {
            "optimizer_step_taken": False,
            "record_key": longest["record_key"],
            "tokens": len(longest["input_ids"]),
            "loss": smoke_loss_value,
        },
        "wall_seconds": round(time.monotonic() - wall_start, 6),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "steps": step_rows,
    }
    write_json(output_dir / "training_metrics.json", result)
    del optimizer
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result


def parse_generated_json(raw: str) -> tuple[Any, str | None]:
    try:
        return json.loads(raw.strip()), None
    except json.JSONDecodeError as error:
        return None, str(error)


def generate_text(
    torch: Any,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    rendered = chat_prompt(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(context_limit, int) and input_tokens + max_new_tokens > context_limit:
        raise RuntimeError(
            f"prompt plus requested generation ({input_tokens} + {max_new_tokens}) "
            f"exceeds model context {context_limit}; truncation is forbidden"
        )
    wall_start = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = generated[0, input_tokens:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    metrics = {
        "input_tokens": input_tokens,
        "generated_tokens": int(new_tokens.shape[-1]),
        "wall_seconds": round(time.monotonic() - wall_start, 6),
    }
    return raw, metrics


def build_prediction(
    targets: list[SentenceRecord],
    predictions_by_sentence: list[list[Entity]],
    relation_rows: list[tuple[int, int, int, str]],
) -> dict[str, Any]:
    output_entities: list[dict[str, Any]] = []
    ids_by_position: dict[tuple[int, int], str] = {}
    for target_index, (target, entities) in enumerate(zip(targets, predictions_by_sentence)):
        for entity_index, entity in enumerate(entities):
            entity_id = f"s{target.sentence_id}e{entity_index}"
            ids_by_position[(target_index, entity_index)] = entity_id
            output_entities.append(
                {
                    "id": entity_id,
                    "segment_id": target.sentence_id,
                    "start": entity.start,
                    "end": entity.end,
                    "type": entity.type,
                    "text": " ".join(target.tokens[entity.start : entity.end + 1]),
                }
            )
    relation_set: set[tuple[str, str, str]] = set()
    for target_index, subject_index, object_index, label in relation_rows:
        head = ids_by_position[(target_index, subject_index)]
        tail = ids_by_position[(target_index, object_index)]
        if label in SYMMETRIC_RELATIONS and tail < head:
            head, tail = tail, head
        relation_set.add((head, label, tail))
    return {
        "document_id": TARGET_DOCUMENT_ID,
        "entities": output_entities,
        "relations": [
            {"head": head, "type": label, "tail": tail}
            for head, label, tail in sorted(relation_set)
        ],
    }


def evaluate_variant(
    *,
    name: str,
    torch: Any,
    model: Any,
    tokenizer: Any,
    targets: list[SentenceRecord],
    training: list[SentenceRecord],
    similarities: Any,
    pair_examples: list[PairExample],
    pairs_by_signature: dict[tuple[str, str], list[int]],
    output_dir: Path,
) -> dict[str, Any]:
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=False)
    trace_path = variant_dir / "trace.jsonl"
    ner_template = NER_TEMPLATE.read_text(encoding="utf-8")
    re_template = RE_TEMPLATE.read_text(encoding="utf-8")
    predictions_by_sentence: list[list[Entity]] = []
    ner_retrieval = []
    warning_count = 0
    torch.cuda.reset_peak_memory_stats()

    for target_index, target in enumerate(targets):
        selected_indices = select_ner_examples(
            training,
            similarities[target_index],
            NER_SHOTS,
            RETRIEVAL_POOL_SIZE,
        )
        examples = [training[index] for index in selected_indices]
        prompt = build_ner_prompt(target, examples, ner_template, output_format="indices")
        print(f"[{name} NER {target_index + 1}/{len(targets)}] {target.key}", flush=True)
        raw, generation = generate_text(
            torch, model, tokenizer, prompt, NER_MAX_NEW_TOKENS
        )
        payload, parse_error = parse_generated_json(raw)
        predicted, warnings = sanitize_ner(payload, len(target.tokens))
        if parse_error is not None:
            warnings.insert(0, f"invalid JSON: {parse_error}")
        warning_count += len(warnings)
        predictions_by_sentence.append(predicted)
        append_jsonl(
            trace_path,
            {
                "key": f"ner:{target.sentence_id}",
                "stage": "ner",
                "document_id": target.doc_id,
                "sentence_id": target.sentence_id,
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "example_keys": [example.key for example in examples],
                "raw_response": raw,
                "prediction": [entity_dict(entity, target) for entity in predicted],
                "warnings": warnings,
                "generation": generation,
            },
        )
        ner_retrieval.append(
            {
                "sentence_id": target.sentence_id,
                "examples": [
                    {
                        "document_id": training[index].doc_id,
                        "sentence_id": training[index].sentence_id,
                        "cosine_similarity": round(float(similarities[target_index][index]), 9),
                        "entity_labels": sorted(
                            training[index].labels, key=ENTITY_ORDER.__getitem__
                        ),
                    }
                    for index in selected_indices
                ],
            }
        )

    total_pairs = sum(len(entities) * (len(entities) - 1) for entities in predictions_by_sentence)
    pair_number = 0
    relation_rows: list[tuple[int, int, int, str]] = []
    re_retrieval = []
    for target_index, (target, entities) in enumerate(zip(targets, predictions_by_sentence)):
        for subject_index, subject in enumerate(entities):
            for object_index, object_ in enumerate(entities):
                if subject_index == object_index:
                    continue
                pair_number += 1
                examples = select_re_examples(
                    subject,
                    object_,
                    pair_examples,
                    pairs_by_signature,
                    training,
                    similarities[target_index],
                    RE_SHOTS,
                    RETRIEVAL_POOL_SIZE,
                )
                prompt_examples = [
                    (training[example.record_index], example) for example in examples
                ]
                prompt = build_re_prompt(
                    target,
                    subject,
                    object_,
                    prompt_examples,
                    re_template,
                )
                print(
                    f"[{name} RE {pair_number}/{total_pairs}] {target.key} "
                    f"{subject_index}->{object_index}",
                    flush=True,
                )
                raw, generation = generate_text(
                    torch, model, tokenizer, prompt, RE_MAX_NEW_TOKENS
                )
                payload, parse_error = parse_generated_json(raw)
                label, warnings = sanitize_re(payload)
                if parse_error is not None:
                    warnings.insert(0, f"invalid JSON: {parse_error}")
                warning_count += len(warnings)
                if label != "NIL":
                    relation_rows.append(
                        (target_index, subject_index, object_index, label)
                    )
                append_jsonl(
                    trace_path,
                    {
                        "key": f"re:{target.sentence_id}:{subject_index}:{object_index}",
                        "stage": "re",
                        "document_id": target.doc_id,
                        "sentence_id": target.sentence_id,
                        "subject_index": subject_index,
                        "object_index": object_index,
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "example_keys": [record.key for record, _ in prompt_examples],
                        "raw_response": raw,
                        "prediction": label,
                        "warnings": warnings,
                        "generation": generation,
                    },
                )
                re_retrieval.append(
                    {
                        "sentence_id": target.sentence_id,
                        "subject_index": subject_index,
                        "object_index": object_index,
                        "examples": [
                            {
                                "document_id": record.doc_id,
                                "sentence_id": record.sentence_id,
                                "subject": entity_dict(example.subject, record),
                                "object": entity_dict(example.object, record),
                                "label": example.label,
                                "cosine_similarity": round(
                                    float(similarities[target_index][example.record_index]),
                                    9,
                                ),
                            }
                            for record, example in prompt_examples
                        ],
                    }
                )

    prediction = build_prediction(targets, predictions_by_sentence, relation_rows)
    prediction_path = variant_dir / "prediction.json"
    write_json(prediction_path, prediction)
    write_json(
        variant_dir / "retrieval.json",
        {
            "schema_version": 1,
            "retriever": RETRIEVER_ID,
            "retriever_revision": RETRIEVER_REVISION,
            "excluded_document_id": TARGET_DOCUMENT_ID,
            "selection": "reconstructed similar+diverse",
            "ner_shots": NER_SHOTS,
            "re_shots": RE_SHOTS,
            "pool_size": RETRIEVAL_POOL_SIZE,
            "ner": ner_retrieval,
            "re": re_retrieval,
        },
    )
    score_path = variant_dir / "scores.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "paper_llm" / "evaluate.py"),
            str(prediction_path),
            "--gold",
            str(TRAIN_DATA),
            "--output",
            str(score_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    return {
        "scores": scores,
        "predicted_entities": len(prediction["entities"]),
        "ordered_re_candidates": total_pairs,
        "predicted_relations": len(prediction["relations"]),
        "warning_count": warning_count,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def comparison(base: dict[str, Any], lora: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("ner", "rel", "rel_plus"):
        base_f1 = base["scores"][name]["f1"]
        lora_f1 = lora["scores"][name]["f1"]
        result[name] = {
            "base_f1": base_f1,
            "lora_f1": lora_f1,
            "delta_f1": lora_f1 - base_f1,
            "base_f1_percent": base["scores"][name]["f1_percent"],
            "lora_f1_percent": lora["scores"][name]["f1_percent"],
            "delta_percentage_points": round(
                lora["scores"][name]["f1_percent"]
                - base["scores"][name]["f1_percent"],
                2,
            ),
        }
    return result


def adapter_files(adapter_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(adapter_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(adapter_dir.rglob("*"))
        if path.is_file()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write and validate sampled prompts without loading model dependencies",
    )
    return parser.parse_args()


def run(args: argparse.Namespace, output_dir: Path) -> None:
    started_at = utc_now()
    wall_start = time.monotonic()
    targets, eligible = load_pilot_records()
    sampled = sample_pilot_examples(eligible, args.seed)
    prepared = prepare_training_material(sampled, eligible, output_dir, args.seed)
    common = {
        "schema_version": 1,
        "experiment": "Qwen3.8-27B GSAP-ERE LoRA pilot",
        "status": "prepared" if args.prepare_only else "running",
        "target": {
            "document_id": TARGET_DOCUMENT_ID,
            "sentence_count": len(targets),
            "source_split": "train",
            "excluded_from_tuning": True,
            "excluded_from_evaluation_demonstrations": True,
        },
        "training": {
            "optimizer_steps": TOTAL_STEPS,
            "ner_steps": NER_STEPS,
            "re_steps": RE_STEPS,
            "re_positive_steps": RE_POSITIVE_STEPS,
            "re_nil_steps": RE_NIL_STEPS,
            "seed": args.seed,
            "learning_rate": LEARNING_RATE,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "class": "Qwen3_5ForCausalLM",
            "dtype": "bfloat16",
            "quantized": False,
            "thinking": False,
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": 0.0,
                "bias": "none",
                "target_modules": "all-linear",
            },
        },
        "evaluation": {
            "target_sentence_count": len(targets),
            "ner_shots": NER_SHOTS,
            "re_shots": RE_SHOTS,
            "full_article_context": False,
            "thinking": False,
            "generation": "greedy Hugging Face generation without an Ollama JSON grammar",
            "parsing": "strict whole-response JSON parsing followed by paper_llm sanitizers",
            "relation_candidates": "all ordered pairs of each system's predicted entities",
        },
        "retriever": {"id": RETRIEVER_ID, "revision": RETRIEVER_REVISION},
        "inputs": {
            "eligible_sentence_count": len(eligible),
            "train_data_sha256": sha256_file(TRAIN_DATA),
            "vocabulary_sha256": sha256_file(VOCABULARY),
            "ner_template_sha256": sha256_file(NER_TEMPLATE),
            "re_template_sha256": sha256_file(RE_TEMPLATE),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "git_commit": git_commit(),
        "started_at": started_at,
    }
    write_json(output_dir / "run.json", common)
    if args.prepare_only:
        common["finished_at"] = utc_now()
        common["wall_seconds"] = round(time.monotonic() - wall_start, 6)
        write_json(output_dir / "run.json", common)
        print(f"Prepared {len(prepared)} training records in {output_dir}", flush=True)
        return

    similarities = compute_similarities(eligible, targets)
    pair_examples, pairs_by_signature = build_pair_examples(eligible)
    torch, tokenizer, model, target_report = load_model_and_tokenizer(args.seed)
    write_json(output_dir / "targeted_modules.json", target_report)
    tokenized = [encode_training_example(tokenizer, example) for example in prepared]
    token_lengths = [len(row["input_ids"]) for row in tokenized]
    write_json(
        output_dir / "tokenization.json",
        {
            "maximum_allowed_tokens": MAX_TRAIN_TOKENS,
            "minimum_tokens": min(token_lengths),
            "maximum_tokens": max(token_lengths),
            "mean_tokens": sum(token_lengths) / len(token_lengths),
            "records": [
                {
                    "optimizer_step": row["optimizer_step"],
                    "task": row["task"],
                    "record_key": row["record_key"],
                    "prompt_tokens": row["prompt_tokens"],
                    "completion_tokens": row["completion_tokens"],
                    "total_tokens": len(row["input_ids"]),
                }
                for row in tokenized
            ],
        },
    )
    training_metrics = train_adapter(torch, model, tokenized, output_dir)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    with model.disable_adapter():
        base_result = evaluate_variant(
            name="base",
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            targets=targets,
            training=eligible,
            similarities=similarities,
            pair_examples=pair_examples,
            pairs_by_signature=pairs_by_signature,
            output_dir=output_dir,
        )
    lora_result = evaluate_variant(
        name="lora",
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        targets=targets,
        training=eligible,
        similarities=similarities,
        pair_examples=pair_examples,
        pairs_by_signature=pairs_by_signature,
        output_dir=output_dir,
    )
    comparison_result = comparison(base_result, lora_result)
    write_json(output_dir / "comparison.json", comparison_result)

    common.update(
        {
            "status": "complete",
            "training_result": {
                key: value for key, value in training_metrics.items() if key != "steps"
            },
            "base_result": base_result,
            "lora_result": lora_result,
            "comparison": comparison_result,
            "adapter_files": adapter_files(output_dir / "adapter"),
            "software": {
                name: package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "accelerate",
                    "sentence-transformers",
                    "gsapere",
                )
            },
            "system": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "gpus": gpu_inventory(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            },
            "finished_at": utc_now(),
            "wall_seconds": round(time.monotonic() - wall_start, 6),
        }
    )
    write_json(output_dir / "run.json", common)
    print(json.dumps(comparison_result, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    output_dir = initialize_output_dir(args.output_dir)
    try:
        run(args, output_dir)
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": utc_now(),
            "git_commit": git_commit(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        write_json(
            output_dir / "failure.json",
            failure,
        )
        run_path = output_dir / "run.json"
        if run_path.exists():
            run_record = json.loads(run_path.read_text(encoding="utf-8"))
            run_record.update(
                {
                    "status": "failed",
                    "failure": {
                        "error_type": failure["error_type"],
                        "error": failure["error"],
                        "failed_at": failure["failed_at"],
                    },
                }
            )
            write_json(run_path, run_record)
        raise


if __name__ == "__main__":
    main()
