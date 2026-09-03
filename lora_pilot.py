#!/usr/bin/env python3
"""Run the fixed Qwen3.8-27B GSAP-ERE LoRA experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
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
PILOT_REGIME = "pilot"
FULL_PASS_REGIME = "full-pass"
TRAINING_REGIMES = (PILOT_REGIME, FULL_PASS_REGIME)
NER_SAMPLE_COUNT = 100
RE_POSITIVE_SAMPLE_COUNT = 50
RE_NIL_SAMPLE_COUNT = 50
RE_SAMPLE_COUNT = RE_POSITIVE_SAMPLE_COUNT + RE_NIL_SAMPLE_COUNT
NER_STEPS = 200
RE_STEPS = 200
TOTAL_STEPS = NER_STEPS + RE_STEPS

LEARNING_RATE = 2e-4
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
CHECKPOINT_INTERVAL = 1000
LORA_R = 16
LORA_ALPHA = 16
MAX_TRAIN_TOKENS = 4096
NER_SHOTS = 10
RE_SHOTS = 1
RETRIEVAL_POOL_SIZE = 100
NER_MAX_NEW_TOKENS = 2048
RE_MAX_NEW_TOKENS = 64
SCORER_PYTHON_ENV = "GSAPERE_SCORER_PYTHON"

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
    sample_index: int
    task: str
    stratum: str
    record_key: str
    source: PilotExample


@dataclass(frozen=True)
class TrainingBatch:
    optimizer_step: int
    task_step: int
    task: str
    rows: tuple[PreparedExample, ...]


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


def qlora_signature() -> dict[str, Any]:
    return {
        "model_revision": MODEL_REVISION,
        "backend": "bitsandbytes",
        "bitsandbytes_version": package_version("bitsandbytes"),
        "bits": 4,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "double_quant": True,
    }


def scorer_python() -> str:
    return os.environ.get(SCORER_PYTHON_ENV, sys.executable)


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
    if len(eligible) < NER_SAMPLE_COUNT:
        raise RuntimeError("not enough eligible sentences for NER sampling")

    ner_rng = random.Random(f"{seed}:ner")
    positive_rng = random.Random(f"{seed}:re-positive")
    nil_rng = random.Random(f"{seed}:re-nil")
    order_rng = random.Random(f"{seed}:order")

    ner_indices = ner_rng.sample(range(len(eligible)), NER_SAMPLE_COUNT)
    pair_examples, _ = build_pair_examples(eligible)
    positive_by_record: dict[int, list[PairExample]] = defaultdict(list)
    nil_by_record: dict[int, list[PairExample]] = defaultdict(list)
    for pair in pair_examples:
        destination = nil_by_record if pair.label == "NIL" else positive_by_record
        destination[pair.record_index].append(pair)

    if len(positive_by_record) < RE_POSITIVE_SAMPLE_COUNT:
        raise RuntimeError("not enough distinct positive RE sentences")
    positive_record_indices = positive_rng.sample(
        sorted(positive_by_record), RE_POSITIVE_SAMPLE_COUNT
    )
    positive_record_set = set(positive_record_indices)
    nil_candidates = sorted(set(nil_by_record) - positive_record_set)
    if len(nil_candidates) < RE_NIL_SAMPLE_COUNT:
        raise RuntimeError("not enough distinct NIL RE sentences after positive sampling")
    nil_record_indices = nil_rng.sample(nil_candidates, RE_NIL_SAMPLE_COUNT)

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
    if task_counts != {"ner": NER_SAMPLE_COUNT, "re": RE_SAMPLE_COUNT}:
        raise AssertionError(f"unexpected task counts: {task_counts}")
    if re_labels != {
        "positive": RE_POSITIVE_SAMPLE_COUNT,
        "NIL": RE_NIL_SAMPLE_COUNT,
    }:
        raise AssertionError(f"unexpected RE balance: {re_labels}")
    if len(set(ner_indices)) != NER_SAMPLE_COUNT:
        raise AssertionError("NER source sentences are not unique")
    if len(set(re_record_indices)) != RE_SAMPLE_COUNT:
        raise AssertionError("RE source sentences are not unique")
    if any(eligible[example.record_index].doc_id == TARGET_DOCUMENT_ID for example in examples):
        raise AssertionError("target document leaked into sampled training examples")
    return examples


def full_pass_examples(
    eligible: list[SentenceRecord], seed: int
) -> list[PilotExample]:
    pair_examples, _ = build_pair_examples(eligible)
    examples = [PilotExample("ner", index) for index in range(len(eligible))]
    examples.extend(
        PilotExample("re", pair.record_index, pair) for pair in pair_examples
    )
    random.Random(f"{seed}:full-pass-order").shuffle(examples)
    if Counter(example.task for example in examples) != {
        "ner": len(eligible),
        "re": len(pair_examples),
    }:
        raise AssertionError("wrong full-pass task counts")
    if any(
        eligible[example.record_index].doc_id == TARGET_DOCUMENT_ID
        for example in examples
    ):
        raise AssertionError("target document leaked into full-pass training examples")
    return examples


def select_training_examples(
    eligible: list[SentenceRecord], seed: int, regime: str
) -> list[PilotExample]:
    if regime == PILOT_REGIME:
        return sample_pilot_examples(eligible, seed)
    if regime == FULL_PASS_REGIME:
        return full_pass_examples(eligible, seed)
    raise ValueError(f"unknown training regime: {regime}")


def example_stratum(example: PilotExample) -> str:
    if example.task == "ner":
        return "ner"
    if example.task == "re" and example.pair is not None:
        return "nil" if example.pair.label == "NIL" else "positive"
    raise AssertionError(f"invalid training example: {example}")


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
    regime: str,
) -> list[PreparedExample]:
    ner_template = NER_TEMPLATE.read_text(encoding="utf-8")
    re_template = RE_TEMPLATE.read_text(encoding="utf-8")
    prepared: list[PreparedExample] = []
    task_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    training_path = output_dir / "training_records.jsonl"
    with training_path.open("w", encoding="utf-8") as stream:
        for sample_index, example in enumerate(examples, start=1):
            record = eligible[example.record_index]
            if record.doc_id == TARGET_DOCUMENT_ID:
                raise AssertionError("target document leaked into training material")
            prompt = training_prompt(example, eligible, ner_template, re_template)
            completion = gold_completion(example, eligible)
            parsed_completion = json.loads(completion)
            stratum = example_stratum(example)
            item = PreparedExample(
                sample_index=sample_index,
                task=example.task,
                stratum=stratum,
                record_key=record.key,
                source=example,
            )
            prepared.append(item)
            task_counts[example.task] += 1
            stratum_counts[stratum] += 1
            row: dict[str, Any] = {
                "sample_index": sample_index,
                "task": example.task,
                "stratum": stratum,
                "document_id": record.doc_id,
                "sentence_id": record.sentence_id,
                "record_key": record.key,
                "prompt_sha256": sha256_text(prompt),
                "completion": parsed_completion,
            }
            if example.pair is not None:
                row.update(
                    {
                        "subject": entity_dict(example.pair.subject, record),
                        "object": entity_dict(example.pair.object, record),
                        "label": example.pair.label,
                    }
                )
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    write_json(
        output_dir / "sample_manifest.json",
        {
            "schema_version": 1,
            "seed": seed,
            "training_regime": regime,
            "excluded_document_id": TARGET_DOCUMENT_ID,
            "selection": (
                {
                    "ner": "100 distinct uniformly sampled eligible sentences",
                    "re_positive": "50 distinct sentences with one uniformly sampled positive ordered pair",
                    "re_nil": "50 further distinct sentences with one uniformly sampled NIL ordered pair",
                    "sample_order": "deterministically shuffled",
                }
                if regime == PILOT_REGIME
                else {
                    "ner": "every eligible stored sentence exactly once, including empty-entity sentences",
                    "re": "every ordered pair of distinct gold entities exactly once",
                    "nil": "ordered pairs without an annotated relation are labeled NIL",
                    "sample_order": "deterministically shuffled",
                }
            ),
            "counts": {
                "unique_training_records": len(prepared),
                "ner": task_counts["ner"],
                "re": task_counts["re"],
                "re_positive": stratum_counts["positive"],
                "re_nil": stratum_counts["nil"],
                "eligible_sentences": len(eligible),
                "eligible_documents": len({record.doc_id for record in eligible}),
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
            "record_manifest": "training_records.jsonl",
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


def encode_training_example(
    tokenizer: Any,
    example: PreparedExample,
    eligible: list[SentenceRecord],
    ner_template: str,
    re_template: str,
) -> dict[str, Any]:
    prompt = training_prompt(example.source, eligible, ner_template, re_template)
    completion = gold_completion(example.source, eligible)
    rendered_prompt = chat_prompt(tokenizer, prompt)
    prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("tokenizer has no EOS token")
    if not completion_ids or completion_ids[-1] != eos_token_id:
        completion_ids.append(eos_token_id)
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > MAX_TRAIN_TOKENS:
        raise RuntimeError(
            f"training record {example.sample_index} has {len(input_ids)} tokens; "
            f"limit is {MAX_TRAIN_TOKENS} and truncation is forbidden"
        )
    return {
        "sample_index": example.sample_index,
        "task": example.task,
        "stratum": example.stratum,
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
        import bitsandbytes as bnb
        import torch
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "torch, transformers, peft, and bitsandbytes are required"
        ) from error

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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
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

    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("model did not load in bitsandbytes 4-bit mode")
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict) or any(
        str(device).lower() in {"cpu", "disk"} for device in device_map.values()
    ):
        raise RuntimeError(f"unexpected quantized-model device map: {device_map!r}")
    quantized_module_count = sum(
        isinstance(module, bnb.nn.Linear4bit) for module in model.modules()
    )
    if not quantized_module_count:
        raise RuntimeError("bitsandbytes resolved zero Linear4bit modules")

    targeted_modules = sorted(
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and getattr(module, "lora_A")
    )
    if len(targeted_modules) != quantized_module_count:
        raise RuntimeError(
            "not every bitsandbytes Linear4bit module received a LoRA adapter: "
            f"{len(targeted_modules)} adapters for {quantized_module_count} modules"
        )
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not targeted_modules:
        raise RuntimeError("PEFT resolved zero LoRA target modules")
    if not trainable:
        raise RuntimeError("PEFT exposed zero trainable parameters")
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    if unexpected:
        raise RuntimeError(f"non-LoRA parameters are trainable: {unexpected[:10]}")
    if any(not parameter.is_floating_point() for _, parameter in trainable):
        raise RuntimeError("a trainable LoRA parameter is not floating point")
    if any(parameter.device.type != "cuda" or parameter.device.index != 0 for _, parameter in trainable):
        raise RuntimeError("trainable LoRA parameters are not all on cuda:0")
    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()
    trainable_dtypes: dict[str, int] = {}
    for _, parameter in trainable:
        dtype = str(parameter.dtype)
        trainable_dtypes[dtype] = trainable_dtypes.get(dtype, 0) + parameter.numel()
    if set(trainable_dtypes) != {"torch.float32"}:
        raise RuntimeError(f"unexpected LoRA parameter dtypes: {trainable_dtypes}")
    frozen_storage_dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            continue
        dtype = str(parameter.dtype)
        frozen_storage_dtypes[dtype] = (
            frozen_storage_dtypes.get(dtype, 0) + parameter.numel()
        )
    target_report = {
        "selector": "all-linear",
        "peft_semantics": "all supported linear modules, including bitsandbytes Linear4bit, except the output head",
        "module_count": len(targeted_modules),
        "module_names": targeted_modules,
        "trainable_parameter_count": trainable_parameters,
        "total_parameter_count": total_parameters,
        "trainable_percent": 100.0 * trainable_parameters / total_parameters,
        "trainable_parameter_names": [name for name, _ in trainable],
        "quantization": {
            "backend": "bitsandbytes",
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "double_quant": True,
            "device_map": device_map,
            "linear4bit_module_count": quantized_module_count,
            "trainable_parameter_dtypes": trainable_dtypes,
            "frozen_parameter_storage_numel_by_dtype": frozen_storage_dtypes,
        },
        "gpu": gpu,
    }
    return torch, tokenizer, model, target_report


def build_training_schedule(
    prepared: list[PreparedExample], seed: int, regime: str = PILOT_REGIME
) -> list[TrainingBatch]:
    pools = {
        task: [row for row in prepared if row.task == task]
        for task in ("ner", "re")
    }
    pool_sizes = {task: len(rows) for task, rows in pools.items()}
    if not all(pool_sizes.values()):
        raise AssertionError(f"empty task pool: {pool_sizes}")

    task_batches: dict[str, list[tuple[PreparedExample, ...]]] = {}
    if regime == PILOT_REGIME:
        if pool_sizes != {"ner": NER_SAMPLE_COUNT, "re": RE_SAMPLE_COUNT}:
            raise AssertionError(f"wrong pilot task pools: {pool_sizes}")
        for task, step_count in (("ner", NER_STEPS), ("re", RE_STEPS)):
            pool = list(pools[task])
            random.Random(f"{seed}:training-stream:{task}").shuffle(pool)
            presentations = step_count * MICRO_BATCH_SIZE
            if presentations % len(pool):
                raise AssertionError(f"{task} presentations do not evenly cover its pool")
            stream = pool * (presentations // len(pool))
            task_batches[task] = [
                tuple(stream[offset : offset + MICRO_BATCH_SIZE])
                for offset in range(0, len(stream), MICRO_BATCH_SIZE)
            ]
    elif regime == FULL_PASS_REGIME:
        for task, pool in pools.items():
            shuffled = list(pool)
            random.Random(f"{seed}:training-stream:{task}").shuffle(shuffled)
            usable = len(shuffled) - len(shuffled) % MICRO_BATCH_SIZE
            task_batches[task] = [
                tuple(shuffled[offset : offset + MICRO_BATCH_SIZE])
                for offset in range(0, usable, MICRO_BATCH_SIZE)
            ]
    else:
        raise ValueError(f"unknown training regime: {regime}")

    task_limits = {task: len(batches) for task, batches in task_batches.items()}

    schedule: list[TrainingBatch] = []
    task_counts: Counter[str] = Counter()
    if regime == PILOT_REGIME:
        task_order = [
            task
            for pair_index in range(max(task_limits.values()))
            for task in (("ner", "re") if pair_index % 2 == 0 else ("re", "ner"))
            if pair_index < task_limits[task]
        ]
    else:
        task_order = []
        total_steps = sum(task_limits.values())
        while len(task_order) < total_steps:
            prefix_size = len(task_order) + 1
            remaining = [
                task
                for task in ("ner", "re")
                if task_counts[task] < task_limits[task]
            ]
            task = max(
                remaining,
                key=lambda name: (
                    prefix_size * task_limits[name] / total_steps
                    - task_counts[name],
                    name == "ner",
                ),
            )
            task_order.append(task)
            task_counts[task] += 1
        task_counts.clear()

    for task in task_order:
        task_counts[task] += 1
        task_step = task_counts[task]
        rows = task_batches[task][task_step - 1]
        if not 1 <= len(rows) <= MICRO_BATCH_SIZE:
            raise AssertionError(f"invalid batch size {len(rows)}")
        sample_indices = [row.sample_index for row in rows]
        if len(sample_indices) != len(set(sample_indices)):
            raise AssertionError("a scheduled batch repeats a training record")
        if {row.task for row in rows} != {task}:
            raise AssertionError("a scheduled batch mixes tasks")
        schedule.append(
            TrainingBatch(
                optimizer_step=len(schedule) + 1,
                task_step=task_step,
                task=task,
                rows=rows,
            )
        )

    if len(schedule) != sum(task_limits.values()):
        raise AssertionError("wrong total scheduled optimizer steps")
    if dict(task_counts) != task_limits:
        raise AssertionError(f"wrong scheduled task counts: {task_counts}")
    sample_presentations = Counter(
        row.sample_index for batch in schedule for row in batch.rows
    )
    expected_repetitions = 16 if regime == PILOT_REGIME else 1
    prepared_ids = {row.sample_index for row in prepared}
    scheduled_ids = set(sample_presentations)
    dropped_ids = prepared_ids - scheduled_ids
    if scheduled_ids - prepared_ids:
        raise AssertionError("schedule contains records outside the manifest")
    expected_dropped = 0 if regime == PILOT_REGIME else sum(
        len(pool) % MICRO_BATCH_SIZE for pool in pools.values()
    )
    if len(dropped_ids) != expected_dropped:
        raise AssertionError(f"wrong dropped-record count: {len(dropped_ids)}")
    if set(sample_presentations.values()) != {expected_repetitions}:
        raise AssertionError("training records do not have equal presentation counts")
    return schedule


def padded_training_rows(
    rows: Sequence[dict[str, Any]], pad_token_id: int
) -> dict[str, list[list[int]]]:
    if not 1 <= len(rows) <= MICRO_BATCH_SIZE:
        raise AssertionError(
            f"expected a micro-batch of 1 through {MICRO_BATCH_SIZE}, found {len(rows)}"
        )
    if any(
        not (
            len(row["input_ids"])
            == len(row["attention_mask"])
            == len(row["labels"])
        )
        for row in rows
    ):
        raise AssertionError("training row tensors have inconsistent lengths")
    if any(not any(label != -100 for label in row["labels"]) for row in rows):
        raise AssertionError("training row has no supervised completion tokens")
    maximum = max(len(row["input_ids"]) for row in rows)

    def right_pad(values: list[int], fill: int) -> list[int]:
        return values + [fill] * (maximum - len(values))

    return {
        "input_ids": [right_pad(row["input_ids"], pad_token_id) for row in rows],
        "attention_mask": [right_pad(row["attention_mask"], 0) for row in rows],
        "labels": [right_pad(row["labels"], -100) for row in rows],
    }


def tensor_batch(
    torch: Any,
    rows: Sequence[dict[str, Any]],
    device: Any,
    pad_token_id: int,
) -> dict[str, Any]:
    padded = padded_training_rows(rows, pad_token_id)
    return {
        name: torch.tensor(values, dtype=torch.long, device=device)
        for name, values in padded.items()
    }


def encode_training_batch(
    tokenizer: Any,
    batch: TrainingBatch,
    eligible: list[SentenceRecord],
    ner_template: str,
    re_template: str,
) -> list[dict[str, Any]]:
    return [
        encode_training_example(
            tokenizer, row, eligible, ner_template, re_template
        )
        for row in batch.rows
    ]


def preflight_training_schedule(
    tokenizer: Any,
    schedule: list[TrainingBatch],
    eligible: list[SentenceRecord],
    output_dir: Path,
    ner_template: str,
    re_template: str,
) -> TrainingBatch:
    seen: set[int] = set()
    token_lengths: list[int] = []
    widest: TrainingBatch | None = None
    widest_shape = (-1, -1, -1)
    schedule_path = output_dir / "training_schedule.jsonl"
    tokenization_path = output_dir / "tokenization_records.jsonl"
    with schedule_path.open("w", encoding="utf-8") as schedule_stream, tokenization_path.open(
        "w", encoding="utf-8"
    ) as tokenization_stream:
        for batch in schedule:
            encoded = encode_training_batch(
                tokenizer, batch, eligible, ner_template, re_template
            )
            lengths = [len(row["input_ids"]) for row in encoded]
            padded_tokens = max(lengths)
            shape = (
                len(encoded) * padded_tokens * padded_tokens,
                len(encoded) * padded_tokens,
                padded_tokens,
            )
            if shape > widest_shape:
                widest = batch
                widest_shape = shape
            schedule_stream.write(
                json.dumps(
                    {
                        "optimizer_step": batch.optimizer_step,
                        "task_step": batch.task_step,
                        "task": batch.task,
                        "batch_size": len(batch.rows),
                        "sample_indices": [row.sample_index for row in batch.rows],
                        "record_keys": [row.record_key for row in batch.rows],
                        "strata": dict(Counter(row.stratum for row in batch.rows)),
                        "unpadded_tokens": lengths,
                        "padded_tokens": padded_tokens,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            for row in encoded:
                sample_index = row["sample_index"]
                if sample_index in seen:
                    continue
                seen.add(sample_index)
                token_lengths.append(len(row["input_ids"]))
                tokenization_stream.write(
                    json.dumps(
                        {
                            "sample_index": sample_index,
                            "task": row["task"],
                            "stratum": row["stratum"],
                            "record_key": row["record_key"],
                            "prompt_tokens": row["prompt_tokens"],
                            "completion_tokens": row["completion_tokens"],
                            "total_tokens": len(row["input_ids"]),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        for stream in (schedule_stream, tokenization_stream):
            stream.flush()
            os.fsync(stream.fileno())
    expected = {row.sample_index for batch in schedule for row in batch.rows}
    if seen != expected:
        raise AssertionError("tokenization preflight did not cover every training record")
    if widest is None or not token_lengths:
        raise AssertionError("empty tokenization preflight")
    write_json(
        output_dir / "tokenization.json",
        {
            "maximum_allowed_tokens": MAX_TRAIN_TOKENS,
            "unique_records": len(token_lengths),
            "minimum_tokens": min(token_lengths),
            "maximum_tokens": max(token_lengths),
            "mean_tokens": sum(token_lengths) / len(token_lengths),
            "record_manifest": "tokenization_records.jsonl",
            "preflight": "all unique records tokenized before optimizer step 1",
            "widest_scheduled_batch": {
                "optimizer_step": widest.optimizer_step,
                "task": widest.task,
                "batch_size": len(widest.rows),
                "sample_indices": [row.sample_index for row in widest.rows],
                "padded_tokens": widest_shape[2],
                "tensor_token_positions": widest_shape[1],
                "quadratic_attention_proxy": widest_shape[0],
            },
        },
    )
    return widest


def load_checkpoint_adapter(model: Any, checkpoint_dir: Path) -> None:
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    adapter_dir = checkpoint_dir / "adapter"
    if not (checkpoint_dir / "COMPLETE").is_file():
        raise RuntimeError(f"incomplete resume checkpoint: {checkpoint_dir}")
    if not adapter_dir.is_dir():
        raise RuntimeError(f"resume checkpoint has no adapter: {checkpoint_dir}")
    state_dict = load_peft_weights(str(adapter_dir), device="cpu")
    load_result = set_peft_model_state_dict(model, state_dict)
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"unexpected adapter keys in resume checkpoint: {load_result.unexpected_keys[:10]}"
        )


def save_training_checkpoint(
    torch: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    output_dir: Path,
    regime: str,
    schedule_sha256: str,
    completed_steps: int,
    step_rows: list[dict[str, Any]],
    task_counts: Counter[str],
    task_presentations: Counter[str],
    stratum_presentations: Counter[str],
) -> Path:
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint_root / f"step-{completed_steps:08d}"
    if checkpoint_dir.exists():
        raise RuntimeError(f"checkpoint already exists: {checkpoint_dir}")
    checkpoint_dir.mkdir()
    model.save_pretrained(checkpoint_dir / "adapter", safe_serialization=True)
    trainer_state = {
        "completed_steps": completed_steps,
        "training_regime": regime,
        "schedule_sha256": schedule_sha256,
        "git_commit": git_commit(),
        "qlora_signature": qlora_signature(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
        "step_rows": step_rows,
        "task_counts": dict(task_counts),
        "task_presentations": dict(task_presentations),
        "stratum_presentations": dict(stratum_presentations),
    }
    state_tmp = checkpoint_dir / "trainer_state.pt.tmp"
    torch.save(trainer_state, state_tmp)
    os.replace(state_tmp, checkpoint_dir / "trainer_state.pt")
    write_json(
        checkpoint_dir / "state.json",
        {
            "completed_steps": completed_steps,
            "training_regime": regime,
            "schedule_sha256": schedule_sha256,
            "git_commit": git_commit(),
            "qlora_signature": qlora_signature(),
            "saved_at": utc_now(),
        },
    )
    (checkpoint_dir / "COMPLETE").write_text(utc_now() + "\n", encoding="utf-8")
    latest_tmp = checkpoint_root / "LATEST.tmp"
    latest_tmp.write_text(checkpoint_dir.name + "\n", encoding="utf-8")
    os.replace(latest_tmp, checkpoint_root / "LATEST")
    completed = sorted(
        path
        for path in checkpoint_root.glob("step-*")
        if (path / "COMPLETE").is_file()
    )
    for obsolete in completed[:-2]:
        if obsolete.parent.resolve() != checkpoint_root.resolve():
            raise AssertionError(f"refusing to remove checkpoint outside root: {obsolete}")
        shutil.rmtree(obsolete)
    return checkpoint_dir


def train_adapter(
    torch: Any,
    model: Any,
    tokenizer: Any,
    prepared: list[PreparedExample],
    eligible: list[SentenceRecord],
    output_dir: Path,
    seed: int,
    pad_token_id: int,
    regime: str,
    resume_from: Path | None,
) -> dict[str, Any]:
    from transformers import get_linear_schedule_with_warmup

    ner_template = NER_TEMPLATE.read_text(encoding="utf-8")
    re_template = RE_TEMPLATE.read_text(encoding="utf-8")
    schedule = build_training_schedule(prepared, seed, regime)
    widest = preflight_training_schedule(
        tokenizer,
        schedule,
        eligible,
        output_dir,
        ner_template,
        re_template,
    )
    schedule_sha256 = sha256_file(output_dir / "training_schedule.jsonl")
    total_steps = len(schedule)
    expected_task_steps = dict(Counter(batch.task for batch in schedule))
    expected_task_presentations = dict(
        Counter(row.task for batch in schedule for row in batch.rows)
    )
    expected_stratum_presentations = dict(
        Counter(row.stratum for batch in schedule for row in batch.rows)
    )
    sample_presentations = Counter(
        row.sample_index for batch in schedule for row in batch.rows
    )
    repetition_values = set(sample_presentations.values())
    if len(repetition_values) != 1:
        raise AssertionError("training records have unequal repetition counts")
    record_repetitions = repetition_values.pop()

    resume_metadata: dict[str, Any] | None = None
    if resume_from is not None:
        resume_from = resume_from.resolve()
        metadata_path = resume_from / "state.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"resume checkpoint has no state.json: {resume_from}")
        resume_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_resume_metadata = {
            "training_regime": regime,
            "schedule_sha256": schedule_sha256,
            "git_commit": git_commit(),
            "qlora_signature": qlora_signature(),
        }
        for key, expected in expected_resume_metadata.items():
            if resume_metadata.get(key) != expected:
                raise RuntimeError(
                    f"resume checkpoint {key} mismatch: "
                    f"{resume_metadata.get(key)!r} != {expected!r}"
                )
        load_checkpoint_adapter(model, resume_from)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    device = parameters[0].device
    model.train()
    widest_encoded = encode_training_batch(
        tokenizer, widest, eligible, ner_template, re_template
    )
    smoke_batch = tensor_batch(torch, widest_encoded, device, pad_token_id)
    parameter_snapshot = [parameter.detach().cpu().clone() for parameter in parameters]
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all()
    smoke_optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        smoke_loss = model(**smoke_batch).loss
    if not torch.isfinite(smoke_loss):
        raise RuntimeError(f"non-finite smoke loss: {smoke_loss.item()}")
    state_creation_loss_value = float(smoke_loss.detach().cpu())
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
    smoke_optimizer.step()
    smoke_optimizer.zero_grad(set_to_none=True)
    del smoke_gradients, smoke_loss
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        resident_state_loss = model(**smoke_batch).loss
    if not torch.isfinite(resident_state_loss):
        raise RuntimeError(
            f"non-finite resident-state smoke loss: {resident_state_loss.item()}"
        )
    resident_state_loss_value = float(resident_state_loss.detach().cpu())
    resident_state_loss.backward()
    smoke_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not smoke_gradients or not all(
        torch.isfinite(gradient).all() for gradient in smoke_gradients
    ):
        raise RuntimeError("resident-state memory smoke produced invalid gradients")
    smoke_optimizer.step()
    smoke_peak_allocated = torch.cuda.max_memory_allocated()
    smoke_peak_reserved = torch.cuda.max_memory_reserved()
    with torch.no_grad():
        for parameter, initial in zip(parameters, parameter_snapshot, strict=True):
            parameter.copy_(initial)
    smoke_optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state_all(cuda_rng_states)
    del (
        smoke_batch,
        smoke_gradients,
        resident_state_loss,
        smoke_optimizer,
        parameter_snapshot,
        cpu_rng_state,
        cuda_rng_states,
    )
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )
    if resume_from is None:
        completed_steps = 0
        step_rows: list[dict[str, Any]] = []
        task_counts: Counter[str] = Counter()
        task_presentations: Counter[str] = Counter()
        stratum_presentations: Counter[str] = Counter()
    else:
        trainer_state = torch.load(
            resume_from / "trainer_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        completed_steps = int(trainer_state["completed_steps"])
        if not 0 < completed_steps < total_steps:
            raise RuntimeError(f"invalid resumed step {completed_steps}/{total_steps}")
        if int(resume_metadata["completed_steps"]) != completed_steps:
            raise RuntimeError("checkpoint JSON and trainer-state steps disagree")
        if trainer_state["training_regime"] != regime:
            raise RuntimeError("trainer-state regime mismatch")
        if trainer_state["schedule_sha256"] != schedule_sha256:
            raise RuntimeError("trainer-state schedule mismatch")
        if trainer_state["git_commit"] != git_commit():
            raise RuntimeError("trainer-state commit mismatch")
        if trainer_state["qlora_signature"] != qlora_signature():
            raise RuntimeError("trainer-state QLoRA signature mismatch")
        optimizer.load_state_dict(trainer_state["optimizer"])
        scheduler.load_state_dict(trainer_state["scheduler"])
        if scheduler.last_epoch != completed_steps:
            raise RuntimeError(
                f"resumed scheduler is at {scheduler.last_epoch}, expected {completed_steps}"
            )
        if any(
            group["lr"] != expected
            for group, expected in zip(
                optimizer.param_groups, scheduler.get_last_lr(), strict=True
            )
        ):
            raise RuntimeError("resumed optimizer and scheduler learning rates disagree")
        step_rows = list(trainer_state["step_rows"])
        task_counts = Counter(trainer_state["task_counts"])
        task_presentations = Counter(trainer_state["task_presentations"])
        stratum_presentations = Counter(trainer_state["stratum_presentations"])
        if len(step_rows) != completed_steps:
            raise RuntimeError("resumed loss history length does not match completed step")
        for saved, scheduled in zip(
            step_rows, schedule[:completed_steps], strict=True
        ):
            if (
                saved["optimizer_step"] != scheduled.optimizer_step
                or saved["task"] != scheduled.task
                or saved["sample_indices"]
                != [row.sample_index for row in scheduled.rows]
            ):
                raise RuntimeError("resumed history does not match the current schedule")
        prefix = schedule[:completed_steps]
        expected_prefix_task_counts = Counter(batch.task for batch in prefix)
        expected_prefix_task_presentations = Counter(
            row.task for batch in prefix for row in batch.rows
        )
        expected_prefix_stratum_presentations = Counter(
            row.stratum for batch in prefix for row in batch.rows
        )
        if task_counts != expected_prefix_task_counts:
            raise RuntimeError("resumed task-step counters do not match the schedule")
        if task_presentations != expected_prefix_task_presentations:
            raise RuntimeError("resumed task presentations do not match the schedule")
        if stratum_presentations != expected_prefix_stratum_presentations:
            raise RuntimeError("resumed stratum presentations do not match the schedule")
        random.setstate(trainer_state["python_random_state"])
        torch.set_rng_state(trainer_state["torch_rng_state"])
        torch.cuda.set_rng_state_all(trainer_state["cuda_rng_states"])
        del trainer_state
    wall_start = time.monotonic()
    log_path = output_dir / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_stream:
        for previous in step_rows:
            log_stream.write(json.dumps(previous, sort_keys=True) + "\n")
        for scheduled in schedule[completed_steps:]:
            encoded = encode_training_batch(
                tokenizer, scheduled, eligible, ner_template, re_template
            )
            batch = tensor_batch(torch, encoded, device, pad_token_id)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at optimizer step {scheduled.optimizer_step}: {loss.item()}"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"non-finite gradient norm at optimizer step {scheduled.optimizer_step}"
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            task_counts[scheduled.task] += 1
            task_presentations[scheduled.task] += len(encoded)
            stratum_presentations.update(row["stratum"] for row in encoded)
            step_row = {
                "optimizer_step": scheduled.optimizer_step,
                "task_step": scheduled.task_step,
                "task": scheduled.task,
                "batch_size": len(encoded),
                "sample_indices": [row["sample_index"] for row in encoded],
                "record_keys": [row["record_key"] for row in encoded],
                "strata": dict(Counter(row["stratum"] for row in encoded)),
                "padded_tokens": max(len(row["input_ids"]) for row in encoded),
                "supervised_tokens": sum(
                    sum(label != -100 for label in row["labels"])
                    for row in encoded
                ),
                "loss": float(loss.detach().cpu()),
                "gradient_norm_before_clipping": float(grad_norm.detach().cpu()),
                "learning_rate": learning_rate,
            }
            step_rows.append(step_row)
            log_stream.write(json.dumps(step_row, sort_keys=True) + "\n")
            if scheduled.optimizer_step % 100 == 0:
                log_stream.flush()
            print(
                f"[train {scheduled.optimizer_step}/{total_steps}] "
                f"{scheduled.task} {scheduled.task_step}/{expected_task_steps[scheduled.task]} "
                f"loss={step_row['loss']:.6f} lr={learning_rate:.8f}",
                flush=True,
            )
            if (
                regime == FULL_PASS_REGIME
                and scheduled.optimizer_step % CHECKPOINT_INTERVAL == 0
                and scheduled.optimizer_step < total_steps
            ):
                checkpoint_dir = save_training_checkpoint(
                    torch,
                    model,
                    optimizer,
                    scheduler,
                    output_dir,
                    regime,
                    schedule_sha256,
                    scheduled.optimizer_step,
                    step_rows,
                    task_counts,
                    task_presentations,
                    stratum_presentations,
                )
                print(f"[checkpoint] {checkpoint_dir}", flush=True)
            del batch, encoded, loss, grad_norm
        log_stream.flush()
        os.fsync(log_stream.fileno())

    if dict(task_counts) != expected_task_steps:
        raise AssertionError(f"wrong completed task counts: {task_counts}")
    if dict(task_presentations) != expected_task_presentations:
        raise AssertionError(f"wrong task presentation counts: {task_presentations}")
    if dict(stratum_presentations) != expected_stratum_presentations:
        raise AssertionError(f"wrong stratum presentation counts: {stratum_presentations}")
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
        "training_regime": regime,
        "optimizer_steps": len(step_rows),
        "task_steps": dict(task_counts),
        "selected_training_records": len(prepared),
        "unique_training_records": len(sample_presentations),
        "dropped_training_records": len(prepared) - len(sample_presentations),
        "example_presentations": sum(task_presentations.values()),
        "task_presentations": dict(task_presentations),
        "stratum_presentations": dict(stratum_presentations),
        "record_repetitions": record_repetitions,
        "resumed_from": str(resume_from) if resume_from is not None else None,
        "resumed_completed_steps": (
            int(resume_metadata["completed_steps"])
            if resume_metadata is not None
            else 0
        ),
        "checkpoint_interval_steps": (
            CHECKPOINT_INTERVAL if regime == FULL_PASS_REGIME else None
        ),
        "schedule_sha256": schedule_sha256,
        "learning_rate": LEARNING_RATE,
        "scheduler": "linear warmup then linear decay",
        "warmup_steps": WARMUP_STEPS,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "actual_batch_size_counts": dict(
            sorted(Counter(len(batch.rows) for batch in schedule).items())
        ),
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "max_gradient_norm": 1.0,
        "bf16": True,
        "base_weights": "bitsandbytes 4-bit NF4 with double quantization",
        "adapter_dtype": "float32",
        "tf32": True,
        "gradient_checkpointing": True,
        "smoke": {
            "optimizer_step_executed_then_parameters_restored": True,
            "scheduled_optimizer_step": widest.optimizer_step,
            "task": widest.task,
            "record_keys": [row.record_key for row in widest.rows],
            "batch_size": len(widest.rows),
            "padded_tokens": max(len(row["input_ids"]) for row in widest_encoded),
            "state_creation_loss": state_creation_loss_value,
            "resident_state_loss": resident_state_loss_value,
            "peak_cuda_memory_allocated_bytes": smoke_peak_allocated,
            "peak_cuda_memory_reserved_bytes": smoke_peak_reserved,
        },
        "wall_seconds": round(time.monotonic() - wall_start, 6),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "steps": step_rows,
    }
    write_json(output_dir / "training_metrics.json", result)
    del optimizer, scheduler
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result


def parse_generated_json(raw: str) -> tuple[Any, str | None]:
    text = raw.strip()
    lines = text.splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text), None
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
            scorer_python(),
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
        "--training-regime",
        choices=TRAINING_REGIMES,
        default=PILOT_REGIME,
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="resume full-pass training from a completed checkpoint directory",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write and validate training records without loading model dependencies",
    )
    return parser.parse_args()


def run(args: argparse.Namespace, output_dir: Path) -> None:
    if args.resume_from is not None and args.training_regime != FULL_PASS_REGIME:
        raise ValueError("--resume-from is only valid with --training-regime full-pass")
    if args.resume_from is not None and args.prepare_only:
        raise ValueError("--resume-from cannot be combined with --prepare-only")
    started_at = utc_now()
    wall_start = time.monotonic()
    targets, eligible = load_pilot_records()
    selected = select_training_examples(eligible, args.seed, args.training_regime)
    prepared = prepare_training_material(
        selected,
        eligible,
        output_dir,
        args.seed,
        args.training_regime,
    )
    planned_schedule = build_training_schedule(
        prepared, args.seed, args.training_regime
    )
    scheduled_sample_indices = {
        row.sample_index for batch in planned_schedule for row in batch.rows
    }
    dropped = [
        row for row in prepared if row.sample_index not in scheduled_sample_indices
    ]
    if args.training_regime == PILOT_REGIME and dropped:
        raise AssertionError("pilot schedule dropped training records")
    if args.training_regime == FULL_PASS_REGIME and (
        len(dropped) != 2 or {row.task for row in dropped} != {"re"}
    ):
        raise AssertionError("full-pass schedule must drop exactly two RE records")
    write_json(
        output_dir / "dropped_training_records.json",
        {
            "policy": "drop_last=True for each task-homogeneous stream",
            "count": len(dropped),
            "records": [
                {
                    "sample_index": row.sample_index,
                    "task": row.task,
                    "stratum": row.stratum,
                    "record_key": row.record_key,
                    "label": (
                        row.source.pair.label
                        if row.source.pair is not None
                        else None
                    ),
                }
                for row in dropped
            ],
        },
    )
    planned_task_steps = dict(Counter(batch.task for batch in planned_schedule))
    planned_task_presentations = dict(
        Counter(row.task for batch in planned_schedule for row in batch.rows)
    )
    planned_stratum_presentations = dict(
        Counter(row.stratum for batch in planned_schedule for row in batch.rows)
    )
    common = {
        "schema_version": 1,
        "experiment": f"Qwen3.8-27B GSAP-ERE 4-bit QLoRA {args.training_regime}",
        "training_regime": args.training_regime,
        "status": "prepared" if args.prepare_only else "running",
        "target": {
            "document_id": TARGET_DOCUMENT_ID,
            "sentence_count": len(targets),
            "source_split": "train",
            "excluded_from_tuning": True,
            "excluded_from_evaluation_demonstrations": True,
        },
        "training": {
            "optimizer_steps": len(planned_schedule),
            "task_steps": planned_task_steps,
            "selected_training_records": len(prepared),
            "trained_unique_records": len(scheduled_sample_indices),
            "dropped_training_records": len(dropped),
            "example_presentations": sum(planned_task_presentations.values()),
            "task_presentations": planned_task_presentations,
            "stratum_presentations": planned_stratum_presentations,
            "record_repetitions": (
                len(planned_schedule) * MICRO_BATCH_SIZE // len(prepared)
                if args.training_regime == PILOT_REGIME
                else 1
            ),
            "seed": args.seed,
            "learning_rate": LEARNING_RATE,
            "scheduler": "linear warmup then linear decay",
            "warmup_steps": WARMUP_STEPS,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "planned_actual_batch_size_counts": dict(
                sorted(Counter(len(batch.rows) for batch in planned_schedule).items())
            ),
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "torch.optim.AdamW(fused=True)",
            "checkpoint_interval_steps": (
                CHECKPOINT_INTERVAL
                if args.training_regime == FULL_PASS_REGIME
                else None
            ),
            "task_batches": "task-homogeneous",
            "drop_last": args.training_regime == FULL_PASS_REGIME,
            "resume_from": (
                str(args.resume_from.resolve())
                if args.resume_from is not None
                else None
            ),
            "full_pass_re_representation": (
                "all ordered pairs of distinct gold entities; no NIL rebalancing"
                if args.training_regime == FULL_PASS_REGIME
                else None
            ),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "class": "Qwen3_5ForCausalLM",
            "quantized_linear_weight_dtype": "4-bit NF4",
            "non_quantized_base_parameter_dtype": "float32 after k-bit preparation",
            "compute_dtype": "bfloat16",
            "adapter_dtype": "float32",
            "quantized": True,
            "quantization": {
                "backend": "bitsandbytes",
                "bits": 4,
                "quant_type": "nf4",
                "compute_dtype": "bfloat16",
                "double_quant": True,
            },
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
            "base_variant": "the same 4-bit quantized base with the LoRA adapter disabled",
            "target_sentence_count": len(targets),
            "ner_shots": NER_SHOTS,
            "re_shots": RE_SHOTS,
            "full_article_context": False,
            "thinking": False,
            "generation": "greedy Hugging Face generation without an Ollama JSON grammar",
            "parsing": "whole-response JSON with one optional outer Markdown JSON fence, followed by paper_llm sanitizers",
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

    del selected, planned_schedule
    torch, tokenizer, model, target_report = load_model_and_tokenizer(args.seed)
    write_json(output_dir / "targeted_modules.json", target_report)
    if tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer has no pad token ID")
    training_metrics = train_adapter(
        torch,
        model,
        tokenizer,
        prepared,
        eligible,
        output_dir,
        args.seed,
        tokenizer.pad_token_id,
        args.training_regime,
        args.resume_from,
    )
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    similarities = compute_similarities(eligible, targets)
    pair_examples, pairs_by_signature = build_pair_examples(eligible)

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
                    "bitsandbytes",
                    "accelerate",
                    "sentence-transformers",
                )
            }
            | {
                "gsapere": base_result["scores"]["scorer"]["version"],
                "gsapere_python": scorer_python(),
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
