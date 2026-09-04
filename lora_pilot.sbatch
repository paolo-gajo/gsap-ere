#!/usr/bin/env bash
#SBATCH --job-name=gsap-lora-pilot
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=.slurm/%x-%j.out
#SBATCH --error=.slurm/%x-%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
expected_commit="${CAMPAIGN_COMMIT:?export the exact pilot commit as CAMPAIGN_COMMIT}"
training_regime="${TRAINING_REGIME:-pilot}"
model_id="${MODEL_ID:-Qwen/Qwen3.8-27B}"
bare="${BARE:-0}"
case "${training_regime}" in
  pilot) artifact_name=lora_pilot ;;
  full-pool-3k) artifact_name=lora_full_pool_3k ;;
  full-pass) artifact_name=lora_full_pass ;;
  *) echo "error: unknown TRAINING_REGIME ${training_regime}" >&2; exit 2 ;;
esac
case "${model_id}" in
  Qwen/Qwen3.8-27B)
    model_slug=qwen38_27b
    model_class=Qwen3_5ForCausalLM
    model_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
    model_wrapper=qwen-chat-template
    ;;
  Qwen/Qwen3-14B-Base)
    model_slug=qwen3_14b_base
    model_class=Qwen3ForCausalLM
    model_revision=0b0bd3732e2c374d483664439ea334928b65f304
    model_wrapper=none
    artifact_name="${artifact_name}_${model_slug}"
    ;;
  *) echo "error: unsupported MODEL_ID ${model_id}" >&2; exit 2 ;;
esac
case "${bare}" in
  0) training_format=templated-chat ;;
  1) training_format=bare; artifact_name="${artifact_name}_bare" ;;
  *) echo "error: BARE must be 0 or 1" >&2; exit 2 ;;
esac
artifact_root="/project/def-hsajjad/pgajo/gsap-ere_artifacts/${artifact_name}"
output_dir="${OUTPUT_DIR:-${artifact_root}/${expected_commit}-job-${SLURM_JOB_ID}}"

if [[ ! "${expected_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: CAMPAIGN_COMMIT must be a full lowercase 40-character SHA" >&2
  exit 2
fi

cd "${repo_root}"
actual_commit="$(git rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  echo "error: checkout ${actual_commit} does not match ${expected_commit}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "error: refusing to run from a dirty checkout" >&2
  git status --short >&2
  exit 1
fi
if [[ -e "${output_dir}" ]]; then
  echo "error: OUTPUT_DIR already exists: ${output_dir}" >&2
  exit 1
fi

module --force purge
module load StdEnv/2023 python/3.11.5

venv_python="${repo_root}/.env/bin/python"
scorer_python="${repo_root}/.eval-env/bin/python"
if [[ ! -x "${venv_python}" ]]; then
  echo "error: missing pilot environment at ${repo_root}/.env" >&2
  exit 1
fi
if [[ ! -x "${scorer_python}" ]]; then
  echo "error: missing scorer environment at ${repo_root}/.eval-env" >&2
  exit 1
fi

"${venv_python}" - <<'PY'
import importlib.metadata as metadata

expected = {
    "accelerate": "1.13.0",
    "bitsandbytes": "0.50.1",
    "huggingface_hub": "1.18.0",
    "numpy": "2.4.2",
    "peft": "0.19.1",
    "safetensors": "0.7.0",
    "sentence-transformers": "6.0.0",
    "torch": "2.12.0",
    "transformers": "5.10.2",
}
actual = {name: metadata.version(name) for name in expected}
base_versions = {name: version.split("+", 1)[0] for name, version in actual.items()}
if base_versions != expected:
    raise SystemExit(f"package-version mismatch: {actual!r}")
assert actual["bitsandbytes"] == "0.50.1+computecanada"
print(actual, flush=True)
PY

"${scorer_python}" - <<'PY'
import importlib.metadata as metadata
from gsapere.evaluation.hgere import evaluate

assert metadata.version("gsapere") == "0.2.4"
print("gsapere evaluator 0.2.4 import passed", flush=True)
PY

export HF_HOME=/scratch/pgajo/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export GSAPERE_SCORER_PYTHON="${scorer_python}"

gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ "${gpu_names^^}" != *H100* ]]; then
  echo "error: expected an H100 allocation, found ${gpu_names}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${output_dir}")"
echo "commit=${actual_commit}"
echo "gpu=${gpu_names}"
echo "training_regime=${training_regime}"
echo "training_format=${training_format}"
echo "model_id=${model_id}"
echo "model_class=${model_class}"
echo "model_revision=${model_revision}"
echo "model_wrapper=${model_wrapper}"
echo "output=${output_dir}"

resume_args=()
bare_args=()
if [[ "${bare}" == 1 ]]; then
  bare_args=(--bare)
fi
if [[ -n "${RESUME_FROM:-}" ]]; then
  if [[ "${training_regime}" != full-pass ]]; then
    echo "error: RESUME_FROM is only valid for full-pass training" >&2
    exit 2
  fi
  resume_args=(--resume-from "${RESUME_FROM}")
  echo "resume_from=${RESUME_FROM}"
fi

srun --ntasks=1 "${venv_python}" -u lora_pilot.py \
  --seed 42 \
  --model-id "${model_id}" \
  --training-regime "${training_regime}" \
  "${bare_args[@]}" \
  "${resume_args[@]}" \
  --output-dir "${output_dir}"

for artifact in \
  run.json \
  dropped_training_records.json \
  sample_manifest.json \
  training_records.jsonl \
  training_schedule.jsonl \
  training_log.jsonl \
  training_metrics.json \
  tokenization.json \
  tokenization_records.jsonl \
  targeted_modules.json \
  adapter/adapter_config.json \
  adapter/adapter_model.safetensors \
  base/prediction.json \
  base/retrieval.json \
  base/scores.json \
  base/trace.jsonl \
  lora/prediction.json \
  lora/retrieval.json \
  lora/scores.json \
  lora/trace.jsonl \
  comparison.json; do
  if [[ ! -s "${output_dir}/${artifact}" ]]; then
    echo "error: missing or empty artifact ${artifact}" >&2
    exit 1
  fi
done

"${venv_python}" - \
  "${output_dir}" \
  "${training_regime}" \
  "${actual_commit}" \
  "${training_format}" \
  "${model_id}" \
  "${model_revision}" \
  "${model_class}" \
  "${model_wrapper}" <<'PY'
from collections import Counter
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
regime = sys.argv[2]
expected_commit = sys.argv[3]
expected_format = sys.argv[4]
expected_model_id = sys.argv[5]
expected_model_revision = sys.argv[6]
expected_model_class = sys.argv[7]
expected_model_wrapper = sys.argv[8]
run = json.loads((root / "run.json").read_text())
manifest = json.loads((root / "sample_manifest.json").read_text())
dropped = json.loads((root / "dropped_training_records.json").read_text())
training = json.loads((root / "training_metrics.json").read_text())
targets = json.loads((root / "targeted_modules.json").read_text())
comparison = json.loads((root / "comparison.json").read_text())
tokenization = json.loads((root / "tokenization.json").read_text())
assert run["status"] == "complete"
assert run["git_commit"] == expected_commit
assert run["training_regime"] == regime
assert run["training_format"] == expected_format
assert run["training"]["format"] == expected_format
assert run["training"]["bare"] == (expected_format == "bare")
assert run["model"]["id"] == expected_model_id
assert run["model"]["revision"] == expected_model_revision
assert run["model"]["class"] == expected_model_class
assert run["training"]["model_wrapper"] == expected_model_wrapper
assert run["evaluation"]["model_wrapper"] == expected_model_wrapper
assert manifest["training_regime"] == regime
assert manifest["training_format"] == expected_format
assert manifest["serialization"]["model_wrapper"] == expected_model_wrapper
assert manifest["excluded_document_id"] == run["target"]["document_id"]
assert manifest["counts"]["selected_manifest_examples"] == training["selected_manifest_examples"]
assert training["training_regime"] == regime
assert training["training_format"] == expected_format
assert training["model_id"] == expected_model_id
assert training["model_revision"] == expected_model_revision
assert tokenization["training_format"] == expected_format
assert tokenization["model_wrapper"] == expected_model_wrapper
assert training["model_wrapper"] == expected_model_wrapper
assert training["learning_rate"] == 2e-4
assert training["warmup_steps"] == 5
assert training["micro_batch_size"] == 8
assert training["gradient_accumulation_steps"] == 1
assert training["weight_decay"] == 0.01
assert training["optimizer"] == "torch.optim.AdamW(fused=True)"
assert run["model"]["quantization"] == {
    "backend": "bitsandbytes",
    "bits": 4,
    "compute_dtype": "bfloat16",
    "double_quant": True,
    "quant_type": "nf4",
}
assert run["software"]["bitsandbytes"] == "0.50.1+computecanada"
assert targets["quantization"]["backend"] == "bitsandbytes"
assert targets["model_id"] == expected_model_id
assert targets["model_revision"] == expected_model_revision
assert targets["model_class"] == expected_model_class
assert targets["quantization"]["linear4bit_module_count"] > 0
assert targets["quantization"]["trainable_parameter_dtypes"] == {
    "torch.float32": targets["trainable_parameter_count"]
}

for variant in ("base", "lora"):
    retrieval = json.loads((root / variant / "retrieval.json").read_text())
    trace = [
        json.loads(line)
        for line in (root / variant / "trace.jsonl").read_text().splitlines()
    ]
    if expected_format == "bare":
        assert run["evaluation"]["prompt_format"] == "bare x-only content"
        assert run["evaluation"]["ner_shots"] == 0
        assert run["evaluation"]["re_shots"] == 0
        assert run["retriever"] is None
        assert retrieval["retriever"] is None
        assert retrieval["retriever_revision"] is None
        assert retrieval["ner_shots"] == 0
        assert retrieval["re_shots"] == 0
        assert retrieval["pool_size"] == 0
        assert all(not row["examples"] for row in retrieval["ner"])
        assert all(not row["examples"] for row in retrieval["re"])
        assert all(not row["example_keys"] for row in trace)
        for row in trace:
            prompt = json.loads(row["prompt"])
            if row["stage"] == "ner":
                assert isinstance(prompt, list)
                assert all(
                    isinstance(token, list)
                    and len(token) == 2
                    and isinstance(token[0], int)
                    and isinstance(token[1], str)
                    for token in prompt
                )
            else:
                assert row["stage"] == "re"
                assert set(prompt) == {"sentence_tokens", "subject", "object"}
    else:
        assert retrieval["ner_shots"] == 10
        assert retrieval["re_shots"] == 1

schedule = [json.loads(line) for line in (root / "training_schedule.jsonl").open()]
presentations = Counter(
    sample_index
    for batch in schedule
    for sample_index in batch["sample_indices"]
)
assert all(len(batch["sample_indices"]) == len(set(batch["sample_indices"])) for batch in schedule)
assert all(len(batch["strata"]) == 1 if batch["task"] == "ner" else True for batch in schedule)
training_record_count = 0
ner_source_counts = Counter()
entity_classes = Counter()
re_pair_keys = set()
relation_classes = Counter()
with (root / "training_records.jsonl").open() as stream:
    for line in stream:
        row = json.loads(line)
        assert row["document_id"] != run["target"]["document_id"]
        assert row["training_format"] == expected_format
        if row["task"] == "ner":
            ner_source_counts[row["record_key"]] += 1
            entity_classes.update(
                entity["type"] for entity in row["completion"]["entities"]
            )
        else:
            assert row["task"] == "re"
            relation_classes[row["completion"]["label"]] += 1
            re_pair_keys.add(
                (
                    row["record_key"],
                    json.dumps(row["subject"], sort_keys=True),
                    json.dumps(row["object"], sort_keys=True),
                )
            )
        training_record_count += 1
tokenization_record_count = sum(1 for _ in (root / "tokenization_records.jsonl").open())
training_log_count = sum(1 for _ in (root / "training_log.jsonl").open())
assert training_record_count == training["selected_manifest_examples"]
assert tokenization_record_count == training["scheduled_manifest_examples"]
assert training_log_count == training["optimizer_steps"] == len(training["steps"])
if regime == "pilot":
    assert training["optimizer_steps"] == 400
    assert training["task_steps"] == {"ner": 200, "re": 200}
    assert training["scheduled_manifest_examples"] == 200
    assert training["selected_manifest_examples"] == 200
    assert training["dropped_training_records"] == dropped["count"] == 0
    assert training["example_presentations"] == 3200
    assert training["task_presentations"] == {"ner": 1600, "re": 1600}
    assert training["stratum_presentations"] == {"ner": 1600, "nil": 800, "positive": 800}
    assert training["manifest_example_repetitions"] == 16
    assert training["actual_batch_size_counts"] == {"8": 400}
    assert len(schedule) == 400
    assert set(presentations.values()) == {16}
elif regime == "full-pool-3k":
    assert training["optimizer_steps"] == 6000
    assert training["task_steps"] == {"ner": 3000, "re": 3000}
    assert training["scheduled_manifest_examples"] == 48000
    assert training["selected_manifest_examples"] == 48000
    assert training["dropped_training_records"] == dropped["count"] == 0
    assert training["example_presentations"] == 48000
    assert training["task_presentations"] == {"ner": 24000, "re": 24000}
    assert training["stratum_presentations"] == {
        "ner": 24000,
        "nil": 12000,
        "positive": 12000,
    }
    assert training["manifest_example_repetitions"] == 1
    assert training["unique_ner_source_sentences"] == 20712
    assert training["maximum_ner_source_presentations"] == 2
    assert training["unique_re_source_pairs"] == 24000
    assert manifest["counts"]["unique_ner_source_sentences"] == 20712
    assert manifest["counts"]["maximum_ner_source_presentations"] == 2
    assert manifest["counts"]["unique_re_source_pairs"] == 24000
    assert Counter(ner_source_counts.values()) == {1: 17424, 2: 3288}
    assert len(re_pair_keys) == 24000
    assert relation_classes["NIL"] == 12000
    assert sum(
        count for label, count in relation_classes.items() if label != "NIL"
    ) == 12000
    assert set(entity_classes) == {
        "MLModel",
        "MLModelGeneric",
        "ModelArchitecture",
        "Method",
        "Dataset",
        "DatasetGeneric",
        "DataSource",
        "ReferenceLink",
        "Task",
        "URL",
    }
    assert set(relation_classes) == {
        "NIL",
        "appliedTo",
        "architecture",
        "benchmarkFor",
        "citation",
        "coreference",
        "evaluatedOn",
        "generatedBy",
        "hasInstanceType",
        "isBasedOn",
        "isComparedTo",
        "isHyponymOf",
        "isPartOf",
        "size",
        "sourcedFrom",
        "trainedOn",
        "transformedFrom",
        "usedFor",
        "url",
    }
    assert training["actual_batch_size_counts"] == {"8": 6000}
    assert len(schedule) == 6000
    assert set(presentations.values()) == {1}
    assert all(
        len(batch["record_keys"]) == len(set(batch["record_keys"]))
        for batch in schedule
        if batch["task"] == "ner"
    )
else:
    assert regime == "full-pass"
    assert training["optimizer_steps"] == 26312
    assert training["task_steps"] == {"ner": 2589, "re": 23723}
    assert training["selected_manifest_examples"] == 210498
    assert training["scheduled_manifest_examples"] == 210496
    assert training["dropped_training_records"] == dropped["count"] == 2
    assert [row["sample_index"] for row in dropped["records"]] == [57757, 118283]
    assert training["example_presentations"] == 210496
    assert training["task_presentations"] == {"ner": 20712, "re": 189784}
    assert training["stratum_presentations"] == {"ner": 20712, "nil": 159508, "positive": 30276}
    assert training["manifest_example_repetitions"] == 1
    assert training["actual_batch_size_counts"] == {"8": 26312}
    assert len(schedule) == 26312
    assert set(presentations.values()) == {1}
for metric in comparison.values():
    for value in metric.values():
        assert isinstance(value, (int, float)) and math.isfinite(value)
print("artifact audit passed", flush=True)
PY

printf 'commit=%s\njob_id=%s\naudited_at=%s\n' \
  "${actual_commit}" \
  "${SLURM_JOB_ID}" \
  "$(date --utc --iso-8601=seconds)" \
  >"${output_dir}/COMPLETE"
test -s "${output_dir}/COMPLETE"
