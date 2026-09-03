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
artifact_root=/project/def-hsajjad/pgajo/gsap-ere_artifacts/lora_pilot
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
echo "output=${output_dir}"

srun --ntasks=1 "${venv_python}" -u lora_pilot.py \
  --seed 42 \
  --output-dir "${output_dir}"

for artifact in \
  run.json \
  sample_manifest.json \
  training_records.jsonl \
  training_metrics.json \
  tokenization.json \
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

"${venv_python}" - "${output_dir}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
run = json.loads((root / "run.json").read_text())
training = json.loads((root / "training_metrics.json").read_text())
comparison = json.loads((root / "comparison.json").read_text())
assert run["status"] == "complete"
assert training["optimizer_steps"] == 200
assert training["task_steps"] == {"ner": 100, "re": 100}
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
