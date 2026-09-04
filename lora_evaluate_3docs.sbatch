#!/usr/bin/env bash
#SBATCH --job-name=gsap-lora-14b-test3
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=.slurm/%x-%j.out
#SBATCH --error=.slurm/%x-%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
expected_commit="${CAMPAIGN_COMMIT:?export the exact evaluation commit as CAMPAIGN_COMMIT}"
source_run="${SOURCE_RUN_DIR:?export the completed adapter run as SOURCE_RUN_DIR}"
model_id="Qwen/Qwen3-14B-Base"
model_revision="0b0bd3732e2c374d483664439ea334928b65f304"
target_data="${repo_root}/data/test.jsonl"
output_root="/project/def-hsajjad/pgajo/gsap-ere_artifacts/lora_eval_3docs_qwen3_14b_base_bare"
output_dir="${OUTPUT_DIR:-${output_root}/${expected_commit}-job-${SLURM_JOB_ID}}"
document_ids=(
  10018_2208_04405.txt
  00050_1406_2661.txt
  10047_2005_06182.txt
)

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
if [[ ! -s "${source_run}/COMPLETE" || ! -s "${source_run}/adapter/adapter_model.safetensors" ]]; then
  echo "error: incomplete source adapter run: ${source_run}" >&2
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
if [[ ! -x "${venv_python}" || ! -x "${scorer_python}" ]]; then
  echo "error: required environments are missing" >&2
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
echo "model_id=${model_id}"
echo "model_revision=${model_revision}"
echo "source_run=${source_run}"
echo "target_data=${target_data}"
echo "documents=${document_ids[*]}"
echo "output=${output_dir}"

document_args=()
for document_id in "${document_ids[@]}"; do
  document_args+=(--document-id "${document_id}")
done

srun --ntasks=1 "${venv_python}" -u lora_evaluate.py \
  --seed 42 \
  --model-id "${model_id}" \
  --source-run "${source_run}" \
  --target-data "${target_data}" \
  "${document_args[@]}" \
  --output-dir "${output_dir}"

for document_id in "${document_ids[@]}"; do
  document_dir="${output_dir}/documents/${document_id}"
  for artifact in prediction.json retrieval.json scores.json trace.jsonl result.json COMPLETE; do
    if [[ ! -s "${document_dir}/${artifact}" ]]; then
      echo "error: missing ${document_id}/${artifact}" >&2
      exit 1
    fi
  done
done
for artifact in run.json targeted_modules.json aggregate_scores.json EVALUATION_COMPLETE; do
  if [[ ! -s "${output_dir}/${artifact}" ]]; then
    echo "error: missing root artifact ${artifact}" >&2
    exit 1
  fi
done

"${venv_python}" - "${output_dir}" "${actual_commit}" "${source_run}" "${document_ids[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_commit = sys.argv[2]
expected_source = str(Path(sys.argv[3]).resolve())
expected_documents = sys.argv[4:]
run = json.loads((root / "run.json").read_text())
aggregate = json.loads((root / "aggregate_scores.json").read_text())
assert run["status"] == "complete"
assert run["git_commit"] == expected_commit
assert run["source_run"]["path"] == expected_source
assert run["model"]["id"] == "Qwen/Qwen3-14B-Base"
assert run["evaluation"]["base_variant_evaluated"] is False
assert run["evaluation"]["training_performed"] is False
assert run["evaluation"]["planned_document_ids"] == expected_documents
assert run["evaluation"]["completed_document_ids"] == expected_documents
assert aggregate["document_ids"] == expected_documents
for document_id in expected_documents:
    document = json.loads(
        (root / "documents" / document_id / "result.json").read_text()
    )
    assert document["document_id"] == document_id
    assert document["result"]["scores"]["document_id"] == document_id
    document_root = root / "documents" / document_id
    for name, expected in document["artifacts"].items():
        path = document_root / name
        assert path.stat().st_size == expected["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
    complete = json.loads((document_root / "COMPLETE").read_text())
    result_path = document_root / "result.json"
    assert complete["result_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()
print("three-document adapter evaluation audit passed", flush=True)
PY

completion_tmp="${output_dir}/.COMPLETE.tmp"
printf 'commit=%s\njob_id=%s\naudited_at=%s\n' \
  "${actual_commit}" \
  "${SLURM_JOB_ID}" \
  "$(date --utc --iso-8601=seconds)" \
  >"${completion_tmp}"
mv "${completion_tmp}" "${output_dir}/COMPLETE"
