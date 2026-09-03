#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="${script_dir}/run_john_qwen.sh"
unset RESULT_DIR

run_case() {
  local zero_icl="$1"
  local re_shots="$2"
  local full_article_context="$3"
  local think="$4"

  NER_OUTPUT_FORMAT=inline \
    ZERO_ICL="${zero_icl}" \
    RE_SHOTS="${re_shots}" \
    FULL_ARTICLE_CONTEXT="${full_article_context}" \
    THINK="${think}" \
    OVERWRITE=0 \
    bash "${runner}"
}

run_case 1 1 0 0
run_case 1 1 0 1

run_case 0 1 0 0
run_case 0 1 0 1

run_case 0 5 0 0
run_case 0 5 0 1

run_case 0 10 0 0
run_case 0 10 0 1

run_case 0 1 1 0
run_case 0 1 1 1
