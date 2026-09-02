#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

model="${MODEL:-qwen3.8:27b}"
base_url="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
think="${THINK:-0}"
if [[ "${think}" != "0" && "${think}" != "1" ]]; then
  echo "error: THINK must be 0 or 1" >&2
  exit 1
fi
zero_icl="${ZERO_ICL:-0}"
if [[ "${zero_icl}" != "0" && "${zero_icl}" != "1" ]]; then
  echo "error: ZERO_ICL must be 0 or 1" >&2
  exit 1
fi
full_article_context="${FULL_ARTICLE_CONTEXT:-0}"
if [[ "${full_article_context}" != "0" && "${full_article_context}" != "1" ]]; then
  echo "error: FULL_ARTICLE_CONTEXT must be 0 or 1" >&2
  exit 1
fi
re_shots="${RE_SHOTS:-1}"
if [[ "${re_shots}" != "1" && "${re_shots}" != "5" ]]; then
  echo "error: RE_SHOTS must be 1 or 5" >&2
  exit 1
fi
if [[ "${think}" == "1" ]]; then
  mode_suffix="-thinking"
  default_ner_num_predict=4096
  default_re_num_predict=4096
  think_args=(--think)
else
  mode_suffix=""
  default_ner_num_predict=2048
  default_re_num_predict=64
  think_args=()
fi
if [[ "${zero_icl}" == "1" ]]; then
  icl_suffix="-zero-icl"
  icl_args=(--ner-shots 0 --re-shots 0)
elif [[ "${re_shots}" == "5" ]]; then
  icl_suffix="-re-5shot"
  icl_args=(--ner-shots 10 --re-shots 5)
else
  icl_suffix=""
  icl_args=()
fi
if [[ "${full_article_context}" == "1" ]]; then
  context_suffix="-full-article"
  context_args=(--full-article-context)
else
  context_suffix=""
  context_args=()
fi
default_result_dir="${script_dir}/results/qwen3.8-27b-ollama-john${mode_suffix}${icl_suffix}${context_suffix}-00016_2106_09462"
result_dir="${RESULT_DIR:-${default_result_dir}}"
ner_num_predict="${NER_NUM_PREDICT:-${default_ner_num_predict}}"
re_num_predict="${RE_NUM_PREDICT:-${default_re_num_predict}}"
venv_dir="${PAPER_LLM_VENV:-${script_dir}/.venv}"
uv_version="0.12.9"
uv_bin="${HOME}/.local/bin/uv"

if [[ ! -x "${uv_bin}" ]]; then
  bootstrap_dir="$(mktemp -d /tmp/gsap-ere-uv.XXXXXX)"
  cleanup_bootstrap() {
    if [[ "${bootstrap_dir}" == /tmp/gsap-ere-uv.* ]]; then
      rm -rf -- "${bootstrap_dir}"
    fi
  }
  trap cleanup_bootstrap RETURN
  archive="${bootstrap_dir}/uv.tar.gz"
  curl --fail --location --retry 3 \
    "https://github.com/astral-sh/uv/releases/download/${uv_version}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    --output "${archive}"
  tar -xzf "${archive}" -C "${bootstrap_dir}"
  mkdir -p -- "$(dirname -- "${uv_bin}")"
  install -m 0755 \
    "${bootstrap_dir}/uv-x86_64-unknown-linux-gnu/uv" "${uv_bin}"
  cleanup_bootstrap
  trap - RETURN
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${uv_bin}" venv --python /usr/bin/python3 "${venv_dir}"
fi
venv_python="${venv_dir}/bin/python"

if ! "${venv_python}" -c \
  'import importlib.metadata as m; assert m.version("sentence-transformers") == "5.1.2"; assert m.version("torch").startswith("2.8.0"); assert m.version("gsapere") == "0.2.4"' \
  >/dev/null 2>&1; then
  "${uv_bin}" pip install --python "${venv_python}" \
    --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0'
  "${uv_bin}" pip install --python "${venv_python}" \
    'sentence-transformers==5.1.2'
  "${uv_bin}" pip install --python "${venv_python}" --no-deps \
    'gsapere==0.2.4'
fi

if command -v ollama >/dev/null 2>&1; then
  ollama_bin="$(command -v ollama)"
elif [[ -x "${HOME}/.local/bin/ollama" ]]; then
  ollama_bin="${HOME}/.local/bin/ollama"
else
  echo "error: Ollama not found in PATH or ${HOME}/.local/bin" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found" >&2
  exit 1
fi

export OLLAMA_HOST="${base_url#http://}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"

server_pid=""
server_log=""
cleanup_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT

if ! curl --silent --fail --max-time 3 "${base_url}/api/version" >/dev/null; then
  server_log="$(mktemp /tmp/gsap-ere-ollama-server.XXXXXX.log)"
  "${ollama_bin}" serve >"${server_log}" 2>&1 &
  server_pid="$!"
  for _attempt in $(seq 1 60); do
    if curl --silent --fail --max-time 3 "${base_url}/api/version" >/dev/null; then
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "error: Ollama server exited; log: ${server_log}" >&2
      tail -100 "${server_log}" >&2
      exit 1
    fi
    sleep 1
  done
fi

if ! curl --silent --fail --max-time 3 "${base_url}/api/version" >/dev/null; then
  echo "error: Ollama server did not become ready; log: ${server_log}" >&2
  exit 1
fi

echo "Pulling ${model} if needed"
"${ollama_bin}" pull "${model}"

inference_mode=(--resume)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  inference_mode=(--overwrite)
fi

"${venv_python}" "${script_dir}/inference.py" \
  --model "${model}" \
  --base-url "${base_url}" \
  --output-dir "${result_dir}" \
  --ner-num-predict "${ner_num_predict}" \
  --re-num-predict "${re_num_predict}" \
  "${think_args[@]}" \
  "${icl_args[@]}" \
  "${context_args[@]}" \
  "${inference_mode[@]}"

"${venv_python}" "${script_dir}/evaluate.py" \
  "${result_dir}/prediction.json" \
  --output "${result_dir}/scores.json"

echo "Scores written to ${result_dir}/scores.json"
sed -n '1,100p' "${result_dir}/scores.json"
