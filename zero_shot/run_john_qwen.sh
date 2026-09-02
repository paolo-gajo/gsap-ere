#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

model="${MODEL:-qwen3.8:27b}"
base_url="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
result_dir="${RESULT_DIR:-${script_dir}/results/qwen3.8-27b-ollama-john}"

if command -v ollama >/dev/null 2>&1; then
  ollama_bin="$(command -v ollama)"
elif [[ -x /home/pgajo/.local/bin/ollama ]]; then
  ollama_bin=/home/pgajo/.local/bin/ollama
else
  echo "error: Ollama not found; install it in PATH or /home/pgajo/.local/bin" >&2
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
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

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

inference_args=(
  --model "${model}"
  --base-url "${base_url}"
  --prompt "${script_dir}/prompt.md"
  --output-dir "${result_dir}"
  --think high
  --temperature 0.0
  --top-p 0.95
  --top-k 20
  --seed 0
  --num-ctx 65536
  --num-predict 32768
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  inference_args+=(--overwrite)
fi

python3 "${script_dir}/inference.py" "${inference_args[@]}"
python3 "${script_dir}/evaluate.py" "${result_dir}/prediction.json" \
  >"${result_dir}/scores.json"

echo "Scores written to ${result_dir}/scores.json"
sed -n '1,100p' "${result_dir}/scores.json"
