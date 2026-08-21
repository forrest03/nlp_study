#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly LOG_DIR="/data/logs"
readonly LOG_FILE="${LOG_DIR}/embedding-server.log"
readonly PID_FILE="${LOG_DIR}/embedding-server.pid"

HOST="${EMBEDDING_HOST:-0.0.0.0}"
PORT="${EMBEDDING_PORT:-8354}"
GPU_INDEX="${EMBEDDING_GPU_INDEX:-1}"
PYTHON_BIN="${EMBEDDING_PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"

mkdir -p "${LOG_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(cat "${PID_FILE}")"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    echo "embedding-server is already running, pid=${EXISTING_PID}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

cd "${PROJECT_ROOT}"
nohup "${PYTHON_BIN}" -m uvicorn main:app --host "${HOST}" --port "${PORT}" \
  >>"${LOG_FILE}" 2>&1 &

SERVER_PID="$!"
echo "${SERVER_PID}" > "${PID_FILE}"

echo "embedding-server started, pid=${SERVER_PID}, gpu=${GPU_INDEX}, python=${PYTHON_BIN}, log=${LOG_FILE}"
