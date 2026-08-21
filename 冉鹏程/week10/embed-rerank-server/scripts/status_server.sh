#!/usr/bin/env bash

set -euo pipefail

readonly LOG_DIR="/data/logs"
readonly LOG_FILE="${LOG_DIR}/embedding-server.log"
readonly PID_FILE="${LOG_DIR}/embedding-server.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "embedding-server is not running"
  exit 1
fi

SERVER_PID="$(cat "${PID_FILE}")"

if [[ -z "${SERVER_PID}" ]]; then
  echo "embedding-server pid file is empty"
  exit 1
fi

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "embedding-server is not running, stale pid=${SERVER_PID}"
  exit 1
fi

echo "embedding-server is running, pid=${SERVER_PID}, log=${LOG_FILE}"
