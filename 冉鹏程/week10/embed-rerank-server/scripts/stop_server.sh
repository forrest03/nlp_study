#!/usr/bin/env bash

set -euo pipefail

readonly LOG_DIR="/data/logs"
readonly PID_FILE="${LOG_DIR}/embedding-server.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "embedding-server is not running"
  exit 0
fi

SERVER_PID="$(cat "${PID_FILE}")"

if [[ -z "${SERVER_PID}" ]]; then
  rm -f "${PID_FILE}"
  echo "embedding-server pid file is empty and has been cleaned"
  exit 0
fi

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "embedding-server process not found, stale pid file removed"
  exit 0
fi

kill "${SERVER_PID}"

for _ in {1..30}; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "embedding-server stopped, pid=${SERVER_PID}"
    exit 0
  fi
  sleep 1
done

echo "embedding-server did not stop within 30 seconds, pid=${SERVER_PID}" >&2
exit 1
