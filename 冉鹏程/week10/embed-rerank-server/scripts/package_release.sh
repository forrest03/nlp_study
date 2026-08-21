#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly DIST_DIR="${PROJECT_ROOT}/dist"
readonly PACKAGE_DATE="$(date '+%Y%m%d')"
readonly TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
readonly PACKAGE_NAME="embedding-server-${TIMESTAMP}.zip"
readonly PACKAGE_PATH="${DIST_DIR}/${PACKAGE_NAME}"
readonly RELEASE_DIR_NAME="embedding-server-${PACKAGE_DATE}"
readonly STAGING_DIR="${DIST_DIR}/${RELEASE_DIR_NAME}"

mkdir -p "${DIST_DIR}"

cd "${PROJECT_ROOT}"
rm -f "${PACKAGE_PATH}"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

cp -R app "${STAGING_DIR}/"
cp -R scripts "${STAGING_DIR}/"
cp -R tests "${STAGING_DIR}/"
cp README.md "${STAGING_DIR}/"
cp requirements.txt "${STAGING_DIR}/"
cp main.py "${STAGING_DIR}/"

find "${STAGING_DIR}" -type d \( -name ".idea" -o -name ".pytest_cache" \) -prune -exec rm -rf {} +
find "${STAGING_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${STAGING_DIR}" -type f \( -name "*.pyc" -o -name ".DS_Store" -o -name "Thumbs.db" \) -delete

cd "${DIST_DIR}"
zip -r "${PACKAGE_PATH}" "${RELEASE_DIR_NAME}"
rm -rf "${STAGING_DIR}"

echo "package created: ${PACKAGE_PATH}"
