#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export TMPDIR="${PROJECT_ROOT}/.tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.pip_cache"
export TORCH_HOME="${PROJECT_ROOT}/.torch"

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${TORCH_HOME}"

if [ "$#" -lt 1 ]; then
  echo "Usage: ./run_in_env.sh <script> [args...]"
  exit 1
fi

exec "${PYTHON_BIN}" "$@"

