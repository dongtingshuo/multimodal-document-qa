#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/models/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python_bin="${DOCQA_PYTHON:-$PWD/.venv/bin/python}"
if [ ! -x "$python_bin" ]; then
  echo "请先按 README 创建项目环境：$PWD/.venv/bin/python" >&2
  exit 1
fi

# Port selection, child cleanup and single-instance locking live in Python.
# Keep braces around shell variables next to non-ASCII text on macOS Bash 3.2.
exec "${python_bin}" scripts/launch.py "$@"
