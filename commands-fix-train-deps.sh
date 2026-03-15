#!/usr/bin/env bash
set -euo pipefail

# Repair or refresh XTTS training dependencies inside the pod.

export REPO_DIR="${REPO_DIR:-/workspace/aivoices}"
export VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv-train}"

cd "$REPO_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[error] virtualenv not found: $VENV_DIR" >&2
  echo "Run the bootstrap first." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REPO_DIR/requirements-xtts-train.txt"
python -m pip install -U coqui-tts coqui-tts-trainer

echo "Dependency refresh finished."
