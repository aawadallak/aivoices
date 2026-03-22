#!/usr/bin/env bash
set -euo pipefail

# Fast dependency repair path for XTTS training.
# Use this first when the venv already exists and only a version pin changed.

export REPO_DIR="${REPO_DIR:-/workspace/aivoices}"
export VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv-train}"

cd "$REPO_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[error] virtualenv not found: $VENV_DIR" >&2
  echo "Run the bootstrap first." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
if [[ -f "$VENV_DIR/bin/xtts-runtime-env.sh" ]]; then
  source "$VENV_DIR/bin/xtts-runtime-env.sh"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -U --no-cache-dir --force-reinstall "transformers==4.46.2"
python -m pip install -U --no-cache-dir --force-reinstall "coqpit-config==0.1.2"

python - <<'PY'
mods = [
    ("torch", "import torch"),
    ("trainer", "import trainer"),
    ("TTS", "import TTS"),
]
failed = False
for name, stmt in mods:
    try:
        exec(stmt, {})
        print(f"[deps-fast] {name}: OK")
    except Exception as exc:
        failed = True
        print(f"[deps-fast] {name}: FAIL -> {exc!r}")
if failed:
    raise SystemExit(1)
PY

echo "Fast dependency repair finished."
