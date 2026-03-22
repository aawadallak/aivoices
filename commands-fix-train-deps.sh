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
if [[ -f "$VENV_DIR/bin/xtts-runtime-env.sh" ]]; then
  source "$VENV_DIR/bin/xtts-runtime-env.sh"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip uninstall -y coqpit coqpit-config coqui-tts coqui-tts-trainer trainer TTS || true
python - <<'PY'
import shutil
import site
from pathlib import Path

for base in site.getsitepackages():
    root = Path(base)
    for pattern in ("coqpit*", "TTS*", "trainer*"):
        for path in root.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink(missing_ok=True)
PY
python -m pip install --no-cache-dir --force-reinstall -r "$REPO_DIR/requirements-xtts-train.txt"

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
        print(f"[deps] {name}: OK")
    except Exception as exc:
        failed = True
        print(f"[deps] {name}: FAIL -> {exc!r}")
if failed:
    raise SystemExit(1)
import coqpit
print(f"[deps] coqpit module: {getattr(coqpit, '__file__', 'unknown')}")
PY

echo "Dependency refresh finished."
