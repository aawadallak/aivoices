#!/usr/bin/env bash
set -euo pipefail

# Start XTTS training in nohup using an already fetched dataset.

require_env() {
  local missing=0
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "[error] missing env: $name" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    exit 1
  fi
}

export REPO_DIR="${REPO_DIR:-/workspace/aivoices}"
export LOG_DIR="${LOG_DIR:-/workspace/logs}"

export NAMESPACE="${NAMESPACE:-}"
export VOICE="${VOICE:-}"
export DATASET_NAME="${DATASET_NAME:-}"
export DATASET_DIR="${DATASET_DIR:-/workspace/datasets/${DATASET_NAME}}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export SAVE_STEP="${SAVE_STEP:-1000}"
export RUN_ID="${RUN_ID:-${VOICE}-${DATASET_NAME}-$(date -u +%Y%m%d-%H%M%S)}"

require_env \
  NAMESPACE \
  VOICE \
  DATASET_NAME

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[error] dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

cd "$REPO_DIR"
source "$REPO_DIR/.venv-train/bin/activate"

echo "$RUN_ID" | tee /workspace/last-run-id.txt
nohup python3 "$REPO_DIR/scripts/jobs/train_xtts.py" \
  --namespace "$NAMESPACE" \
  --dataset-dir "$DATASET_DIR" \
  --voice "$VOICE" \
  --run-id "$RUN_ID" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --save-step "$SAVE_STEP" \
  > "$LOG_DIR/${RUN_ID}.log" 2>&1 < /dev/null &

echo "Training PID: $!"
echo "RUN_ID saved to /workspace/last-run-id.txt"
echo "Tail logs with:"
echo "tail -f $LOG_DIR/${RUN_ID}.log"
