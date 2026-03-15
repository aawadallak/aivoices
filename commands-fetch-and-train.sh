#!/usr/bin/env bash
set -euo pipefail

# Step 1: fetch the XTTS package and start training in nohup.
# R2 credentials should come from the pod environment.

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

export RCLONE_CONFIG_R2_TYPE="${RCLONE_CONFIG_R2_TYPE:-s3}"
export RCLONE_CONFIG_R2_PROVIDER="${RCLONE_CONFIG_R2_PROVIDER:-Cloudflare}"
export RCLONE_CONFIG_R2_ENDPOINT="${RCLONE_CONFIG_R2_ENDPOINT:-https://318a76701b3d283740ba549a321cee13.r2.cloudflarestorage.com}"

export DATASET_REMOTE_PREFIX="${DATASET_REMOTE_PREFIX:-}"
export NAMESPACE="${NAMESPACE:-}"
export VOICE="${VOICE:-}"
export DATASET_NAME="${DATASET_NAME:-}"
export DATASET_DIR="${DATASET_DIR:-/workspace/datasets/${DATASET_NAME}}"
export RUN_ID="${VOICE}-${DATASET_NAME}-$(date -u +%Y%m%d-%H%M%S)"

require_env \
  RCLONE_CONFIG_R2_TYPE \
  RCLONE_CONFIG_R2_PROVIDER \
  RCLONE_CONFIG_R2_ENDPOINT \
  RCLONE_CONFIG_R2_ACCESS_KEY_ID \
  RCLONE_CONFIG_R2_SECRET_ACCESS_KEY \
  DATASET_REMOTE_PREFIX \
  NAMESPACE \
  VOICE \
  DATASET_NAME

mkdir -p "$LOG_DIR" "$(dirname "$DATASET_DIR")"

cd "$REPO_DIR"
source "$REPO_DIR/.venv-train/bin/activate"

python3 "$REPO_DIR/scripts/jobs/fetch_xtts_dataset.py" \
  --remote-prefix "$DATASET_REMOTE_PREFIX" \
  --namespace "$NAMESPACE" \
  --voice "$VOICE" \
  --dataset-name "$DATASET_NAME" \
  --output-dir "$DATASET_DIR"

echo "$RUN_ID" | tee /workspace/last-run-id.txt
nohup python3 "$REPO_DIR/scripts/jobs/train_xtts.py" \
  --namespace "$NAMESPACE" \
  --dataset-dir "$DATASET_DIR" \
  --voice "$VOICE" \
  --run-id "$RUN_ID" \
  --epochs 10 \
  --batch-size 4 \
  --grad-accum 8 \
  --save-step 1000 \
  > "$LOG_DIR/${RUN_ID}.log" 2>&1 < /dev/null &

echo "Training PID: $!"
echo "RUN_ID saved to /workspace/last-run-id.txt"
echo "Tail logs with:"
echo "tail -f $LOG_DIR/${RUN_ID}.log"
