#!/usr/bin/env bash
set -euo pipefail

# Step 1: fetch the XTTS package and start training in nohup.
# R2 credentials should come from the pod environment.

export REPO_DIR="/workspace/aivoices"
export DATASET_DIR="/workspace/datasets/bob-esponja-clean-v1"
export LOG_DIR="/workspace/logs"

export RCLONE_CONFIG_R2_TYPE='s3'
export RCLONE_CONFIG_R2_PROVIDER='Cloudflare'
export RCLONE_CONFIG_R2_ENDPOINT='https://318a76701b3d283740ba549a321cee13.r2.cloudflarestorage.com'

export DATASET_REMOTE_PREFIX="r2:aivoices/training/datasets"
export NAMESPACE="square-spongebob"
export VOICE="bob-esponja"
export DATASET_NAME="bob-esponja-clean-v1"
export RUN_ID="${VOICE}-${DATASET_NAME}-$(date -u +%Y%m%d-%H%M%S)"

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
