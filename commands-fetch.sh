#!/usr/bin/env bash
set -euo pipefail

# Fetch one XTTS dataset package from R2.

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
export RCLONE_CONFIG_R2_TYPE="${RCLONE_CONFIG_R2_TYPE:-s3}"
export RCLONE_CONFIG_R2_PROVIDER="${RCLONE_CONFIG_R2_PROVIDER:-Cloudflare}"
export RCLONE_CONFIG_R2_ENDPOINT="${RCLONE_CONFIG_R2_ENDPOINT:-https://318a76701b3d283740ba549a321cee13.r2.cloudflarestorage.com}"

export DATASET_REMOTE_PREFIX="${DATASET_REMOTE_PREFIX:-}"
export NAMESPACE="${NAMESPACE:-}"
export VOICE="${VOICE:-}"
export DATASET_NAME="${DATASET_NAME:-}"
export DATASET_DIR="${DATASET_DIR:-/workspace/datasets/${DATASET_NAME}}"

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

mkdir -p "$(dirname "$DATASET_DIR")"

cd "$REPO_DIR"
source "$REPO_DIR/.venv-train/bin/activate"

python3 "$REPO_DIR/scripts/jobs/fetch_xtts_dataset.py" \
  --remote-prefix "$DATASET_REMOTE_PREFIX" \
  --namespace "$NAMESPACE" \
  --voice "$VOICE" \
  --dataset-name "$DATASET_NAME" \
  --output-dir "$DATASET_DIR"

echo "Dataset fetched to: $DATASET_DIR"
