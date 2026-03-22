#!/usr/bin/env bash
set -euo pipefail

# Step 3: promote best/last artifacts after review.
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
export RUN_ID="${RUN_ID:-$(cat /workspace/last-run-id.txt)}"

export RCLONE_CONFIG_R2_TYPE="${RCLONE_CONFIG_R2_TYPE:-s3}"
export RCLONE_CONFIG_R2_PROVIDER="${RCLONE_CONFIG_R2_PROVIDER:-Cloudflare}"
export RCLONE_CONFIG_R2_ENDPOINT="${RCLONE_CONFIG_R2_ENDPOINT:-https://318a76701b3d283740ba549a321cee13.r2.cloudflarestorage.com}"

export RUNS_REMOTE_PREFIX="${RUNS_REMOTE_PREFIX:-}"
export MODELS_REMOTE_PREFIX="${MODELS_REMOTE_PREFIX:-}"
export NAMESPACE="${NAMESPACE:-}"
export VOICE="${VOICE:-}"

require_env \
  RCLONE_CONFIG_R2_TYPE \
  RCLONE_CONFIG_R2_PROVIDER \
  RCLONE_CONFIG_R2_ENDPOINT \
  RCLONE_CONFIG_R2_ACCESS_KEY_ID \
  RCLONE_CONFIG_R2_SECRET_ACCESS_KEY \
  RUNS_REMOTE_PREFIX \
  MODELS_REMOTE_PREFIX \
  NAMESPACE \
  VOICE \
  RUN_ID

cd "$REPO_DIR"
source "$REPO_DIR/.venv-train/bin/activate"

python3 "$REPO_DIR/scripts/jobs/promote_xtts_run.py" \
  --namespace "$NAMESPACE" \
  --voice "$VOICE" \
  --run-id "$RUN_ID" \
  --remote-prefix "$RUNS_REMOTE_PREFIX" \
  --models-remote-prefix "$MODELS_REMOTE_PREFIX"

echo
echo "Promoted artifacts:"
echo "$RUNS_REMOTE_PREFIX/$NAMESPACE/$VOICE/$RUN_ID/artifacts/best"
echo "$RUNS_REMOTE_PREFIX/$NAMESPACE/$VOICE/$RUN_ID/artifacts/last"
echo "$MODELS_REMOTE_PREFIX/$NAMESPACE/$VOICE/current.json"
