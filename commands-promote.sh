#!/usr/bin/env bash
set -euo pipefail

# Step 3: promote best/last artifacts after review.
# R2 credentials should come from the pod environment.

export REPO_DIR="/workspace/aivoices"
export RUN_ID="${RUN_ID:-$(cat /workspace/last-run-id.txt)}"

export RCLONE_CONFIG_R2_TYPE="s3"
export RCLONE_CONFIG_R2_PROVIDER="Cloudflare"
export RCLONE_CONFIG_R2_ENDPOINT="https://318a76701b3d283740ba549a321cee13.r2.cloudflarestorage.com"

export RUNS_REMOTE_PREFIX="r2:aivoices/training/runs"
export MODELS_REMOTE_PREFIX="r2:aivoices/training/models"
export NAMESPACE="square-spongebob"
export VOICE="bob-esponja"

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
