#!/usr/bin/env bash
set -euo pipefail

# Step 2: generate smoke samples and upload them for review.
# R2 credentials should come from the pod environment.

export REPO_DIR="/workspace/aivoices"
export RUN_ID="${RUN_ID:-$(cat /workspace/last-run-id.txt)}"

export RCLONE_CONFIG_R2_TYPE="s3"
export RCLONE_CONFIG_R2_PROVIDER="Cloudflare"
export RCLONE_CONFIG_R2_ENDPOINT="https://318a76701b3d283740ba549a321cee13.r2.cloudflarestorage.com"

export RUNS_REMOTE_PREFIX="r2:aivoices/training/runs"
export NAMESPACE="square-spongebob"
export VOICE="bob-esponja"

cd "$REPO_DIR"
source "$REPO_DIR/.venv-train/bin/activate"

python3 "$REPO_DIR/scripts/jobs/export_xtts_smoke_review.py" \
  --namespace "$NAMESPACE" \
  --voice "$VOICE" \
  --run-id "$RUN_ID" \
  --speaker-wav "$REPO_DIR/metadata/square-spongebob/speakers/references/bob-esponja/approved/square-spongebob-bob-esponja-reference-001-seed.wav" \
  --remote-prefix "$RUNS_REMOTE_PREFIX"

echo
echo "Review remote path:"
echo "$RUNS_REMOTE_PREFIX/$NAMESPACE/$VOICE/$RUN_ID"
echo
echo "After listening review, create:"
echo "$REPO_DIR/training/xtts/$NAMESPACE/$VOICE/$RUN_ID/promotion.json"
echo
echo "Example:"
cat <<'EOF'
{
  "promote_checkpoint": "best_model",
  "keep_last": true,
  "notes": "melhor naturalidade e estabilidade"
}
EOF
