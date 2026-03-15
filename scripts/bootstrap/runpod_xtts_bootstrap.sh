#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:aawadallak/aivoices.git}"
REPO_REF="${REPO_REF:-main}"
REPO_DIR="${REPO_DIR:-/workspace/aivoices}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv-train}"
REQUIRED_PYTHON_VERSION="${REQUIRED_PYTHON_VERSION:-3.11.9}"
BOOTSTRAP_ONLY="${BOOTSTRAP_ONLY:-1}"
MICROMAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT_PREFIX:-/workspace/micromamba}"
PYTHON_ENV_NAME="${PYTHON_ENV_NAME:-xtts-py3119}"

DATASET_REMOTE_PREFIX="${DATASET_REMOTE_PREFIX:-}"
DATASET_NAMESPACE="${DATASET_NAMESPACE:-}"
DATASET_VOICE="${DATASET_VOICE:-}"
DATASET_NAME="${DATASET_NAME:-}"
DATASET_DIR="${DATASET_DIR:-/workspace/datasets/${DATASET_NAME:-dataset}}"

RUN_NAMESPACE="${RUN_NAMESPACE:-$DATASET_NAMESPACE}"
RUN_VOICE="${RUN_VOICE:-$DATASET_VOICE}"
RUN_EPOCHS="${RUN_EPOCHS:-10}"
RUN_BATCH_SIZE="${RUN_BATCH_SIZE:-4}"
RUN_GRAD_ACCUM="${RUN_GRAD_ACCUM:-8}"
RUN_EXTRA_ARGS="${RUN_EXTRA_ARGS:-}"

log() {
  printf '[bootstrap] %s\n' "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing command: $1" >&2
    exit 1
  }
}

install_system_deps() {
  log "installing system dependencies"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    bzip2 \
    rclone
}

install_micromamba() {
  if command -v micromamba >/dev/null 2>&1; then
    return 0
  fi
  log "installing micromamba"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /tmp bin/micromamba
  install -m 0755 /tmp/bin/micromamba /usr/local/bin/micromamba
}

ensure_python_runtime() {
  export MAMBA_ROOT_PREFIX="$MICROMAMBA_ROOT_PREFIX"
  local env_prefix="$MICROMAMBA_ROOT_PREFIX/envs/$PYTHON_ENV_NAME"
  local python_bin="$env_prefix/bin/python"
  if [[ ! -x "$python_bin" ]]; then
    log "creating micromamba env $PYTHON_ENV_NAME with python $REQUIRED_PYTHON_VERSION"
    micromamba create -y -n "$PYTHON_ENV_NAME" "python=$REQUIRED_PYTHON_VERSION"
  fi
  local detected
  detected="$("$python_bin" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"
  if [[ "$detected" != "$REQUIRED_PYTHON_VERSION" ]]; then
    echo "managed python version mismatch: expected $REQUIRED_PYTHON_VERSION, got $detected" >&2
    exit 1
  fi
  export TRAIN_PYTHON_BIN="$python_bin"
  log "using managed python at $TRAIN_PYTHON_BIN"
}

sync_repo() {
  mkdir -p "$(dirname "$REPO_DIR")"
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "cloning repository into $REPO_DIR"
    git clone --branch "$REPO_REF" "$REPO_URL" "$REPO_DIR"
  else
    log "updating repository at $REPO_DIR"
    git -C "$REPO_DIR" fetch --all --tags
    git -C "$REPO_DIR" checkout "$REPO_REF"
    git -C "$REPO_DIR" pull --ff-only origin "$REPO_REF"
  fi
}

prepare_venv() {
  log "preparing virtualenv at $VENV_DIR"
  "$TRAIN_PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip uninstall -y coqpit >/dev/null 2>&1 || true
  python -m pip install -r "$REPO_DIR/requirements-xtts-train.txt"
}

fetch_dataset() {
  if [[ "$BOOTSTRAP_ONLY" == "1" ]]; then
    log "BOOTSTRAP_ONLY=1; skipping dataset fetch"
    return 0
  fi
  if [[ -z "$DATASET_REMOTE_PREFIX" || -z "$DATASET_NAMESPACE" || -z "$DATASET_VOICE" || -z "$DATASET_NAME" ]]; then
    log "dataset remote variables not fully set; skipping dataset fetch"
    return 0
  fi
  log "fetching dataset $DATASET_NAMESPACE/$DATASET_VOICE/$DATASET_NAME"
  python "$REPO_DIR/scripts/jobs/fetch_xtts_dataset.py" \
    --remote-prefix "$DATASET_REMOTE_PREFIX" \
    --namespace "$DATASET_NAMESPACE" \
    --voice "$DATASET_VOICE" \
    --dataset-name "$DATASET_NAME" \
    --output-dir "$DATASET_DIR"
}

run_training() {
  if [[ "$BOOTSTRAP_ONLY" == "1" ]]; then
    log "BOOTSTRAP_ONLY=1; skipping training"
    return 0
  fi
  if [[ -z "$RUN_NAMESPACE" || -z "$RUN_VOICE" ]]; then
    log "RUN_NAMESPACE or RUN_VOICE missing; skipping training"
    return 0
  fi
  log "starting XTTS training"
  # shellcheck disable=SC2086
  python "$REPO_DIR/scripts/jobs/train_xtts.py" \
    --namespace "$RUN_NAMESPACE" \
    --dataset-dir "$DATASET_DIR" \
    --voice "$RUN_VOICE" \
    --epochs "$RUN_EPOCHS" \
    --batch-size "$RUN_BATCH_SIZE" \
    --grad-accum "$RUN_GRAD_ACCUM" \
    $RUN_EXTRA_ARGS
}

main() {
  need_cmd python3
  install_system_deps
  install_micromamba
  ensure_python_runtime
  sync_repo
  prepare_venv
  fetch_dataset
  run_training
  log "bootstrap finished; keeping container alive"
  tail -f /dev/null
}

main "$@"
