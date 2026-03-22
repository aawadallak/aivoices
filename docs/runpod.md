# Runpod Bootstrap

This document describes the quickest bootstrap path for XTTS training on Runpod using the base image:

- `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`

Runpod documents that this image family comes preloaded with PyTorch, Python, CUDA, and cuDNN, and that Pod templates can override the container start command. Runpod also documents that its base images include helpful base packages such as `runpodctl` and `nginx`.

Sources:

- https://www.runpod.io/articles/guides/pytorch-2-8-cuda-12-8
- https://docs.runpod.io/pods/templates/manage-templates
- https://docs.runpod.io/serverless/development/dual-mode-worker

## Recommendation

For the first training runs:

- use the official Runpod PyTorch base image
- override the start command
- let the start command clone this repository and execute a bootstrap script
- pin the operational Python baseline to `3.11.9`

This is faster to adopt than maintaining a custom image immediately.

## Python Baseline

For this training flow, the current operational baseline is:

- `Python 3.11.9`

This is a conservative project choice based on prior successful runs, not an official Coqui requirement for that exact patch version.

The bootstrap now installs a managed `Python 3.11.9` runtime with `micromamba` and builds the training virtualenv from that interpreter, so the base image no longer needs to ship the exact patch version.

## Bootstrap Script

Use:

- [`scripts/bootstrap/runpod_xtts_bootstrap.sh`](/home/awadallak/aivoices/scripts/bootstrap/runpod_xtts_bootstrap.sh)

It currently does:

- installs missing system packages such as `rclone`
- installs `micromamba` when needed
- creates a managed `Python 3.11.9` runtime
- clones or updates this repository
- creates a virtualenv
- installs `requirements-xtts-train.txt`
- optionally fetches a dataset from R2
- optionally starts training
- keeps the container alive afterward

## Dependency Notes

The Runpod image currently preloads a mismatched nightly stack:

- `torch 2.8.0.dev...`
- `torchaudio 2.6.0.dev...`

Leaving `torch`, `torchaudio`, and Coqui packages unpinned causes pip to backtrack across incompatible combinations and attempt upgrades toward `torch 2.10` / `torchaudio 2.10`.

The training requirements file now pins a coherent stack for this image family:

- `coqui-tts==0.25.3`
- `coqui-tts-trainer==0.2.2`
- `coqpit-config==0.1.2`
- `torch==2.8.0`
- `torchaudio==2.8.0`
- `numpy==1.26.4`

The recommended default for long-lived VMs is:

- `BOOTSTRAP_ONLY=1`

In that mode, the VM is prepared but does not automatically fetch datasets or launch training.

## Suggested Start Command

In the Runpod template, set the container start command to:

```bash
bash -lc 'set -euo pipefail; mkdir -p /workspace; cd /workspace; if [ ! -d aivoices/.git ]; then git clone --branch main https://github.com/aawadallak/aivoices.git aivoices; fi; bash /workspace/aivoices/scripts/bootstrap/runpod_xtts_bootstrap.sh'
```

If you want it to always refresh the repo before running:

```bash
bash -lc 'set -euo pipefail; mkdir -p /workspace; if [ ! -d /workspace/aivoices/.git ]; then git clone --branch main https://github.com/aawadallak/aivoices.git /workspace/aivoices; fi; cd /workspace/aivoices; git fetch --all --tags; git checkout main; git pull --ff-only origin main; bash scripts/bootstrap/runpod_xtts_bootstrap.sh'
```

## Suggested Environment Variables

Set these in the Pod template when you want the bootstrap to fetch a dataset and start training automatically:

```text
REPO_URL=https://github.com/aawadallak/aivoices.git
REPO_REF=main
BOOTSTRAP_ONLY=1
DATASET_REMOTE_PREFIX=r2:aivoices/training/datasets
DATASET_NAMESPACE=square-spongebob
DATASET_VOICE=bob-esponja
DATASET_NAME=bob-esponja-clean-v1
RUN_NAMESPACE=square-spongebob
RUN_VOICE=bob-esponja
RUN_EPOCHS=10
RUN_BATCH_SIZE=4
RUN_GRAD_ACCUM=8
```

For a VM that stays up and is used manually, the recommended minimal set is:

```text
REPO_URL=https://github.com/aawadallak/aivoices.git
REPO_REF=main
REQUIRED_PYTHON_VERSION=3.11.9
BOOTSTRAP_ONLY=1

RCLONE_CONFIG_R2_TYPE=s3
RCLONE_CONFIG_R2_PROVIDER=Cloudflare
RCLONE_CONFIG_R2_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
RCLONE_CONFIG_R2_ACCESS_KEY_ID=<ACCESS_KEY_ID>
RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=<SECRET_ACCESS_KEY>

DATASET_REMOTE_PREFIX=r2:aivoices/training/datasets
RUNS_REMOTE_PREFIX=r2:aivoices/training/runs
MODELS_REMOTE_PREFIX=r2:aivoices/training/models
```

Then fetch and train manually after the VM is ready.

Optional:

```text
RUN_EXTRA_ARGS=--save-step 500
REQUIRED_PYTHON_VERSION=3.11.9
MICROMAMBA_ROOT_PREFIX=/workspace/micromamba
PYTHON_ENV_NAME=xtts-py3119
```

## From Zero Checklist

Use this sequence on a fresh Runpod pod created from:

- `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`

### 1. Set Pod Environment Variables

Minimum bootstrap variables:

```text
REPO_URL=https://github.com/aawadallak/aivoices.git
REPO_REF=main
REQUIRED_PYTHON_VERSION=3.11.9
BOOTSTRAP_ONLY=1
```

If the pod will fetch datasets from R2, also set:

```text
RCLONE_CONFIG_R2_TYPE=s3
RCLONE_CONFIG_R2_PROVIDER=Cloudflare
RCLONE_CONFIG_R2_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
RCLONE_CONFIG_R2_ACCESS_KEY_ID=<ACCESS_KEY_ID>
RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=<SECRET_ACCESS_KEY>

DATASET_REMOTE_PREFIX=r2:aivoices/training/datasets
RUNS_REMOTE_PREFIX=r2:aivoices/training/runs
MODELS_REMOTE_PREFIX=r2:aivoices/training/models
```

### 2. Set the Container Start Command

Recommended:

```bash
bash -lc 'set -euo pipefail; mkdir -p /workspace; if [ ! -d /workspace/aivoices/.git ]; then git clone --branch main https://github.com/aawadallak/aivoices.git /workspace/aivoices; fi; cd /workspace/aivoices; git fetch --all --tags; git checkout main; git pull --ff-only origin main; bash scripts/bootstrap/runpod_xtts_bootstrap.sh'
```

### 3. Wait for Bootstrap to Finish

The bootstrap should:

- install system packages
- install `micromamba`
- create managed Python `3.11.9`
- create `/workspace/aivoices/.venv-train`
- install the pinned XTTS stack

### 4. Verify the Runtime

Open a shell in the pod and run:

```bash
cd /workspace/aivoices
source .venv-train/bin/activate

python --version
python - <<'PY'
import torch, torchaudio, trainer, TTS
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("trainer ok")
print("TTS ok")
PY
```

### 5. Fetch a Dataset

Example:

```bash
cd /workspace/aivoices
source .venv-train/bin/activate

python scripts/jobs/fetch_xtts_dataset.py \
  --remote-prefix r2:aivoices/training/datasets \
  --namespace square-spongebob \
  --voice bob-esponja \
  --dataset-name bob-esponja-clean-v1 \
  --output-dir /workspace/datasets/bob-esponja-clean-v1
```

### 6. Start Training

Example:

```bash
export NAMESPACE=square-spongebob
export VOICE=bob-esponja
export DATASET_NAME=bob-esponja-clean-v1
export DATASET_DIR=/workspace/datasets/bob-esponja-clean-v1
export EPOCHS=10
export BATCH_SIZE=4
export GRAD_ACCUM=8

cd /workspace/aivoices
bash commands-train.sh
```

### 7. Watch the Logs

```bash
cat /workspace/last-run-id.txt
tail -f /workspace/logs/$(cat /workspace/last-run-id.txt).log
```

### 8. If the Pod Was Bootstrapped Before

Refresh the repo and rebuild the training environment:

```bash
cd /workspace/aivoices
git fetch --all --tags
git checkout main
git pull --ff-only origin main
bash commands-fix-train-deps.sh
```

### 9. Optional One-Shot Bootstrap Plus Train

If you want the pod to fetch the dataset and start training automatically during startup, set:

```text
BOOTSTRAP_ONLY=0
DATASET_NAMESPACE=square-spongebob
DATASET_VOICE=bob-esponja
DATASET_NAME=bob-esponja-clean-v1
RUN_NAMESPACE=square-spongebob
RUN_VOICE=bob-esponja
RUN_EPOCHS=10
RUN_BATCH_SIZE=4
RUN_GRAD_ACCUM=8
```

Then keep the same container start command from step 2.

## When To Move Beyond This

This bootstrap is good for the first real training runs.

If repeated startup time becomes annoying, the next step should be:

- build a custom training image
- preinstall XTTS dependencies into that image
- keep this script only for repo sync, dataset fetch, and train orchestration
