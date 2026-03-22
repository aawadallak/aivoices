# Training

This document describes the first repository-native XTTS training flow.

## Execution Model

- dataset preparation happens on bare metal
- XTTS packages are published to R2
- training runs on ephemeral cloud GPU VMs
- the preferred mode is container-first

## Runtime

Container scaffold:

- [`Dockerfile.xtts-train`](/home/awadallak/aivoices/Dockerfile.xtts-train)
- [`requirements-xtts-train.txt`](/home/awadallak/aivoices/requirements-xtts-train.txt)

The XTTS training requirements are pinned to a coherent stack for the current Runpod base image. In particular, keep `torch`, `torchaudio`, and Coqui packages pinned together; leaving them floating causes pip to backtrack into incompatible combinations and can try to upgrade the image away from its CUDA-matched baseline.

The container is intended to run on a VM that already has NVIDIA runtime support.

Current operational Python baseline for the first training runs:

- `Python 3.11.9`

Treat this as a pinned project baseline until the training stack is validated more broadly.
The Runpod bootstrap installs this version with `micromamba` and creates the training virtualenv from that managed interpreter.

For long-lived VMs, prefer bootstrap-only startup:

- `BOOTSTRAP_ONLY=1`

Then fetch datasets and start training manually after the machine is ready.

## Dataset Contract

The canonical training input is a portable XTTS package directory containing:

- `metadata_train.csv`
- `metadata_eval.csv`
- `wavs/`
- optional `dataset-info.json`

The training CLI consumes:

```bash
python scripts/jobs/train_xtts.py \
  --namespace square-spongebob \
  --dataset-dir /data/bob-esponja-clean-v1
```

## R2 Dataset Transport

Publish a dataset:

```bash
python scripts/jobs/publish_xtts_dataset.py \
  --dataset-dir datasets/square-spongebob/exports/export_xtts/bob-esponja-clean-v1 \
  --remote-prefix r2:aivoices/training/datasets \
  --namespace square-spongebob \
  --voice bob-esponja \
  --dataset-name bob-esponja-clean-v1
```

Fetch a dataset on the training VM:

```bash
python scripts/jobs/fetch_xtts_dataset.py \
  --remote-prefix r2:aivoices/training/datasets \
  --namespace square-spongebob \
  --voice bob-esponja \
  --dataset-name bob-esponja-clean-v1 \
  --output-dir /data/bob-esponja-clean-v1
```

## First Training Run

Example:

```bash
python scripts/jobs/train_xtts.py \
  --namespace square-spongebob \
  --dataset-dir /data/bob-esponja-clean-v1 \
  --voice bob-esponja \
  --epochs 10 \
  --batch-size 4 \
  --grad-accum 8
```

Run output layout:

```text
training/xtts/<namespace>/<voice>/<run-id>/
  run.json
  trainer/
```

## Smoke Review

Generate smoke-test samples for candidate checkpoints:

```bash
python scripts/jobs/export_xtts_smoke_review.py \
  --namespace square-spongebob \
  --voice bob-esponja \
  --run-id bob-esponja-bob-esponja-clean-v1-20260314-154500 \
  --speaker-wav metadata/square-spongebob/speakers/references/bob-esponja/approved/square-spongebob-bob-esponja-reference-001-seed.wav
```

This writes:

- `candidates.json`
- `samples/<checkpoint-id>/sample-*.wav`
- `sample-manifest.csv`

## Promotion

After external listening review, create `promotion.json` under the run directory and promote:

```bash
python scripts/jobs/promote_xtts_run.py \
  --namespace square-spongebob \
  --voice bob-esponja \
  --run-id bob-esponja-bob-esponja-clean-v1-20260314-154500
```

This creates:

- `artifacts/best/`
- `artifacts/last/`
- `training/xtts/models/<namespace>/<voice>/current.json`

## Resume

Current design direction:

- `continue_path` is the primary resume mechanism
- `restore_path` is advanced mode
- `artifacts/last/` is the intended portable resume bundle

Resume still needs empirical validation before being considered fully closed.
