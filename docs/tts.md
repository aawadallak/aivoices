# TTS Export

This document records the repeatable export flow from curated dataset clips in this repository into XTTS-ready training packages and optional upload to Cloudflare R2.

## Purpose

The goal is to export one speaker at a time into a structure that downstream XTTS training pipelines can consume directly.

Expected output layout:

```text
datasets/<namespace>/exports/export_xtts/<export-name>/
  metadata_train.csv
  metadata_eval.csv
  wavs/
    <speaker>-0001.wav
    <speaker>-0002.wav
```

The CSV format must use `|` as the delimiter:

```text
audio_file|text|speaker_name
wavs/bob-esponja-0001.wav|Eu só quero saber quem de nós é o Carinha.|bob-esponja
```

## Source Material

The current XTTS export flow assumes the repository already produced:

- conservative matched clips under `metadata/<namespace>/speakers/runs/reference-match-*/`
- `clip-quality.csv` files for those runs
- provisional dataset exports under `datasets/<namespace>/exports/`

The current experimental exports for `square-spongebob` are:

- `datasets/square-spongebob/exports/provisional-quality-v1/`
- `datasets/square-spongebob/exports/provisional-quality-clean-v1/`

## XTTS Export Job

Use `scripts/jobs/export_xtts_dataset.py`.

Examples:

```bash
source /home/awadallak/aivoices/.venv-dataset/bin/activate

python scripts/jobs/export_xtts_dataset.py \
  --namespace square-spongebob \
  --source-export provisional-quality-clean-v1 \
  --speaker bob-esponja \
  --output-name bob-esponja-clean-v1

python scripts/jobs/export_xtts_dataset.py \
  --namespace square-spongebob \
  --source-export provisional-quality-v1 \
  --speaker lula-molusco \
  --output-name lula-molusco-clean-usable-v1
```

Default filtering in the XTTS export job:

- one speaker per export
- `eval_ratio = 0.1`
- `max_duration = 11.0`
- `max_text_length = 200`

These defaults intentionally keep the output close to the training guidance used by the downstream XTTS pipeline.

## Current XTTS Exports

The current generated exports for `square-spongebob` are:

- `datasets/square-spongebob/exports/export_xtts/bob-esponja-clean-v1/`
- `datasets/square-spongebob/exports/export_xtts/bob-esponja-clean-usable-v1/`
- `datasets/square-spongebob/exports/export_xtts/lula-molusco-clean-v1/`
- `datasets/square-spongebob/exports/export_xtts/lula-molusco-clean-usable-v1/`

Current split counts:

- `bob-esponja-clean-v1`: `121` train, `14` eval
- `bob-esponja-clean-usable-v1`: `294` train, `33` eval
- `lula-molusco-clean-v1`: `153` train, `17` eval
- `lula-molusco-clean-usable-v1`: `318` train, `36` eval

## Cloudflare R2 Upload

If the training machine will pull packages from R2, use `rclone copy`.

The R2 remote can be injected through environment variables instead of a persistent config file.

Example environment setup:

```bash
export AWS_ACCESS_KEY_ID='<access-key>'
export AWS_SECRET_ACCESS_KEY='<secret-key>'
export AWS_REGION='auto'
export RCLONE_CONFIG_R2_TYPE='s3'
export RCLONE_CONFIG_R2_PROVIDER='Cloudflare'
export RCLONE_CONFIG_R2_ENDPOINT='<account-id>.r2.cloudflarestorage.com'
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2_ACL='private'
```

Upload one XTTS export:

```bash
rclone copy \
  datasets/square-spongebob/exports/export_xtts/bob-esponja-clean-v1 \
  r2:aivoices/training/datasets/square-spongebob/bob-esponja/bob-esponja-clean-v1 \
  -P
```

Upload targets currently in use:

- `r2:aivoices/training/datasets/square-spongebob/bob-esponja/bob-esponja-clean-v1`
- `r2:aivoices/training/datasets/square-spongebob/bob-esponja/bob-esponja-clean-usable-v1`
- `r2:aivoices/training/datasets/square-spongebob/lula-molusco/lula-molusco-clean-v1`
- `r2:aivoices/training/datasets/square-spongebob/lula-molusco/lula-molusco-clean-usable-v1`

## Pull On Training Machine

Examples:

```bash
rclone copy r2:aivoices/training/datasets/square-spongebob/bob-esponja/bob-esponja-clean-v1 ./bob-esponja-clean-v1 -P
rclone copy r2:aivoices/training/datasets/square-spongebob/bob-esponja/bob-esponja-clean-usable-v1 ./bob-esponja-clean-usable-v1 -P
rclone copy r2:aivoices/training/datasets/square-spongebob/lula-molusco/lula-molusco-clean-v1 ./lula-molusco-clean-v1 -P
rclone copy r2:aivoices/training/datasets/square-spongebob/lula-molusco/lula-molusco-clean-usable-v1 ./lula-molusco-clean-usable-v1 -P
```

## Notes

- Do not commit credentials into the repository.
- If credentials were exposed in terminal history or chat, rotate them.
- Prefer `clean` exports for the first training run and use `clean-usable` as a comparison run.

## Versioning Policy

For this XTTS flow, commit:

- export jobs and helper scripts
- docs that describe the export and upload process
- small QA manifests that help decide what should be exported next

Do not commit:

- generated XTTS packages under `datasets/<namespace>/exports/export_xtts/`
- copied `wavs/` payloads
- `metadata_train.csv` and `metadata_eval.csv` generated for one specific export
- training outputs, checkpoints, and experiment logs

Treat XTTS packages as delivery artifacts. Keep the recipe in git and publish the payload through object storage such as R2.
