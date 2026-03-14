# XTTS Dataset Fetch And Publish

## Status

Planned

## Goal

Standardize how XTTS datasets are published to and fetched from Cloudflare R2 for remote training.

## Scope

- publish XTTS-ready datasets to the agreed R2 prefix
- document or script fetch on the training VM
- preserve the portable package contract

## Remote Prefix

- `training/datasets/<namespace>/<voice>/<dataset-name>/`

## Must Support

- public R2 access as the default path
- `rclone`-based fetch workflow
- dataset packages containing:
  - `metadata_train.csv`
  - `metadata_eval.csv`
  - `wavs/`

## Notes

- `scp` remains fallback only
- this task is about transport and contract, not trainer logic
