# XTTS Trainer Core

## Status

Planned

## Goal

Implement the first-party XTTS trainer engine and canonical CLI for this repository.

## Scope

- add the training engine module
- add the canonical CLI entrypoint
- consume portable XTTS packages through `--dataset-dir`
- implement the mandatory preflight validation
- write `run.json`

## Target Files

- `scripts/pipeline/xtts_training.py`
- `scripts/jobs/train_xtts.py`
- `docs/training.md`

## Must Support

- `--dataset-dir`
- `--language`
- `--output-dir`
- `--run-id` or automatic run-id generation
- container-first execution

## Must Not Assume

- that the full dataset-prep workspace is present on the VM
- that datasets are resolved from `namespace/source-export` on the training VM

## Notes

- use `continue_path` as the primary resume direction
- keep `restore_path` as advanced mode only
