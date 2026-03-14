# XTTS Smoke And Review Flow

## Status

Planned

## Goal

Implement the smoke-test generation and review export flow for candidate checkpoints.

## Scope

- load the canonical `pt-BR` smoke-test file
- generate smoke samples for candidate checkpoints
- write `candidates.json`
- publish review artifacts to the run prefix in R2

## Must Support

- `pt-BR` smoke-test phrases from `metadata/tts/smoke-tests/pt-br-default-v1.txt`
- candidate set capped at 3 checkpoints
- `last` always included
- `best_loss` included when different from `last`
- optional single intermediate candidate

## Review Contract

- listening review happens outside the training VM
- smoke samples are uploaded before promotion
- checkpoints without smoke samples cannot become `best`
