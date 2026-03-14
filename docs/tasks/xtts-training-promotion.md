# XTTS Promotion Flow

## Status

Planned

## Goal

Implement explicit checkpoint promotion using remote manifests.

## Scope

- consume `promotion.json`
- publish `artifacts/best/`
- publish `artifacts/last/`
- write or update `training/models/<namespace>/<voice>/current.json`

## Must Support

- promotion by explicit command only
- no automatic promotion on manifest appearance
- no checkpoint duplication into `models/`
- `current.json` as a lightweight pointer to the promoted run artifact

## Artifact Rules

- `best` and `last` stay under the run prefix
- `models/` stores metadata, not duplicate checkpoint payloads
