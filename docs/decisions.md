# Decisions

## Repository Organization

- `episodes/` is the source-of-truth area for raw media grouped by namespace or content source
- `.venv-dataset/` is local tooling state and is not part of the dataset
- documentation must live in-repo so future runs do not depend on terminal history or chat history

## Documentation Policy

- Important setup steps must be written down in `docs/`
- Environment failures must be documented with both the error and the working fix
- Durable project decisions belong in `docs/decisions.md`
- Time-bound execution notes belong in `docs/runs.md`
- Repeatable environment instructions belong in `docs/setup.md`

## Naming Guidance

- Use lowercase names with hyphens for folders
- Use one folder per namespace under `episodes/`
- Use the same namespace slug across `episodes/`, `metadata/`, and `datasets/`
- Prefer a stable per-namespace layout instead of inventing a new folder scheme for each source
- Approved reusable references should use `<namespace>-<speaker>-reference-<index>-seed.<ext>`

## Planned Top-Level Expansion

If the repository grows beyond raw source storage, prefer this split:

- `episodes/` for raw source media
- `datasets/` for processed dataset outputs
- `scripts/` for repeatable preprocessing utilities
- `metadata/` for manifests, speaker maps, transcripts, and annotations
- `docs/` for setup, runs, and project decisions

## Initial Working Layout

The repository now includes:

- `episodes/` for raw source inputs
- `datasets/` for processed clips, split manifests, and export-ready artifacts
- `metadata/` for transcripts, speaker maps, manifests, and QA notes
- `scripts/` for repeatable preprocessing and export automation
- `docs/` for durable project documentation

## Per-Namespace Layout

The default convention for each namespace is:

- `episodes/<namespace>/raw/` for original media
- `episodes/<namespace>/extracted-audio/` for audio extracted from the originals
- `episodes/<namespace>/notes/` for source-specific notes
- `metadata/<namespace>/transcripts/` for text and ASR outputs
- `metadata/<namespace>/speakers/` for diarization, speaker mapping, and reference management
- `metadata/<namespace>/speakers/references/<speaker>/` for reusable voice references per speaker
- `metadata/<namespace>/speakers/runs/` for outputs of diarization or speaker-specific extraction runs
- `metadata/<namespace>/manifests/` for machine-readable indexes
- `metadata/<namespace>/qa/` for review notes and exclusions
- `datasets/<namespace>/clips/` for final clips
- `datasets/<namespace>/splits/` for train, validation, and test splits
- `datasets/<namespace>/exports/` for downstream packaged outputs

## Minimal Manifest Policy

Each namespace should start with a small manifest set:

- `sources.csv` for raw source inventory
- `clips.csv` for accepted dataset clips
- `rejected.csv` for excluded segments

The `sources.csv` manifest should carry an explicit `status` column so runs can track whether a source is only available, already extracted, missing, or known-corrupt.

Add new manifest files only when the existing three no longer cover the workflow clearly.

## Speaker Reference Policy

If a namespace contains multiple target speakers, store reusable voice references under:

- `metadata/<namespace>/speakers/references/<speaker>/`

This keeps speaker-identification material separate from final dataset clips and makes repeated extraction runs easier to reproduce.

Reference clips should be classified into:

- `approved/` when they pass basic technical checks
- `pending/` when they still need manual review

When a speaker has more than one reference clip, catalog them in:

- `metadata/<namespace>/speakers/references/<speaker>/references.csv`

Each reference must have a stable `reference_id` so pipelines and manual runs can choose specific references explicitly.

Reference catalogs should be maintained through scripts when possible rather than hand-editing CSV rows.

## Extracted Speaker Output Policy

- `references/` is only for reusable seed samples used to identify a speaker
- speaker audio already extracted from a source belongs under `metadata/<namespace>/speakers/runs/<run-id>/<speaker>/extracted-clips/`
- only validated final assets should move from run outputs into `datasets/<namespace>/clips/`

## Pipeline Model

- A pipeline is a single execution triggered from the terminal
- A pipeline is composed of one or more named jobs
- Each job should live in its own file under `scripts/jobs/`
- Jobs should remain small and composable rather than becoming one large all-in-one script
- Shared parameters such as `namespace`, `speaker`, and `run-id` can be passed through the pipeline runner
- Remote source acquisition should also be implemented as jobs, not ad-hoc terminal commands

## Raw Media Storage

- raw source media and extracted bulk audio should stay out of git history
- `episodes/<namespace>/raw/` and `episodes/<namespace>/extracted-audio/` should be synced to Google Drive instead
- the default sync posture should be conservative: `rclone copy` for uploads and `rclone sync --dry-run` for verification
- avoid `rclone bisync` until there is a real two-way sync requirement and a tested recovery workflow
- repeatable storage operations should live under `scripts/storage/`
