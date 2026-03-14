# AGENTS.md

## Purpose

This repository is used to prepare and organize voice/audio datasets.

## Current Structure

- `episodes/`: source material grouped by namespace
- `.venv-dataset/`: local Python virtual environment for dataset tooling; not part of the dataset itself

## Folder Organization Rules

- Keep raw media under `episodes/<namespace>/`
- Use one top-level folder per namespace
- A namespace may be a series, a show, a project, or another stable content grouping
- Do not assume every namespace is a sub-series of another namespace
- Use lowercase names with hyphens for folder names
- Do not mix tooling, scripts, and dataset assets in the same folder
- Do not store generated caches inside source media folders unless the workflow requires it

## Preferred Next Step

If the repository grows, split it into explicit top-level areas:

- `episodes/`: raw source audio or video
- `datasets/`: processed outputs ready for training or labeling
- `scripts/`: repeatable preprocessing and utility scripts
- `metadata/`: manifests, speaker maps, transcripts, and annotations
- `docs/`: process notes and dataset decisions

## Namespace Model

- `episodes/<namespace>/` is the canonical root for a content group
- Treat `dragonball`, `chaves`, `square-spongebob`, and `beerschool` as separate namespaces unless documentation says otherwise
- A namespace can contain one or more workspaces under `raw/`
- Use workspace folders to separate imports, arcs, movies, channels, or batches inside the same namespace
- Do not force unrelated content into an existing namespace just because it feels similar

## Naming Conventions

- Prefer stable, slug-like filenames
- Keep source identifiers when they help trace origin
- After downloads or imports, rename files into the repository naming pattern instead of leaving provider-generated names in place
- For downloaded raw media, prefer `<namespace>-<descriptive-slug>-<source-id>.<ext>` when a source identifier exists
- Reference clips should use the pattern `<namespace>-<speaker>-reference-<index>-seed.<ext>` once approved as reusable seed samples
- If a speaker has multiple references, maintain `metadata/<namespace>/speakers/references/<speaker>/references.csv` with stable `reference_id` values
- Avoid vague suffixes such as `alt`, `final`, `new`, or `clean` when a clearer role-based suffix is available

## Script Organization

- `scripts/jobs/`: canonical location for isolated job definitions
- `scripts/pipeline/`: pipeline orchestration and execution entrypoints
- `scripts/preprocess/`: concrete preprocessing implementations that jobs may call

## Code Hygiene

- Do not leave dead code, abandoned scripts, or obsolete directories after reorganizing the project
- When a folder stops being the canonical location for a responsibility, either remove it or document its remaining purpose explicitly
- Avoid duplicate implementations of the same workflow in different script folders
- If a script is kept as a lower-level implementation used by a job, document that relationship clearly
- Prefer one obvious entrypoint per responsibility

## Commit Policy

- Break changes into small, reviewable commits instead of batching unrelated work together
- Always use Conventional Commits for commit messages
- Prefer separating docs, tooling, data-manifest, and workflow changes when they can stand on their own

## Documentation For Future Runs

- Record setup steps, failures, fixes, and important commands in repository docs
- Prefer short operational notes over undocumented trial-and-error
- When a command fails because of environment issues, document both the error and the working fix
- Use exact paths, package names, Python versions, and dependency indexes when they matter
- Include enough context so the same setup can be repeated later without relying on chat history
- Keep `README.md` accurate and current as the main onboarding document for future readers of the repository
- When workflows, paths, commands, or project conventions change, update `README.md` in the same run whenever practical

## Recommended Docs Layout

- `README.md`: high-level onboarding, setup summary, workflow entrypoints, and storage model
- `docs/setup.md`: environment setup, system packages, virtualenv creation, Python version requirements
- `docs/runs.md`: dated run notes, what was attempted, what worked, what failed, and follow-up items
- `docs/decisions.md`: durable decisions about folder structure, naming, formats, and tooling choices
- `docs/storage.md`: raw media sync and flush workflow for Google Drive via `rclone`

## Storage Policy

- do not commit `episodes/<namespace>/raw/` or `episodes/<namespace>/extracted-audio/` to git
- store heavy raw media outside git history and sync it to Google Drive
- prefer `rclone copy` for uploads and `rclone sync --dry-run` for verification before any mirroring step
- avoid adding ad-hoc one-off storage commands when a repeatable script under `scripts/storage/` should exist instead

## Working Notes

- Treat files in `episodes/` as source-of-truth inputs
- Keep generated outputs reproducible from scripts when possible
- Avoid committing virtual environments, caches, and temporary artifacts
- Do not rely on terminal history or chat history as the only record of important project steps
- Keep `scripts/jobs/` and `scripts/pipeline/` aligned with the current architecture so outdated paths do not linger
- When a user names a namespace explicitly, prefer that exact top-level folder over inferred hierarchy
- When renaming files, prefer improving clarity and consistency without losing traceability
