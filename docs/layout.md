# Layout

## Per-Namespace Convention

Each namespace should use the same identifier across `episodes/`, `metadata/`, and `datasets/`.

Example namespace IDs already present in this repository:

- `chaves`
- `dragonball`
- `square-spongebob`
- `beerschool`

## Recommended Tree

```text
episodes/
  <namespace>/
    raw/
    extracted-audio/
    notes/

metadata/
  <namespace>/
    transcripts/
    speakers/
      references/
        <speaker>/
          approved/
          pending/
      runs/
        <run-id>/
          <speaker>/
            extracted-clips/
    manifests/
    qa/

datasets/
  <namespace>/
    clips/
    splits/
    exports/
```

## Folder Purpose

- `episodes/<namespace>/raw/`: original video or audio files as collected
- `episodes/<namespace>/extracted-audio/`: audio extracted from raw media before segmentation
- `episodes/<namespace>/notes/`: source-specific notes such as episode naming issues or missing files
- `metadata/<namespace>/transcripts/`: subtitles, ASR outputs, or cleaned transcripts
- `metadata/<namespace>/speakers/`: speaker-related artifacts
- `metadata/<namespace>/speakers/references/<speaker>/`: reference samples for a single speaker across runs
- `metadata/<namespace>/speakers/references/<speaker>/references.csv`: catalog of reference clips with stable IDs
- `metadata/<namespace>/speakers/references/<speaker>/approved/`: clips that passed basic quality checks
- `metadata/<namespace>/speakers/references/<speaker>/pending/`: clips that were cut but still need review
- `metadata/<namespace>/speakers/runs/`: outputs from speaker-specific or diarization runs
- `metadata/<namespace>/speakers/runs/<run-id>/<speaker>/extracted-audio/`: extracted audio generated for a specific speaker-focused run
- `metadata/<namespace>/speakers/runs/<run-id>/<speaker>/extracted-clips/`: clips already extracted for a speaker during a run
- `metadata/<namespace>/manifests/`: machine-readable indexes for clips and training inputs
- `metadata/<namespace>/qa/`: exclusion lists, quality checks, and review notes
- `datasets/<namespace>/clips/`: final training-ready audio clips
- `datasets/<namespace>/splits/`: train, validation, and test definitions
- `datasets/<namespace>/exports/`: packaged artifacts for downstream training workflows

## Naming Rules

- Use the same `<series>` slug across all top-level areas
- Use the same `<namespace>` slug across all top-level areas
- Keep folder names lowercase and hyphenated
- Prefer descriptive filenames that include episode or source identifiers
- Do not rename raw source files unless there is a documented reason
- Approved reusable references should use `<namespace>-<speaker>-reference-<index>-seed.<ext>`

## Example

```text
episodes/chaves/raw/
episodes/chaves/extracted-audio/
metadata/chaves/transcripts/
metadata/chaves/speakers/
metadata/chaves/speakers/references/
metadata/chaves/manifests/sources.csv
metadata/chaves/manifests/clips.csv
datasets/chaves/clips/
datasets/chaves/splits/
```

## Minimal Manifest Set

For each namespace, start with:

- `metadata/<namespace>/manifests/sources.csv`
- `metadata/<namespace>/manifests/clips.csv`
- `metadata/<namespace>/manifests/rejected.csv`

This is enough to track raw inputs, accepted clips, and discarded segments without overdesigning the schema.

## Speaker References

When a namespace has multiple speakers to extract in separate runs, keep voice references here:

- `metadata/<namespace>/speakers/references/<speaker>/`

Example:

```text
metadata/chaves/speakers/references/chaves/
metadata/chaves/speakers/references/seu-madruga/
metadata/chaves/speakers/references/chiquinha/
```

Use these folders for short reference clips, seed samples, and notes that help identify the same speaker in future runs.

When a speaker has multiple references, assign stable IDs in `references.csv` so jobs can select them explicitly.

## Extracted Speaker Clips

Do not store already-extracted speaker clips in `references/`.

Use:

- `metadata/<namespace>/speakers/runs/<run-id>/<speaker>/extracted-clips/`

This keeps seed references separate from actual run outputs.
