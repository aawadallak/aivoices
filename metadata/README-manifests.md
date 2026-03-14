# Manifest Schema

This repository uses a minimal manifest schema to track source files and generated clips.

## Files

- `sources.csv`: inventory of original source media
- `clips.csv`: inventory of accepted training-ready clips
- `rejected.csv`: inventory of rejected or excluded segments
- `references.csv`: inventory of speaker reference clips

## `sources.csv`

Columns:

- `source_id`: stable identifier for the source file
- `namespace`: namespace slug
- `episode_id`: episode or source identifier
- `file_path`: path to the raw source file
- `language`: language code when known
- `status`: source state such as `available`, `extracted`, `missing`, or `corrupt`
- `notes`: short free-text note

## `clips.csv`

Columns:

- `clip_id`: stable identifier for the clip
- `source_id`: source reference from `sources.csv`
- `speaker`: speaker label or best-known identity
- `start_sec`: start timestamp in seconds
- `end_sec`: end timestamp in seconds
- `text`: transcript for the clip
- `audio_path`: path to the generated clip file
- `split`: `train`, `val`, `test`, or blank until assigned

## `rejected.csv`

Columns:

- `segment_id`: identifier for the rejected segment
- `source_id`: source reference from `sources.csv`
- `start_sec`: start timestamp in seconds
- `end_sec`: end timestamp in seconds
- `reason`: why the segment was excluded
- `notes`: optional supporting note

## `references.csv`

Columns:

- `reference_id`: stable identifier for the reference clip
- `speaker`: speaker identifier
- `namespace`: namespace slug
- `status`: `approved` or `pending`
- `kind`: for example `seed`
- `file_path`: path to the reference clip
- `notes`: optional supporting note

## Notes

- Keep IDs stable once they are referenced by scripts or downstream data
- Prefer relative paths rooted at the repository
- Keep the schema minimal until a real workflow requires extra columns
