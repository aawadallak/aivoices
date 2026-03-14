# Storage

This repository keeps code, manifests, and documentation in git, while large media and selected durable artifacts are synced through object storage or Drive instead of git.

## Policy

- keep `episodes/<namespace>/raw/` out of git
- keep `episodes/<namespace>/extracted-audio/` out of git
- keep scripts, manifests, references, and docs in git when they are small and durable
- use remote storage for heavy raw media and expensive-to-reproduce artifacts
- use `rclone copy` for uploads by default
- use `rclone sync --dry-run` to inspect differences before any destructive sync
- avoid `rclone bisync` until the workflow genuinely requires two-way sync
- do not upload `.zip`, `.rar`, `.7z`, or other archive duplicates when the unpacked folder is already being synced

## What To Sync

Always sync:

- `episodes/<namespace>/raw/`
- `episodes/<namespace>/extracted-audio/`
- `metadata/<namespace>/speakers/references/`
- `metadata/<namespace>/qa/`
- `metadata/<namespace>/manifests/`

Sync when present and useful:

- `metadata/<namespace>/transcripts/`
- `datasets/<namespace>/exports/`

Usually do not sync by default:

- `metadata/<namespace>/speakers/runs/`
- `datasets/<namespace>/clips/`
- `datasets/<namespace>/splits/`
- local training outputs, checkpoints, logs, or caches

The intent is to preserve:

- human review effort
- approved references
- QA decisions
- manifests
- expensive ASR and diarization outputs
- ready-to-consume downstream exports

without turning remote storage into a dump of temporary run artifacts.

## Recommended Remote Layout

Use a dedicated `rclone` remote, for example `gdrive-aivoices`, and store synced data under:

```text
gdrive-aivoices:aivoices/
```

Recommended layout:

```text
episodes/<namespace>/
  raw/
  extracted-audio/

metadata/<namespace>/
  speakers/references/
  qa/
  manifests/
  transcripts/

datasets/<namespace>/
  exports/
```

## One-Time Setup

Install and configure `rclone` locally:

```bash
rclone config
```

Create a Google Drive remote such as `gdrive-aivoices`.

## Scripts

The repository includes helper scripts under `scripts/storage/` for raw media:

- `flush_raw.sh`: upload local raw media and extracted audio to Drive
- `pull_raw.sh`: restore raw media and extracted audio from Drive
- `check_raw.sh`: run `rclone sync --dry-run` to preview differences

Additional sync commands for `metadata/` and `datasets/exports/` can be run directly with `rclone copy` until dedicated scripts are added.

## Examples

Flush everything:

```bash
bash scripts/storage/flush_raw.sh gdrive-aivoices aivoices/episodes
```

Flush selected namespaces:

```bash
bash scripts/storage/flush_raw.sh gdrive-aivoices aivoices/episodes dragonball beerschool
```

Preview changes before sync:

```bash
bash scripts/storage/check_raw.sh gdrive-aivoices aivoices/episodes dragonball
```

Pull selected namespaces back to the workspace:

```bash
bash scripts/storage/pull_raw.sh gdrive-aivoices aivoices/episodes dragonball square-spongebob
```

Sync durable metadata for one namespace:

```bash
rclone copy metadata/square-spongebob/speakers/references \
  gdrive-aivoices:aivoices/metadata/square-spongebob/speakers/references -P

rclone copy metadata/square-spongebob/qa \
  gdrive-aivoices:aivoices/metadata/square-spongebob/qa -P

rclone copy metadata/square-spongebob/manifests \
  gdrive-aivoices:aivoices/metadata/square-spongebob/manifests -P

rclone copy metadata/square-spongebob/transcripts \
  gdrive-aivoices:aivoices/metadata/square-spongebob/transcripts -P
```

Sync downstream exports for one namespace:

```bash
rclone copy datasets/square-spongebob/exports \
  gdrive-aivoices:aivoices/datasets/square-spongebob/exports -P
```

Exclude archive duplicates explicitly when needed:

```bash
rclone copy datasets/square-spongebob/exports \
  gdrive-aivoices:aivoices/datasets/square-spongebob/exports \
  --exclude '*.zip' \
  --exclude '*.rar' \
  --exclude '*.7z' \
  -P
```

## Notes

- `flush_raw.sh` uses `rclone copy`, so it does not delete from Drive
- `check_raw.sh` uses `rclone sync --dry-run` to show what a mirror operation would change
- `pull_raw.sh` is intentionally namespace-scoped to avoid refilling the whole workspace by accident
- if you later want true mirroring, run `rclone sync` manually only after reviewing a dry-run
- prefer syncing extracted folders directly instead of creating archives first
- if a folder is already synced remotely, do not upload a zipped copy of the same payload
