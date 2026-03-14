# Pipeline Scripts

This directory contains a simple job-based pipeline runner.

## Model

- one pipeline execution = one terminal command
- each job lives in its own file under `scripts/jobs/`
- `run_pipeline.py` executes the selected jobs in order

## Available Jobs

- `download-google-drive-folder`
- `download-youtube-video`
- `extract-audio`
- `export-review-chunks`
- `export-speaker-samples`
- `extract-diarized-speaker`
- `register-sources`

## Examples

Run one job for a whole series:

```bash
python scripts/pipeline/run_pipeline.py --series chaves --jobs extract-audio
```

Download one YouTube video into a series workspace:

```bash
python scripts/pipeline/run_pipeline.py --series dragonball --jobs download-youtube-video --url 'https://www.youtube.com/watch?v=example' --workspace batalha-dos-deuses
```

Download one Google Drive folder into a namespace workspace:

```bash
python scripts/pipeline/run_pipeline.py --series beerschool --jobs download-google-drive-folder --url 'https://drive.google.com/drive/folders/...' --workspace drive-import
```

Run one job for a specific file:

```bash
python scripts/pipeline/run_pipeline.py --input episodes/chaves/raw/example.mp4 --jobs extract-audio
```

Run a speaker-specific execution:

```bash
python scripts/pipeline/run_pipeline.py --series chaves --jobs extract-audio --speaker seu-madruga --run-id 2026-03-13-pass-01
```

Extract one diarized speaker directly from a WhisperX JSON result:

```bash
python scripts/pipeline/run_pipeline.py \
  --namespace square-spongebob \
  --jobs extract-diarized-speaker \
  --episode-id square-spongebob-episode-244 \
  --json-path metadata/square-spongebob/transcripts/whisperx-pilot-244-diarized/square-spongebob-episode-244.json \
  --diarized-speaker SPEAKER_03 \
  --speaker bob-esponja \
  --run-id square-spongebob-direct-244
```

Preview commands without executing:

```bash
python scripts/pipeline/run_pipeline.py --series dragonball --jobs download-youtube-video --url 'https://www.youtube.com/watch?v=example' --workspace batalha-dos-deuses --dry-run
```

## Adding Jobs

To add a new job:

1. Create a new file in `scripts/jobs/`
2. Export a `JOB_NAME`
3. Export a `build_command(args)` function
4. Register the job in `scripts/jobs/registry.py`
