# aivoices

Workspace for collecting, organizing, and preparing audio datasets by namespace.

## What Lives Here

- `episodes/`: raw source media and bulk extracted audio
- `metadata/`: manifests, speaker references, run outputs, transcripts, and QA notes
- `datasets/`: training-ready clips, splits, and exports
- `scripts/`: repeatable jobs, pipeline runners, and storage helpers
- `docs/`: setup notes, run history, decisions, layout, and storage workflow

## Namespace Model

This repository uses `namespace` as the main grouping key.

Examples currently present:

- `dragonball`
- `chaves`
- `square-spongebob`
- `beerschool`

The same namespace should be used across:

- `episodes/<namespace>/`
- `metadata/<namespace>/`
- `datasets/<namespace>/`

## Recommended Layout

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
          references.csv
      runs/
        <run-id>/
          <speaker>/
            extracted-clips/
    manifests/
      sources.csv
      clips.csv
      rejected.csv
    qa/

datasets/
  <namespace>/
    clips/
    splits/
    exports/
```

## Environment Setup

Base system packages:

```bash
sudo apt update
sudo apt install -y ffmpeg python3 python3-venv python3-pip git
```

Create the local virtualenv:

```bash
python3 -m venv /home/awadallak/aivoices/.venv-dataset
source /home/awadallak/aivoices/.venv-dataset/bin/activate
python -m pip install --upgrade "pip<24.1" setuptools wheel
```

## Python Notes

The current `.venv-dataset` is working on `Python 3.11.15`.

Before installing ML dependencies, confirm:

```bash
python --version
pip --version
```

If the interpreter is `Python 3.13`, recreate the environment with `Python 3.11` first.

The current working stack in this repository is:

- `torch 2.8.0+cu128`
- `torchaudio 2.8.0`
- `torchvision 0.23.0`
- `whisperx 3.8.2`
- `pyannote.audio 4.0.4`
- `resemblyzer 0.1.4`

Important notes:

- `resemblyzer` currently needs `setuptools<81`
- GPU access is currently intermittent on this host; some checks see the RTX 3060 Ti correctly and others fail with NVML initialization errors
- `pyannote.audio` imports successfully but warns about `torchcodec`, so prefer using audio already extracted with `ffmpeg`

## Useful Dependencies

Common tools used in this repository:

- `ffmpeg`
- `yt-dlp`
- `gdown`
- `rclone`

Potential Python packages for later pipeline stages:

- `pydub`
- `soundfile`
- `librosa`

## Current Workflow

The project is organized around small jobs and a simple pipeline runner.

Relevant entrypoints:

- `scripts/jobs/`: single-purpose jobs
- `scripts/pipeline/run_pipeline.py`: pipeline runner
- `scripts/preprocess/extract_audio.py`: low-level bulk audio extraction
- `scripts/storage/`: Google Drive sync helpers for raw media
- `scripts/jobs/export_xtts_dataset.py`: XTTS-ready export with `metadata_train.csv`, `metadata_eval.csv`, and `wavs/`
- `scripts/jobs/train_xtts.py`: XTTS training entrypoint using `--dataset-dir`

For a fresh Runpod XTTS setup, start with [`docs/runpod.md`](/home/awadallak/aivoices/docs/runpod.md), which now includes a copy-paste "from zero" checklist for pod bootstrap, dataset fetch, dependency verification, and training start.

## Important Jobs

Extract audio for a namespace:

```bash
python scripts/preprocess/extract_audio.py --namespace square-spongebob
```

Register raw sources into the manifest:

```bash
python scripts/jobs/register_sources.py --namespace dragonball --language pt-BR
```

Run the same through the pipeline:

```bash
python scripts/pipeline/run_pipeline.py --namespace dragonball --jobs register-sources
```

List references for a speaker:

```bash
python scripts/jobs/list_references.py --namespace beerschool --speaker 12
```

Extract one diarized speaker directly after manual review:

```bash
python scripts/jobs/extract_diarized_speaker.py \
  --namespace square-spongebob \
  --episode-id square-spongebob-episode-244 \
  --json-path metadata/square-spongebob/transcripts/whisperx-pilot-244-diarized/square-spongebob-episode-244.json \
  --diarized-speaker SPEAKER_03 \
  --speaker bob-esponja \
  --run-id square-spongebob-direct-244
```

For difficult content, prefer this controlled flow over cross-episode speaker matching:

- diarize one episode
- export per-speaker samples
- map `SPEAKER_XX` manually
- extract that diarized speaker directly
- promote only the clean clips into references

## Source Manifests

Each namespace should maintain:

- `metadata/<namespace>/manifests/sources.csv`
- `metadata/<namespace>/manifests/clips.csv`
- `metadata/<namespace>/manifests/rejected.csv`

`sources.csv` now includes:

- `source_id`
- `namespace`
- `episode_id`
- `file_path`
- `language`
- `status`
- `notes`

Typical `status` values:

- `available`
- `extracted`
- `missing`
- `corrupt`

## Speaker References

Reusable speaker seeds live under:

```text
metadata/<namespace>/speakers/references/<speaker>/
```

Approved reference naming:

```text
<namespace>-<speaker>-reference-<index>-seed.<ext>
```

When a speaker has multiple references, catalog them in:

```text
metadata/<namespace>/speakers/references/<speaker>/references.csv
```

## Raw Media Storage

Heavy raw media should not go to git.

Ignored by git:

- `episodes/*/raw/`
- `episodes/*/extracted-audio/`

Use Google Drive via `rclone` instead.

Beyond raw media, also sync the durable artifacts that are expensive to recreate:

- `metadata/<namespace>/speakers/references/`
- `metadata/<namespace>/qa/`
- `metadata/<namespace>/manifests/`
- `metadata/<namespace>/transcripts/`
- `datasets/<namespace>/exports/`

Do not upload archive duplicates such as `.zip`, `.rar`, or `.7z` when the unpacked folders are already being synced.

Examples:

```bash
bash scripts/storage/flush_raw.sh gdrive-aivoices aivoices/episodes
bash scripts/storage/check_raw.sh gdrive-aivoices aivoices/episodes dragonball
bash scripts/storage/pull_raw.sh gdrive-aivoices aivoices/episodes square-spongebob
```

For setup and policy details, see [`docs/storage.md`](/home/awadallak/aivoices/docs/storage.md).

## Version Control

Commit:

- scripts, jobs, and repeatable pipeline code
- docs and operational notes
- small manifests or QA CSVs that record durable decisions
- approved speaker reference metadata such as `references.csv`

Do not commit:

- `episodes/*/raw/`
- `episodes/*/extracted-audio/`
- `metadata/*/speakers/runs/`
- bulk WhisperX transcript outputs under `metadata/*/transcripts/whisperx-*/`
- generated dataset outputs under `datasets/*/clips/`, `datasets/*/splits/`, and `datasets/*/exports/`
- local training outputs, checkpoints, TensorBoard logs, or experiment caches

The default rule is simple: if an artifact is large, reproducible, or tied to one run, keep it out of git and document the workflow instead.

## XTTS Export

This repository can now export speaker-specific XTTS training packages under:

```text
datasets/<namespace>/exports/export_xtts/<export-name>/
```

Each package contains:

- `metadata_train.csv`
- `metadata_eval.csv`
- `wavs/`

The CSV format is:

```text
audio_file|text|speaker_name
```

Example:

```bash
python scripts/jobs/export_xtts_dataset.py \
  --namespace square-spongebob \
  --source-export provisional-quality-clean-v1 \
  --speaker bob-esponja \
  --output-name bob-esponja-clean-v1
```

For the repeatable XTTS export and R2 upload workflow, see [`docs/tts.md`](/home/awadallak/aivoices/docs/tts.md).

## XTTS Training

The repository now includes the first training-side building blocks for XTTS:

- fetch portable XTTS packages from R2
- train from `--dataset-dir`
- generate smoke-test review samples
- promote `best` and `last` artifacts

See [`docs/training.md`](/home/awadallak/aivoices/docs/training.md).

## Additional Docs

- [`docs/setup.md`](/home/awadallak/aivoices/docs/setup.md)
- [`docs/layout.md`](/home/awadallak/aivoices/docs/layout.md)
- [`docs/decisions.md`](/home/awadallak/aivoices/docs/decisions.md)
- [`docs/runs.md`](/home/awadallak/aivoices/docs/runs.md)
- [`docs/storage.md`](/home/awadallak/aivoices/docs/storage.md)
- [`docs/tts.md`](/home/awadallak/aivoices/docs/tts.md)
- [`docs/training.md`](/home/awadallak/aivoices/docs/training.md)
- [`docs/runpod.md`](/home/awadallak/aivoices/docs/runpod.md)
- [`docs/tasks/README.md`](/home/awadallak/aivoices/docs/tasks/README.md)
- [`docs/tasks/xtts-training-pipeline-design.md`](/home/awadallak/aivoices/docs/tasks/xtts-training-pipeline-design.md)
