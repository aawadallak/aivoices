# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**aivoices** is an end-to-end audio dataset preparation and TTS training workspace. The workflow goes: raw media collection → audio extraction → ASR/diarization → speaker matching → clip curation → XTTS export → cloud training → smoke review → promotion.

Content is organized by **namespace** (e.g., `square-spongebob`, `dragonball`, `chaves`). The same namespace slug is used across `episodes/`, `metadata/`, `datasets/`, and `training/`.

## Running Jobs

There is no formal build system, test suite, or linter. Jobs are run directly via Python.

**Via pipeline runner:**
```bash
python scripts/pipeline/run_pipeline.py --namespace <ns> --jobs <job-name> [extra args]
```

**Directly (most jobs):**
```bash
python scripts/jobs/<job_name>.py --namespace <ns> [args]
```

**Key jobs:**
- `register_sources.py` — scan raw media into manifests
- `batch_whisperx_diarize.py` — ASR + speaker diarization
- `extract_diarized_speaker.py` — extract clips for identified speakers
- `export_xtts_dataset.py` — convert curated clips to XTTS format
- `train_xtts.py` — run XTTS training
- `export_xtts_smoke_review.py` — generate review samples from checkpoints
- `promote_xtts_run.py` — promote best/last training artifacts
- `fetch_xtts_dataset.py` / `publish_xtts_dataset.py` — R2 dataset transport

**XTTS training (on Runpod):**
```bash
bash scripts/bootstrap/runpod_xtts_bootstrap.sh   # full VM setup
python scripts/jobs/train_xtts.py --namespace <ns> --dataset-dir /data/<dataset> --voice <voice>
```

## Architecture

### Three-Layer Design

1. **Ingestion & Metadata** — Raw media in `episodes/<ns>/`, transcripts and speaker data in `metadata/<ns>/`
2. **Curation** — Diarization → speaker matching → scoring → provisional exports under `datasets/<ns>/exports/`
3. **Training** — XTTS export (CSV + wavs/) → R2 publish → Runpod training → smoke review → promotion

### Job System

- `scripts/jobs/` — Self-contained single-purpose jobs. Each exports `JOB_NAME` and `build_command(args)`.
- `scripts/pipeline/run_pipeline.py` — Orchestrates job sequences.
- `scripts/pipeline/xtts_training.py` / `xtts_common.py` — XTTS training engine and shared utilities.
- `scripts/preprocess/extract_audio.py` — Standalone ffmpeg-based audio extraction.
- `scripts/bootstrap/runpod_xtts_bootstrap.sh` — Complete Runpod VM setup (micromamba, Python 3.11.9, venv, deps).
- Only a subset of jobs are registered in `scripts/jobs/registry.py` for pipeline use; most are run directly.

### XTTS Dataset Contract

Training input is a portable directory with: `metadata_train.csv`, `metadata_eval.csv`, `wavs/`, optional `dataset-info.json`. CSV format is pipe-delimited: `audio_file|text|speaker_name`.

### Training Output Layout

```
training/xtts/<namespace>/<voice>/<run-id>/
  run.json
  trainer/
  artifacts/best/   (after promotion)
  artifacts/last/   (after promotion)
```

### Storage Model

- **Git-tracked:** scripts, docs, manifests, QA notes, speaker references, XTTS export metadata
- **Git-excluded:** `episodes/*/raw/`, `episodes/*/extracted-audio/`, training outputs, generated datasets, caches
- **Google Drive (rclone):** heavy raw media sync via `scripts/storage/`
- **Cloudflare R2:** XTTS dataset packages for training VM distribution

## Conventions

- **Commits:** Always use Conventional Commits (`feat`, `fix`, `docs`, etc.). Break changes into small, reviewable commits. Separate docs, tooling, data-manifest, and workflow changes.
- **Naming:** Lowercase hyphenated folder names. Same namespace slug across all top-level directories. Reference clips: `<ns>-<speaker>-reference-<index>-seed.<ext>`.
- **Code hygiene:** No dead code or abandoned scripts. One obvious entrypoint per responsibility. Keep `scripts/jobs/` and `scripts/pipeline/` aligned with current architecture.
- **Documentation:** Record setup steps, failures, and fixes in `docs/`. Update README.md when workflows change.

## Python Environment

**Local development (.venv-dataset):** Python 3.11.15, torch 2.8.0+cu128, whisperx 3.8.2, pyannote.audio 4.0.4

**Training (Runpod):** Python 3.11.9 (via micromamba), coqui-tts 0.25.3, torch 2.8.0, torchaudio 2.8.0. Dependencies pinned in `requirements-xtts-train.txt` — keep torch/torchaudio/Coqui pinned together to avoid pip backtracking.

## Key Documentation

- `AGENTS.md` — Folder organization rules, naming conventions, commit policy
- `docs/setup.md` — Environment setup details
- `docs/decisions.md` — Durable architectural decisions
- `docs/training.md` — XTTS training execution model, dataset contract, smoke review, promotion
- `docs/runpod.md` — Runpod VM bootstrap guide
- `docs/storage.md` — Google Drive rclone workflow
- `docs/tts.md` — XTTS export format and R2 upload
- `docs/runs.md` — Execution history and lessons learned
