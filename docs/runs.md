# Runs

## 2026-03-13

### Objective

Prepare a local dataset environment with system packages, a virtual environment, PyTorch, and audio-processing dependencies.

### Commands Attempted

The session attempted:

- `sudo apt update`
- `sudo apt install -y ffmpeg python3 python3-venv python3-pip git`
- `python3 -m venv .venv-dataset`
- `source .venv-dataset/bin/activate`
- `python -m pip install --upgrade "pip<24.1" setuptools wheel`
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- `pip install demucs whisperx pyannote.audio pydub soundfile librosa resemblyzer`

### Issues Observed

- `sudo` required an interactive password prompt, so the setup could not be completed autonomously
- `apt update` failed because of third-party repository problems
- a pasted multi-line shell command was split incorrectly by `zsh`, causing commands such as `--upgrade` to be interpreted on their own
- `torch` was not installed successfully, so the CUDA validation command failed with `ModuleNotFoundError`

### APT Problems

Observed repository issues:

- `warpdotdev`: GPG key expired for `https://releases.warp.dev/linux/deb`
- `hashicorp`: invalid `zara` release entry and no `Release` file

### Fixes Applied

- `hashicorp` was already disabled in `/etc/apt/sources.list.d/hashicorp.list.disabled`
- `warpdotdev` was identified as the remaining active broken repository and should be disabled before re-running `apt update`
- subsequent instructions were rewritten as one command per line to avoid shell paste errors

### Current Status

- `/home/awadallak/aivoices/.venv-dataset` exists
- the virtual environment can be activated
- PyTorch is not installed yet
- CUDA availability is not verified yet
- the repository now has explicit top-level folders for `datasets/`, `metadata/`, and `scripts`
- each current namespace now has minimal manifest templates for sources, accepted clips, and rejected segments
- the first preprocessing script now exists at `scripts/preprocess/extract_audio.py`
- each namespace now has a dedicated area for reusable speaker references and per-run speaker outputs
- `extract_audio.py` now supports `--speaker` and `--run-id` to route run-specific outputs into `metadata/<namespace>/speakers/runs/`
- a simple pipeline runner now exists at `scripts/pipeline/run_pipeline.py` with isolated job files under `scripts/jobs/`
- speaker references and already-extracted speaker clips are now treated as different categories in the repository layout
- jobs now exist to create and validate reference clips with automatic placement into `approved/` or `pending/`
- a job now exists to download YouTube videos directly into `episodes/<namespace>/raw/<workspace>/`
- the repository terminology is being standardized around `namespace` rather than assuming every content root is a series
- approved reusable references now follow the pattern `<namespace>-<speaker>-reference-<index>-seed.<ext>`
- jobs now exist to register and list references via `references.csv`

### Next Step

1. Confirm Python version with `python --version`
2. If needed, recreate the virtual environment with Python 3.11
3. Retry the PyTorch install
4. Only then install `demucs`, `whisperx`, `pyannote.audio`, `pydub`, `soundfile`, `librosa`, and `resemblyzer`

## 2026-03-14

### Media Processing

- bulk audio extraction was run across current namespaces with raw media under `episodes/*/raw/`
- `square-spongebob` extraction completed successfully for all currently downloaded source files
- `dragonball` extraction completed for episodes `001`, `002`, `004`, `005`, `006`, `007`, `009`, `010`, `011`, `012`, `013`, and `014`
- `dragonball` episode `001` is explicitly marked as already extracted in `metadata/dragonball/manifests/sources.csv` and should not be rerun for now
- `dragonball` episodes `003` and `008` were confirmed corrupt, failed extraction with ffmpeg input parsing errors, and were removed from `episodes/dragonball/raw/batalha-dos-deuses/`

### Batch Extraction Fix

- `scripts/preprocess/extract_audio.py` now uses `ffmpeg -nostdin` to avoid ffmpeg consuming pipeline stdin during batch runs

### Source Registration

- a new job now exists at `scripts/jobs/register_sources.py`
- the pipeline runner now accepts `--namespace` directly and can run `register-sources`
- `metadata/beerschool/manifests/sources.csv` was generated with 5 extracted sources
- `metadata/dragonball/manifests/sources.csv` was generated with 12 extracted sources currently available in raw media
- `metadata/square-spongebob/manifests/sources.csv` was generated with 33 extracted sources
- `sources.csv` now carries a `status` column so sources can be tracked as `available`, `extracted`, `missing`, or `corrupt`

### Storage Workflow

- `.gitignore` now excludes `episodes/*/raw/` and `episodes/*/extracted-audio/` from git
- storage workflow is documented in `docs/storage.md`
- helper scripts now exist in `scripts/storage/` for:
  - flushing raw media to Google Drive
  - pulling raw media back into the workspace
  - checking sync differences with `rclone sync --dry-run`
- the chosen model is git for code/docs/manifests and Google Drive via `rclone` for heavy raw media

### Square-Spongebob Pilot

- true diarization dependencies are still missing from `.venv-dataset`, so a manual-review pilot was used instead
- a new job now exists at `scripts/jobs/export_review_chunks.py`
- `square-spongebob-episode-244` was selected as the first pilot episode
- the first 5 minutes were exported into 20-second review chunks under `metadata/square-spongebob/speakers/runs/square-spongebob-pilot-244/review/square-spongebob-episode-244/chunks/`
- this pilot is intended to help identify target speakers and cut the first reusable reference seeds
- a full `whisperx` test was then run on `square-spongebob-episode-244` with alignment and diarization enabled
- the resulting JSON is stored at `metadata/square-spongebob/transcripts/whisperx-pilot-244-diarized/square-spongebob-episode-244.json`
- the pilot detected 7 speakers labeled `SPEAKER_00` through `SPEAKER_06`
- a new job now exists at `scripts/jobs/export_speaker_samples.py`
- 40 short review clips were exported across the 7 detected speakers under `metadata/square-spongebob/speakers/runs/square-spongebob-pilot-244/review/square-spongebob-episode-244/speaker-samples/`
- a review table was written to `metadata/square-spongebob/speakers/runs/square-spongebob-pilot-244/review/square-spongebob-episode-244/speaker-samples/speaker-samples.csv`
- a provisional `speaker-map.csv` was created for the pilot episode
- manual mapping from the pilot currently identifies:
  - `SPEAKER_02` as `lula-molusco`
  - `SPEAKER_03` as `bob-esponja`
  - `SPEAKER_06` as `patrick`
- provisional seed references were promoted for those three speakers and registered in their respective `references.csv` catalogs
- a new matching script now exists at `scripts/jobs/match_whisperx_speakers.py`
- a new extraction script now exists at `scripts/jobs/extract_matched_segments.py`
- `square-spongebob-episode-245` was processed with WhisperX diarization and matched against the pilot 244 embeddings
- the resulting match file is `metadata/square-spongebob/speakers/runs/square-spongebob-match-245/speaker-matches.csv`
- 89 matched clips were exported for episode 245 under `metadata/square-spongebob/speakers/runs/square-spongebob-match-245/matched-clips/square-spongebob-episode-245/`
- first-pass distribution for episode 245 was:
  - `lula-molusco`: 68 clips
  - `bob-esponja`: 19 clips
  - `patrick`: 2 clips
- this first pass should still be treated as provisional because the pilot references are not clean yet and the speaker distribution is likely over-matching `lula-molusco`
- the user reviewed the first-pass outputs and promoted better seeds from episode 245:
  - `bob-esponja-clip-0008.wav` became a new Bob reference
  - `lula-molusco-clip-0019.wav` and `lula-molusco-clip-0020.wav` became new Lula references
- `patrick` was explicitly removed from the next matching pass because its initial seed quality was too poor
- `match_whisperx_speakers.py` now supports `--allowed-speakers` so a rerun can exclude unstable speakers
- episode 245 was rerun without `patrick`
- the no-Patrick match file is `metadata/square-spongebob/speakers/runs/square-spongebob-match-245-no-patrick/speaker-matches.csv`
- 87 matched clips were exported for the no-Patrick rerun under `metadata/square-spongebob/speakers/runs/square-spongebob-match-245-no-patrick/matched-clips/square-spongebob-episode-245/`
- no-Patrick distribution for episode 245 was:
  - `lula-molusco`: 68 clips
  - `bob-esponja`: 19 clips
- two more false positives from the Bob bucket were later confirmed as mostly `lula-molusco`:
  - `bob-esponja-clip-0005.wav`
  - `bob-esponja-clip-0013.wav`
- those two clips were promoted into new Lula references:
  - `square-spongebob-lula-molusco-ref-004`
  - `square-spongebob-lula-molusco-ref-005`
- a second matching experiment then switched from diarized-speaker-level matching to per-segment reference matching
- that experiment wrote `metadata/square-spongebob/speakers/runs/square-spongebob-match-245-refclips-v2/speaker-matches.csv`
- it extracted 107 clips:
  - `lula-molusco`: 72
  - `bob-esponja`: 35
- the user judged this second experiment worse than the simpler no-Patrick pass
- because of that, the preferred direction changed from cross-episode auto-matching toward a more controlled per-episode workflow
- a new job now exists at `scripts/jobs/extract_diarized_speaker.py`
- the new preferred flow for difficult content is:
  - run WhisperX diarization on one episode
  - export speaker samples for manual review
  - identify the diarized speaker label manually
  - extract clips directly from that diarized speaker without cross-episode matching
  - only then promote clean clips into approved references
- this direct diarized-speaker flow was validated on `square-spongebob-episode-244`
- `SPEAKER_03` was extracted directly as `bob-esponja` into:
  - `metadata/square-spongebob/speakers/runs/square-spongebob-direct-244/diarized-clips/square-spongebob-episode-244/bob-esponja/`
- that direct extraction produced 12 Bob clips with a manifest at:
  - `metadata/square-spongebob/speakers/runs/square-spongebob-direct-244/diarized-clips/square-spongebob-episode-244/bob-esponja/diarized-clips.csv`

### ML Environment Repair

- `.venv-dataset` was confirmed on `Python 3.11.15`
- the environment now has working imports for `whisperx`, `pyannote.audio`, `resemblyzer`, `librosa`, and `soundfile`
- installing `whisperx` upgraded the torch stack from `2.5.1+cu121` to `2.8.0+cu128`
- `torchvision` had to be upgraded to `0.23.0` to match `torch 2.8.0`
- `resemblyzer` required `setuptools<81` because it still imports `pkg_resources`
- GPU access in this host is currently intermittent:
  - one validation pass detected `NVIDIA GeForce RTX 3060 Ti` and returned `torch.cuda.is_available() == True`
  - later validation passes failed again with `nvidia-smi` reporting `Failed to initialize NVML: Unknown Error`
  - `.venv-dataset` currently reports `torch.cuda.is_available() == False`
- `pyannote.audio` still warns about `torchcodec`, so built-in decoding should not be trusted in this environment
- the safe current practice is to continue using `.wav` files already extracted with `ffmpeg`

### Square-Spongebob Batch Diarization And Experimental QA

- all 33 currently extracted `square-spongebob` episodes were successfully diarized
- new batch diarization outputs were written under `metadata/square-spongebob/transcripts/whisperx-batch-diarized/`
- conservative reference matching was run for the 30 batch-diarized episodes under `metadata/square-spongebob/speakers/runs/reference-match-square-spongebob-episode-*/`
- an experimental clip-quality workflow was added with:
  - stricter VAD checks
  - RMS/loudness and estimated SNR scoring
  - a lightweight music heuristic
  - overlap and speaker-turn proximity checks from the WhisperX diarization JSON
- the full provisional export using `clean,usable` labels was written to:
  - `datasets/square-spongebob/exports/provisional-quality-v1/`
- a stricter `clean`-only export was written to:
  - `datasets/square-spongebob/exports/provisional-quality-clean-v1/`
- the current experimental shortlist of clips that passed the `keep_for_reference=yes` gate was written to:
  - `metadata/square-spongebob/qa/reference-candidates-experimental-v1.csv`
- the current experimental shortlist is intentionally very small:
  - 2 candidate clips
  - both for `lula-molusco`
  - from episodes `399` and `402`
- because the current `keep_for_reference` gate is very strict, treat this file as a first-pass QA artifact rather than a final speaker-reference decision
