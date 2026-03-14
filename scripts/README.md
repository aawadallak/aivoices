# Scripts

Automation and repeatable preprocessing tools live here.

Suggested structure:

- `scripts/bootstrap/`: environment and setup helpers
- `scripts/preprocess/`: extraction, conversion, normalization, and segmentation
- `scripts/labeling/`: transcription, diarization, and speaker-mapping helpers
- `scripts/export/`: manifest creation and final dataset packaging

Prefer small, composable scripts over manual one-off terminal workflows.

Current scripts:

- `scripts/jobs/`: isolated job files used by the pipeline runner
- `scripts/jobs/download_google_drive_folder.py`: downloads a Google Drive folder into a namespace workspace
- `scripts/jobs/download_youtube_video.py`: downloads a YouTube video into a namespace workspace with yt-dlp
- `scripts/jobs/export_review_chunks.py`: exports fixed-length review chunks to help manual speaker discovery
- `scripts/jobs/export_speaker_samples.py`: exports short per-speaker review clips from WhisperX diarization output
- `scripts/jobs/batch_whisperx_diarize.py`: runs WhisperX diarization in batch for all extracted audio in a namespace
- `scripts/jobs/extract_diarized_speaker.py`: extracts clips for a single diarized speaker from a WhisperX JSON file
- `scripts/jobs/match_reference_segments.py`: conservatively matches diarized segments against approved references
- `scripts/jobs/batch_match_reference_segments.py`: runs the conservative matcher across all diarized JSON files in a namespace
- `scripts/jobs/score_matched_clips.py`: scores extracted matched clips with VAD, loudness, music, and overlap heuristics
- `scripts/jobs/batch_score_matched_clips.py`: runs clip scoring across multiple matched-clip runs
- `scripts/jobs/export_provisional_dataset.py`: copies scored clips into a provisional dataset export
- `scripts/jobs/export_xtts_dataset.py`: writes `metadata_train.csv`, `metadata_eval.csv`, and `wavs/` in XTTS-compatible format
- `scripts/jobs/publish_xtts_dataset.py`: publishes one XTTS package to remote storage with `rclone`
- `scripts/jobs/fetch_xtts_dataset.py`: fetches one XTTS package from remote storage with `rclone`
- `scripts/jobs/train_xtts.py`: runs the repository-native XTTS training flow from `--dataset-dir`
- `scripts/jobs/export_xtts_smoke_review.py`: generates smoke-test review samples and `candidates.json`
- `scripts/jobs/promote_xtts_run.py`: promotes `best` and `last` artifacts and updates `current.json`
- `scripts/jobs/enhance_candidate_clip.py`: applies lightweight enhancement or optional Demucs separation to one clip
- `scripts/jobs/register_sources.py`: scans raw media for a namespace and updates `metadata/<namespace>/manifests/sources.csv`
- `scripts/jobs/register_reference.py`: adds a reference entry to `references.csv`
- `scripts/jobs/list_references.py`: lists available references for a speaker
- `scripts/jobs/make_reference_clip.py`: creates and auto-validates reference clips
- `scripts/jobs/validate_reference_clip.py`: validates an existing reference clip
- `scripts/preprocess/extract_audio.py`: extracts audio from raw episode files
- `scripts/pipeline/run_pipeline.py`: executes selected jobs in sequence
