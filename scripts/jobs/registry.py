"""Job registry for the pipeline runner."""

from . import (
    download_google_drive_folder_job,
    download_youtube_video_job,
    export_review_chunks_job,
    export_speaker_samples_job,
    extract_diarized_speaker_job,
    extract_audio_job,
    register_sources_job,
)


JOBS = {
    download_google_drive_folder_job.JOB_NAME: download_google_drive_folder_job,
    download_youtube_video_job.JOB_NAME: download_youtube_video_job,
    export_review_chunks_job.JOB_NAME: export_review_chunks_job,
    export_speaker_samples_job.JOB_NAME: export_speaker_samples_job,
    extract_diarized_speaker_job.JOB_NAME: extract_diarized_speaker_job,
    extract_audio_job.JOB_NAME: extract_audio_job,
    register_sources_job.JOB_NAME: register_sources_job,
}
