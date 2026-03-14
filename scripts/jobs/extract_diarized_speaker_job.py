"""Pipeline job for extracting one diarized speaker from a WhisperX JSON file."""

from __future__ import annotations

from pathlib import Path


JOB_NAME = "extract-diarized-speaker"
SCRIPT_PATH = Path(__file__).resolve().parent / "extract_diarized_speaker.py"


def build_command(args) -> list[str]:
    namespace = getattr(args, "namespace", None) or getattr(args, "series", None)
    if not namespace:
        raise SystemExit("The job 'extract-diarized-speaker' requires --namespace.")
    if not getattr(args, "episode_id", None):
        raise SystemExit("The job 'extract-diarized-speaker' requires --episode-id.")
    if not getattr(args, "json_path", None):
        raise SystemExit("The job 'extract-diarized-speaker' requires --json-path.")
    if not getattr(args, "diarized_speaker", None):
        raise SystemExit("The job 'extract-diarized-speaker' requires --diarized-speaker.")
    if not getattr(args, "speaker", None):
        raise SystemExit("The job 'extract-diarized-speaker' requires --speaker.")
    if not getattr(args, "run_id", None):
        raise SystemExit("The job 'extract-diarized-speaker' requires --run-id.")

    command = [
        "python3",
        str(SCRIPT_PATH),
        "--namespace",
        namespace,
        "--episode-id",
        args.episode_id,
        "--json-path",
        args.json_path,
        "--diarized-speaker",
        args.diarized_speaker,
        "--speaker",
        args.speaker,
        "--run-id",
        args.run_id,
    ]

    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")

    return command
