"""Pipeline job for exporting review samples by diarized speaker."""

from __future__ import annotations

from pathlib import Path


JOB_NAME = "export-speaker-samples"
SCRIPT_PATH = Path(__file__).resolve().parent / "export_speaker_samples.py"


def build_command(args) -> list[str]:
    namespace = getattr(args, "namespace", None) or getattr(args, "series", None)
    if not namespace:
        raise SystemExit("The job 'export-speaker-samples' requires --namespace.")
    if not getattr(args, "episode_id", None):
        raise SystemExit("The job 'export-speaker-samples' requires --episode-id.")
    if not getattr(args, "json_path", None):
        raise SystemExit("The job 'export-speaker-samples' requires --json-path.")

    command = [
        "python3",
        str(SCRIPT_PATH),
        "--namespace",
        namespace,
        "--episode-id",
        args.episode_id,
        "--json-path",
        args.json_path,
    ]

    if getattr(args, "run_id", None):
        command.extend(["--run-id", args.run_id])
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")

    return command
