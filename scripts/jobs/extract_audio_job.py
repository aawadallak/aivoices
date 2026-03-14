"""Pipeline job for raw audio extraction."""

from __future__ import annotations

from pathlib import Path


JOB_NAME = "extract-audio"
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "preprocess" / "extract_audio.py"


def build_command(args) -> list[str]:
    command = ["python3", str(SCRIPT_PATH)]
    namespace = getattr(args, "namespace", None) or getattr(args, "series", None)

    if args.input:
        command.extend(["--input", args.input])
    else:
        if not namespace:
            raise SystemExit("The job 'extract-audio' requires --namespace.")
        command.extend(["--namespace", namespace])

    command.extend(
        [
            "--format",
            args.format,
            "--sample-rate",
            str(args.sample_rate),
            "--channels",
            str(args.channels),
        ]
    )

    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    if args.speaker:
        command.extend(["--speaker", args.speaker])
    if args.run_id:
        command.extend(["--run-id", args.run_id])

    return command
