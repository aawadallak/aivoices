"""Pipeline job for review chunk export."""

from __future__ import annotations

from pathlib import Path


JOB_NAME = "export-review-chunks"
SCRIPT_PATH = Path(__file__).resolve().parent / "export_review_chunks.py"


def build_command(args) -> list[str]:
    namespace = getattr(args, "namespace", None) or getattr(args, "series", None)
    if not namespace:
        raise SystemExit("The job 'export-review-chunks' requires --namespace.")
    if not getattr(args, "episode_id", None):
        raise SystemExit("The job 'export-review-chunks' requires --episode-id.")

    command = [
        "python3",
        str(SCRIPT_PATH),
        "--namespace",
        namespace,
        "--episode-id",
        args.episode_id,
    ]

    if getattr(args, "input", None):
        command.extend(["--input", args.input])
    if getattr(args, "run_id", None):
        command.extend(["--run-id", args.run_id])
    if getattr(args, "chunk_seconds", None):
        command.extend(["--chunk-seconds", str(args.chunk_seconds)])
    if getattr(args, "limit_seconds", None):
        command.extend(["--limit-seconds", str(args.limit_seconds)])
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")

    return command
