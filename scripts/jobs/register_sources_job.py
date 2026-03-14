"""Pipeline job for source manifest registration."""

from __future__ import annotations

from pathlib import Path


JOB_NAME = "register-sources"
SCRIPT_PATH = Path(__file__).resolve().parent / "register_sources.py"


def build_command(args) -> list[str]:
    namespace = getattr(args, "namespace", None) or getattr(args, "series", None)
    if not namespace:
        raise SystemExit("The job 'register-sources' requires --namespace.")

    command = [
        "python3",
        str(SCRIPT_PATH),
        "--namespace",
        namespace,
    ]

    if getattr(args, "language", None):
        command.extend(["--language", args.language])
    if args.dry_run:
        command.append("--dry-run")

    return command
