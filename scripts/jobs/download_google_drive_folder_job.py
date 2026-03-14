"""Pipeline job for Google Drive folder download."""

from __future__ import annotations

from pathlib import Path


JOB_NAME = "download-google-drive-folder"
SCRIPT_PATH = Path(__file__).resolve().parent / "download_google_drive_folder.py"


def build_command(args) -> list[str]:
    if not getattr(args, "url", None):
        raise SystemExit("The job 'download-google-drive-folder' requires --url.")
    if not getattr(args, "workspace", None):
        raise SystemExit("The job 'download-google-drive-folder' requires --workspace.")
    namespace = getattr(args, "namespace", None) or getattr(args, "series", None)
    if not namespace:
        raise SystemExit("The job 'download-google-drive-folder' requires --namespace.")

    command = [
        "python3",
        str(SCRIPT_PATH),
        "--url",
        args.url,
        "--namespace",
        namespace,
        "--workspace",
        args.workspace,
    ]

    if args.dry_run:
        command.append("--dry-run")

    return command
