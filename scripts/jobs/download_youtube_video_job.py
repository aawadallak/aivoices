"""Pipeline job for YouTube video download."""

from __future__ import annotations

from pathlib import Path


JOB_NAME = "download-youtube-video"
SCRIPT_PATH = Path(__file__).resolve().parent / "download_youtube_video.py"


def build_command(args) -> list[str]:
    if not getattr(args, "url", None):
        raise SystemExit("The job 'download-youtube-video' requires --url.")
    if not getattr(args, "workspace", None):
        raise SystemExit("The job 'download-youtube-video' requires --workspace.")
    namespace = getattr(args, "namespace", None) or getattr(args, "series", None)
    if not namespace:
        raise SystemExit("The job 'download-youtube-video' requires --namespace.")

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

    if getattr(args, "name", None):
        command.extend(["--name", args.name])
    if getattr(args, "download_format", None):
        command.extend(["--download-format", args.download_format])
    if getattr(args, "write_info_json", False):
        command.append("--write-info-json")
    if getattr(args, "write_thumbnail", False):
        command.append("--write-thumbnail")
    if args.dry_run:
        command.append("--dry-run")

    return command
