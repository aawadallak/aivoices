#!/usr/bin/env python3
"""Download a YouTube video into a series-specific workspace."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODES_DIR = REPO_ROOT / "episodes"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video into episodes/<series>/raw/<workspace>/."
    )
    parser.add_argument("--url", required=True, help="YouTube video URL.")
    parser.add_argument("--series", required=True, help="Series slug, for example 'dragonball'.")
    parser.add_argument(
        "--workspace",
        required=True,
        help="Workspace folder under episodes/<series>/raw/, for example 'batalha-dos-deuses'.",
    )
    parser.add_argument(
        "--name",
        help="Optional fixed basename for the downloaded file, without extension.",
    )
    parser.add_argument(
        "--download-format",
        default="bv*+ba/b",
        help="yt-dlp format selector. Default: bv*+ba/b",
    )
    parser.add_argument(
        "--write-info-json",
        action="store_true",
        help="Write yt-dlp metadata JSON next to the downloaded file.",
    )
    parser.add_argument(
        "--write-thumbnail",
        action="store_true",
        help="Write the thumbnail next to the downloaded file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the yt-dlp command without executing it.",
    )
    return parser.parse_args()


def ensure_yt_dlp() -> None:
    if shutil.which("yt-dlp") is None:
        sys.exit("yt-dlp was not found in PATH.")


def build_output_dir(args: argparse.Namespace) -> Path:
    output_dir = EPISODES_DIR / args.series / "raw" / slugify(args.workspace)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_output_template(args: argparse.Namespace) -> str:
    if args.name:
        return f"{slugify(args.name)}.%(ext)s"
    return "%(title).180B [%(id)s].%(ext)s"


def build_command(args: argparse.Namespace) -> list[str]:
    output_dir = build_output_dir(args)
    command = [
        "yt-dlp",
        "--no-progress",
        "--restrict-filenames",
        "--merge-output-format",
        "mkv",
        "-f",
        args.download_format,
        "-P",
        str(output_dir),
        "-o",
        build_output_template(args),
        args.url,
    ]

    if args.write_info_json:
        command.append("--write-info-json")
    if args.write_thumbnail:
        command.append("--write-thumbnail")

    return command


def main() -> int:
    args = parse_args()
    ensure_yt_dlp()
    command = build_command(args)
    print(" ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


JOB_NAME = "download-youtube-video"


if __name__ == "__main__":
    raise SystemExit(main())
