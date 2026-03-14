#!/usr/bin/env python3
"""Download a Google Drive folder into a namespace-specific workspace."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODES_DIR = REPO_ROOT / "episodes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Google Drive folder into episodes/<namespace>/raw/<workspace>/."
    )
    parser.add_argument("--url", required=True, help="Google Drive folder URL.")
    parser.add_argument("--namespace", "--series", dest="namespace", required=True, help="Namespace slug.")
    parser.add_argument(
        "--workspace",
        required=True,
        help="Workspace folder under episodes/<namespace>/raw/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gdown command without executing it.",
    )
    return parser.parse_args()


def ensure_gdown() -> None:
    if shutil.which("gdown") is None:
        sys.exit("gdown was not found in PATH. Install it before using this job.")


def build_output_dir(args: argparse.Namespace) -> Path:
    output_dir = EPISODES_DIR / args.namespace / "raw" / args.workspace
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_command(args: argparse.Namespace) -> list[str]:
    output_dir = build_output_dir(args)
    return [
        "gdown",
        "--folder",
        args.url,
        "--output",
        str(output_dir),
        "--remaining-ok",
    ]


def main() -> int:
    args = parse_args()
    ensure_gdown()
    command = build_command(args)
    print(" ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


JOB_NAME = "download-google-drive-folder"


if __name__ == "__main__":
    raise SystemExit(main())
