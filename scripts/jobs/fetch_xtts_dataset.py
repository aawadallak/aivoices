#!/usr/bin/env python3
"""Fetch an XTTS dataset package from remote storage with rclone."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch an XTTS dataset package with rclone.")
    parser.add_argument("--remote-prefix", required=True, help="Remote prefix, e.g. r2:aivoices/training/datasets.")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True, help="Local target directory.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("rclone") is None:
        raise SystemExit("rclone not found in PATH")
    src = f"{args.remote_prefix.rstrip('/')}/{args.namespace}/{args.voice}/{args.dataset_name}"
    dst = str(Path(args.output_dir).expanduser().resolve())
    cmd = ["rclone", "copy", src, dst, "-P", "--create-empty-src-dirs"]
    if args.dry_run:
        cmd.append("--dry-run")
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
