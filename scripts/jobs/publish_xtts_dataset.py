#!/usr/bin/env python3
"""Publish an XTTS dataset package to remote storage with rclone."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish an XTTS dataset package with rclone.")
    parser.add_argument("--dataset-dir", required=True, help="Local XTTS package directory.")
    parser.add_argument("--remote-prefix", required=True, help="Remote prefix, e.g. r2:aivoices/training/datasets.")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("rclone") is None:
        raise SystemExit("rclone not found in PATH")
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")
    dst = f"{args.remote_prefix.rstrip('/')}/{args.namespace}/{args.voice}/{args.dataset_name}"
    cmd = [
        "rclone",
        "copy",
        str(dataset_dir),
        dst,
        "-P",
        "--create-empty-src-dirs",
        "--exclude",
        "*.zip",
        "--exclude",
        "*.rar",
        "--exclude",
        "*.7z",
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
