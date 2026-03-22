#!/usr/bin/env python3
"""Fetch an RVC dataset package from R2 remote storage."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch an RVC dataset from R2.")
    parser.add_argument("--remote-prefix", required=True, help="rclone remote prefix, e.g. r2:aivoices/training/datasets")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True, help="Local target directory.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not shutil.which("rclone"):
        raise SystemExit("rclone is not installed or not in PATH")

    src = f"{args.remote_prefix.rstrip('/')}/{args.namespace}/{args.voice}/{args.dataset_name}"
    dst = Path(args.output_dir).expanduser().resolve()

    cmd = ["rclone", "copy", src, str(dst), "-P", "--create-empty-src-dirs"]
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"[fetch-rvc] {src} -> {dst}")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
