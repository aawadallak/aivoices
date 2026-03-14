#!/usr/bin/env python3
"""Enhance a candidate clip with a lightweight ffmpeg chain or optional Demucs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhance a candidate clip for manual comparison.")
    parser.add_argument("--input", required=True, help="Input clip path.")
    parser.add_argument("--output", required=True, help="Output clip path.")
    parser.add_argument(
        "--method",
        choices=["ffmpeg-basic", "demucs"],
        default="ffmpeg-basic",
        help="Enhancement method. Default: ffmpeg-basic.",
    )
    return parser.parse_args()


def ffmpeg_basic(input_path: Path, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-af",
        "highpass=f=100,lowpass=f=7600,afftdn=nf=-25,volume=1.2",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]


def demucs_command(input_path: Path, output_dir: Path) -> list[str]:
    return [
        "python3",
        "-m",
        "demucs.separate",
        "-o",
        str(output_dir),
        "--two-stems=vocals",
        str(input_path),
    ]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "ffmpeg-basic":
        if shutil.which("ffmpeg") is None:
            sys.exit("ffmpeg was not found in PATH.")
        return subprocess.run(ffmpeg_basic(input_path, output_path), check=False).returncode

    if shutil.which("python3") is None:
        sys.exit("python3 was not found in PATH.")
    temp_dir = output_path.parent / ".demucs-temp"
    code = subprocess.run(demucs_command(input_path, temp_dir), check=False).returncode
    if code != 0:
        return code

    vocals = next(temp_dir.rglob("vocals.wav"), None)
    if vocals is None:
        sys.exit("Demucs finished but vocals.wav was not found.")
    shutil.copy2(vocals, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
