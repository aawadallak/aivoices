#!/usr/bin/env python3
"""Export fixed-length review chunks from a source audio file."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from math import ceil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODES_DIR = REPO_ROOT / "episodes"
METADATA_DIR = REPO_ROOT / "metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fixed-length review chunks for manual speaker discovery."
    )
    parser.add_argument("--namespace", required=True, help="Namespace slug.")
    parser.add_argument("--episode-id", required=True, help="Episode basename without extension.")
    parser.add_argument(
        "--input",
        help="Optional explicit input audio path. Defaults to episodes/<namespace>/extracted-audio/<episode-id>.wav",
    )
    parser.add_argument(
        "--run-id",
        default="pilot-review",
        help="Run identifier under metadata/<namespace>/speakers/runs/. Default: pilot-review",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=20,
        help="Chunk length in seconds. Default: 20",
    )
    parser.add_argument(
        "--limit-seconds",
        type=int,
        default=600,
        help="Optional cap from the start of the file in seconds. Default: 600",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing chunk files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    return parser.parse_args()


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit("ffmpeg and ffprobe are required in PATH.")


def resolve_input(args: argparse.Namespace) -> Path:
    if args.input:
        input_path = Path(args.input).expanduser().resolve()
    else:
        input_path = (
            EPISODES_DIR / args.namespace / "extracted-audio" / f"{args.episode_id}.wav"
        ).resolve()
    if not input_path.is_file():
        sys.exit(f"Input audio not found: {input_path}")
    return input_path


def output_dir(args: argparse.Namespace) -> Path:
    out = (
        METADATA_DIR
        / args.namespace
        / "speakers"
        / "runs"
        / args.run_id
        / "review"
        / args.episode_id
        / "chunks"
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def audio_duration_seconds(input_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit(f"ffprobe failed for {input_path}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise SystemExit(f"Could not parse duration for {input_path}") from exc


def format_timestamp(seconds: int) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def chunk_command(input_path: Path, start: int, duration: int, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(input_path),
        "-vn",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "22050",
        "-ac",
        "1",
        str(output_path),
    ]


def main() -> int:
    args = parse_args()
    ensure_ffmpeg()
    input_path = resolve_input(args)
    out_dir = output_dir(args)
    total_seconds = int(audio_duration_seconds(input_path))
    review_limit = min(total_seconds, args.limit_seconds)
    total_chunks = ceil(review_limit / args.chunk_seconds)

    for index in range(total_chunks):
        start = index * args.chunk_seconds
        duration = min(args.chunk_seconds, review_limit - start)
        if duration <= 0:
            break

        start_label = format_timestamp(start)
        end_label = format_timestamp(start + duration)
        output_path = out_dir / (
            f"{args.namespace}-{args.episode_id}-review-"
            f"{index + 1:03d}-{start_label}-{end_label}.wav"
        )
        if output_path.exists() and not args.force:
            print(f"skip {output_path} (already exists)")
            continue

        command = chunk_command(input_path, start, duration, output_path)
        print(f"export chunk {index + 1:03d}: {output_path}")
        if args.dry_run:
            print(" ".join(command))
            continue

        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"ffmpeg failed for chunk starting at {start}s", file=sys.stderr)
            return 1

    print("review chunk export completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
