#!/usr/bin/env python3
"""Extract audio tracks from files in episodes/<series>/raw/."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODES_DIR = REPO_ROOT / "episodes"
METADATA_DIR = REPO_ROOT / "metadata"
SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract audio from source files under episodes/<series>/raw/."
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--series",
        help="Series slug under episodes/, for example 'chaves'.",
    )
    target_group.add_argument(
        "--input",
        help="Single input file to process.",
    )
    parser.add_argument(
        "--format",
        default="wav",
        choices=["wav", "mp3", "flac"],
        help="Output audio format. Default: wav.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Output sample rate in Hz. Default: 22050.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of output channels. Default: 1.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracted audio files.",
    )
    parser.add_argument(
        "--speaker",
        help="Speaker slug for speaker-specific runs, for example 'goku' or 'seu-madruga'.",
    )
    parser.add_argument(
        "--run-id",
        help="Run identifier used under metadata/<series>/speakers/runs/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned ffmpeg commands without executing them.",
    )
    return parser.parse_args()


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found in PATH.")


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.is_file():
            sys.exit(f"Input file not found: {input_path}")
        return [input_path]

    raw_dir = EPISODES_DIR / args.series / "raw"
    if not raw_dir.is_dir():
        sys.exit(f"Series raw directory not found: {raw_dir}")

    files = [
        path
        for path in sorted(raw_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        sys.exit(f"No supported media files found in: {raw_dir}")
    return files


def infer_series_from_input(input_path: Path) -> str | None:
    if "episodes" not in input_path.parts:
        return None
    try:
        series_index = input_path.parts.index("episodes") + 1
        return input_path.parts[series_index]
    except (ValueError, IndexError):
        return None


def resolve_series(args: argparse.Namespace, input_path: Path) -> str | None:
    if args.series:
        return args.series
    return infer_series_from_input(input_path)


def output_path_for(input_path: Path, args: argparse.Namespace) -> Path:
    series = resolve_series(args, input_path)

    if args.speaker or args.run_id:
        if not series:
            sys.exit("Could not infer series for speaker-specific output.")
        if not args.speaker or not args.run_id:
            sys.exit("Use --speaker and --run-id together for speaker-specific output.")
        output_dir = (
            METADATA_DIR
            / series
            / "speakers"
            / "runs"
            / args.run_id
            / args.speaker
            / "extracted-audio"
        )
    elif args.input:
        if not series:
            output_dir = input_path.parent
        else:
            output_dir = EPISODES_DIR / series / "extracted-audio"
    else:
        output_dir = EPISODES_DIR / args.series / "extracted-audio"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}.{args.format}"


def ffmpeg_command(input_path: Path, output_path: Path, args: argparse.Namespace) -> list[str]:
    command = [
        "ffmpeg",
        "-nostdin",
        "-y" if args.force else "-n",
        "-i",
        str(input_path),
        "-vn",
        "-ar",
        str(args.sample_rate),
        "-ac",
        str(args.channels),
    ]

    if args.format == "wav":
        command.extend(["-c:a", "pcm_s16le"])
    elif args.format == "mp3":
        command.extend(["-c:a", "libmp3lame", "-b:a", "192k"])
    elif args.format == "flac":
        command.extend(["-c:a", "flac"])

    command.append(str(output_path))
    return command


def process_file(input_path: Path, args: argparse.Namespace) -> int:
    output_path = output_path_for(input_path, args)
    if output_path.exists() and not args.force:
        print(f"skip {input_path} -> {output_path} (already exists)")
        return 0

    command = ffmpeg_command(input_path, output_path, args)
    print(f"extract {input_path} -> {output_path}")
    if args.dry_run:
        print(" ".join(command))
        return 0

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"ffmpeg failed for: {input_path}", file=sys.stderr)
    return result.returncode


def main() -> int:
    args = parse_args()
    ensure_ffmpeg()
    failures = 0

    for input_path in resolve_inputs(args):
        failures += 1 if process_file(input_path, args) != 0 else 0

    if failures:
        print(f"completed with {failures} failure(s)", file=sys.stderr)
        return 1

    print("audio extraction completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
