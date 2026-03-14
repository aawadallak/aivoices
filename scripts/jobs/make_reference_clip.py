#!/usr/bin/env python3
"""Create and auto-validate a speaker reference clip."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reference clip from an existing speaker audio file."
    )
    parser.add_argument("--input", required=True, help="Input audio file.")
    parser.add_argument("--series", required=True, help="Series slug.")
    parser.add_argument("--speaker", required=True, help="Speaker slug.")
    parser.add_argument("--start", required=True, help="Start timestamp, for example 00:00:50.")
    parser.add_argument("--end", required=True, help="End timestamp, for example 00:01:04.")
    parser.add_argument("--label", required=True, help="Source label, for example batalha-dos-deuses-001.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate. Default: 16000.")
    parser.add_argument("--channels", type=int, default=1, help="Output channels. Default: 1.")
    parser.add_argument("--min-duration", type=float, default=6.0, help="Minimum approved duration in seconds.")
    parser.add_argument("--max-duration", type=float, default=20.0, help="Maximum approved duration in seconds.")
    parser.add_argument(
        "--max-edge-silence",
        type=float,
        default=0.75,
        help="Maximum allowed leading or trailing silence in seconds.",
    )
    parser.add_argument(
        "--min-speech-ratio",
        type=float,
        default=0.6,
        help="Minimum ratio of non-silence to total duration.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite if the output already exists.")
    return parser.parse_args()


def ensure_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} was not found in PATH.")


def run_command(command: list[str], capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=capture_output,
    )


def probe_duration(path: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        sys.exit(f"Could not probe duration for: {path}")
    return float(result.stdout.strip())


def detect_silence(path: Path) -> tuple[float, float, float]:
    result = run_command(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(path),
            "-af",
            "silencedetect=noise=-35dB:d=0.30",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
    )
    stderr = result.stderr
    duration = probe_duration(path)
    starts = [float(value) for value in re.findall(r"silence_start: ([0-9.]+)", stderr)]
    ends = [
        (float(end), float(dur))
        for end, dur in re.findall(r"silence_end: ([0-9.]+) \| silence_duration: ([0-9.]+)", stderr)
    ]

    leading = 0.0
    trailing = 0.0
    total_silence = sum(item[1] for item in ends)

    if starts and ends:
        first_start = starts[0]
        first_end = ends[0][0]
        if first_start <= 0.05:
            leading = first_end
        last_end = ends[-1][0]
        last_duration = ends[-1][1]
        if abs(last_end - duration) <= 0.05:
            trailing = last_duration

    return leading, trailing, total_silence


def slug_timestamp(value: str) -> str:
    parts = value.split(":")
    if len(parts) != 3:
        sys.exit(f"Timestamp must be HH:MM:SS, got: {value}")
    hours, minutes, seconds = parts
    total_minutes = int(hours) * 60 + int(minutes)
    return f"{total_minutes:02d}m{int(seconds):02d}s"


def next_reference_index(series: str, speaker: str) -> int:
    speaker_dir = METADATA_DIR / series / "speakers" / "references" / speaker
    numbers = []
    for path in speaker_dir.rglob("*.wav"):
        match = re.search(r"-reference-(\d+)-", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return (max(numbers) + 1) if numbers else 1


def build_output_path(args: argparse.Namespace, status: str, index: int) -> Path:
    speaker_dir = METADATA_DIR / args.series / "speakers" / "references" / args.speaker / status
    speaker_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{args.series}-{args.speaker}-reference-{index:03d}-{args.label}-"
        f"{slug_timestamp(args.start)}-{slug_timestamp(args.end)}.wav"
    )
    return speaker_dir / name


def extract_clip(args: argparse.Namespace, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y" if args.force else "-n",
        "-i",
        str(Path(args.input).resolve()),
        "-ss",
        args.start,
        "-to",
        args.end,
        "-ar",
        str(args.sample_rate),
        "-ac",
        str(args.channels),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = run_command(command)
    if result.returncode != 0:
        sys.exit("ffmpeg failed while creating the reference clip.")


def validate_clip(path: Path, args: argparse.Namespace) -> dict[str, float | bool | str]:
    duration = probe_duration(path)
    leading, trailing, total_silence = detect_silence(path)
    speech_ratio = max(0.0, 1.0 - (total_silence / duration)) if duration > 0 else 0.0

    checks = {
        "duration_ok": args.min_duration <= duration <= args.max_duration,
        "leading_silence_ok": leading <= args.max_edge_silence,
        "trailing_silence_ok": trailing <= args.max_edge_silence,
        "speech_ratio_ok": speech_ratio >= args.min_speech_ratio,
    }

    return {
        "duration_sec": round(duration, 3),
        "leading_silence_sec": round(leading, 3),
        "trailing_silence_sec": round(trailing, 3),
        "total_silence_sec": round(total_silence, 3),
        "speech_ratio": round(speech_ratio, 3),
        "approved": all(checks.values()),
        "checks": checks,
    }


def write_report(output_path: Path, validation: dict[str, object], args: argparse.Namespace) -> None:
    report = {
        "input_path": str(Path(args.input).resolve()),
        "output_path": str(output_path),
        "series": args.series,
        "speaker": args.speaker,
        "label": args.label,
        "start": args.start,
        "end": args.end,
        "validation": validation,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_tools()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        sys.exit(f"Input file not found: {input_path}")

    index = next_reference_index(args.series, args.speaker)
    pending_path = build_output_path(args, "pending", index)
    extract_clip(args, pending_path)
    validation = validate_clip(pending_path, args)

    final_status = "approved" if validation["approved"] else "pending"
    final_path = build_output_path(args, final_status, index)
    if pending_path != final_path:
        pending_path.replace(final_path)
    write_report(final_path, validation, args)

    print(f"reference clip created: {final_path}")
    print(f"approved: {validation['approved']}")
    print(f"duration_sec: {validation['duration_sec']}")
    print(f"speech_ratio: {validation['speech_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
