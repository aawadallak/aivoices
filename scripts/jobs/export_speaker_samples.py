#!/usr/bin/env python3
"""Export short review clips grouped by diarized speaker."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODES_DIR = REPO_ROOT / "episodes"
METADATA_DIR = REPO_ROOT / "metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export short review samples grouped by speaker from a WhisperX JSON result."
    )
    parser.add_argument("--namespace", required=True, help="Namespace slug.")
    parser.add_argument("--episode-id", required=True, help="Episode basename without extension.")
    parser.add_argument("--json-path", required=True, help="Path to WhisperX JSON output.")
    parser.add_argument(
        "--run-id",
        default="speaker-sample-export",
        help="Run identifier under metadata/<namespace>/speakers/runs/.",
    )
    parser.add_argument(
        "--max-clips-per-speaker",
        type=int,
        default=6,
        help="Maximum number of clips to export per speaker. Default: 6.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.5,
        help="Minimum accepted segment duration in seconds. Default: 1.5.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=12.0,
        help="Maximum accepted segment duration in seconds. Default: 12.0.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Padding in seconds added to both sides of the clip. Default: 0.15.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing sample files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned exports without writing files.",
    )
    return parser.parse_args()


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found in PATH.")


def resolve_audio_path(namespace: str, episode_id: str) -> Path:
    path = EPISODES_DIR / namespace / "extracted-audio" / f"{episode_id}.wav"
    if not path.is_file():
        sys.exit(f"Extracted audio not found: {path}")
    return path


def resolve_json_path(json_path: str) -> Path:
    path = Path(json_path).expanduser().resolve()
    if not path.is_file():
        sys.exit(f"WhisperX JSON not found: {path}")
    return path


def load_segments(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text())
    return data.get("segments", [])


def output_root(namespace: str, run_id: str, episode_id: str) -> Path:
    root = (
        METADATA_DIR
        / namespace
        / "speakers"
        / "runs"
        / run_id
        / "review"
        / episode_id
        / "speaker-samples"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def filter_segments(segments: list[dict], min_duration: float, max_duration: float) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for seg in segments:
        speaker = seg.get("speaker")
        if not speaker:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        duration = end - start
        text = (seg.get("text") or "").strip()
        if duration < min_duration or duration > max_duration:
            continue
        if not text:
            continue
        grouped.setdefault(speaker, []).append(seg)

    for speaker, items in grouped.items():
        items.sort(key=lambda item: (-(float(item["end"]) - float(item["start"])), float(item["start"])))
    return grouped


def clip_command(audio_path: Path, start: float, end: float, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(audio_path),
        "-vn",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]


def sanitize_text(text: str) -> str:
    text = " ".join(text.split())
    return text[:160]


def write_summary(summary_path: Path, rows: list[dict[str, str]]) -> None:
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "speaker",
                "sample_index",
                "start_sec",
                "end_sec",
                "duration_sec",
                "text",
                "audio_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    ensure_ffmpeg()
    audio_path = resolve_audio_path(args.namespace, args.episode_id)
    json_path = resolve_json_path(args.json_path)
    segments = load_segments(json_path)
    grouped = filter_segments(segments, args.min_duration, args.max_duration)
    root = output_root(args.namespace, args.run_id, args.episode_id)
    summary_rows: list[dict[str, str]] = []

    for speaker, items in sorted(grouped.items()):
        speaker_dir = root / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)

        for index, seg in enumerate(items[: args.max_clips_per_speaker], start=1):
            raw_start = float(seg["start"])
            raw_end = float(seg["end"])
            start = max(0.0, raw_start - args.padding)
            end = raw_end + args.padding
            output_path = speaker_dir / f"{speaker.lower()}-sample-{index:03d}.wav"

            summary_rows.append(
                {
                    "speaker": speaker,
                    "sample_index": str(index),
                    "start_sec": f"{raw_start:.3f}",
                    "end_sec": f"{raw_end:.3f}",
                    "duration_sec": f"{raw_end - raw_start:.3f}",
                    "text": sanitize_text(seg.get("text", "")),
                    "audio_path": str(output_path.relative_to(REPO_ROOT)),
                }
            )

            if output_path.exists() and not args.force:
                print(f"skip {output_path} (already exists)")
                continue

            command = clip_command(audio_path, start, end, output_path)
            print(f"export {speaker} sample {index:03d} -> {output_path}")
            if args.dry_run:
                print(" ".join(command))
                continue

            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                print(f"ffmpeg failed for {speaker} sample {index:03d}", file=sys.stderr)
                return 1

    summary_path = root / "speaker-samples.csv"
    if args.dry_run:
        print(f"would write summary: {summary_path}")
    else:
        write_summary(summary_path, summary_rows)
        counts = Counter(row["speaker"] for row in summary_rows)
        print(f"exported {sum(counts.values())} samples across {len(counts)} speakers")
        print(f"summary written to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
