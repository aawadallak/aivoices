#!/usr/bin/env python3
"""Extract matched speaker segments from a diarized WhisperX JSON result."""

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
        description="Extract speaker clips from a target WhisperX JSON using a speaker match CSV."
    )
    parser.add_argument("--namespace", required=True, help="Namespace slug.")
    parser.add_argument("--episode-id", required=True, help="Episode basename without extension.")
    parser.add_argument("--target-json", required=True, help="Target WhisperX JSON path.")
    parser.add_argument("--match-csv", required=True, help="Speaker match CSV path.")
    parser.add_argument("--run-id", required=True, help="Run identifier for extracted clips.")
    parser.add_argument("--padding", type=float, default=0.12, help="Padding around each segment. Default: 0.12.")
    parser.add_argument("--min-duration", type=float, default=0.8, help="Minimum segment duration. Default: 0.8.")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Maximum segment duration. Default: 15.0.")
    parser.add_argument("--include-review", action="store_true", help="Also extract rows marked as review.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing clips.")
    return parser.parse_args()


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found in PATH.")


def load_json(path_str: str) -> dict:
    path = Path(path_str).expanduser().resolve()
    return json.loads(path.read_text())


def load_matches(path_str: str, include_review: bool) -> tuple[str, object]:
    path = Path(path_str).expanduser().resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and rows[0].get("start_sec"):
        filtered = [
            row
            for row in rows
            if row["status"] == "matched" or include_review
        ]
        return ("segment_rows", filtered)
    mapping: dict[str, str] = {}
    for row in rows:
        if row["status"] == "matched" or include_review:
            mapping[row["target_diarized_speaker"]] = row["matched_speaker"]
    return ("speaker_map", mapping)


def audio_path(namespace: str, episode_id: str) -> Path:
    path = EPISODES_DIR / namespace / "extracted-audio" / f"{episode_id}.wav"
    if not path.is_file():
        sys.exit(f"Extracted audio not found: {path}")
    return path


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


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "matched_speaker",
                "clip_index",
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
    target_json = load_json(args.target_json)
    match_mode, match_data = load_matches(args.match_csv, args.include_review)
    source_audio = audio_path(args.namespace, args.episode_id)

    out_root = (
        METADATA_DIR
        / args.namespace
        / "speakers"
        / "runs"
        / args.run_id
        / "matched-clips"
        / args.episode_id
    )
    out_root.mkdir(parents=True, exist_ok=True)

    counters: Counter[str] = Counter()
    manifest_rows: list[dict[str, str]] = []

    if match_mode == "segment_rows":
        segment_rows = match_data
    else:
        speaker_map = match_data
        segment_rows = []
        for segment in target_json.get("segments", []):
            diarized = segment.get("speaker")
            matched = speaker_map.get(diarized)
            if not matched:
                continue
            segment_rows.append(
                {
                    "matched_speaker": matched,
                    "start_sec": f"{float(segment.get('start', 0.0)):.3f}",
                    "end_sec": f"{float(segment.get('end', 0.0)):.3f}",
                    "text": " ".join((segment.get("text") or "").split())[:160],
                }
            )

    for segment in segment_rows:
        matched = segment["matched_speaker"]
        start = float(segment.get("start_sec", 0.0))
        end = float(segment.get("end_sec", 0.0))
        duration = end - start
        if duration < args.min_duration or duration > args.max_duration:
            continue

        counters[matched] += 1
        clip_index = counters[matched]
        speaker_dir = out_root / matched
        speaker_dir.mkdir(parents=True, exist_ok=True)
        output_path = speaker_dir / f"{matched}-clip-{clip_index:04d}.wav"

        manifest_rows.append(
            {
                "matched_speaker": matched,
                "clip_index": str(clip_index),
                "start_sec": f"{start:.3f}",
                "end_sec": f"{end:.3f}",
                "duration_sec": f"{duration:.3f}",
                "text": " ".join((segment.get("text") or "").split())[:160],
                "audio_path": str(output_path.relative_to(REPO_ROOT)),
            }
        )

        if output_path.exists() and not args.force:
            continue

        pad_start = max(0.0, start - args.padding)
        pad_end = end + args.padding
        result = subprocess.run(clip_command(source_audio, pad_start, pad_end, output_path), check=False)
        if result.returncode != 0:
            print(f"ffmpeg failed for {matched} clip {clip_index}", file=sys.stderr)
            return 1

    manifest_path = out_root / "matched-clips.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"exported {len(manifest_rows)} matched clips to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
