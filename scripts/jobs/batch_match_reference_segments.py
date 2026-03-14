#!/usr/bin/env python3
"""Run conservative reference-based matching for all diarized JSON files in a namespace."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODES_DIR = REPO_ROOT / "episodes"
METADATA_DIR = REPO_ROOT / "metadata"
MATCH_SCRIPT = Path(__file__).resolve().parent / "match_reference_segments.py"
EXTRACT_SCRIPT = Path(__file__).resolve().parent / "extract_matched_segments.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run conservative reference-based matching across all diarized JSON files in a namespace."
    )
    parser.add_argument("--namespace", "--series", dest="namespace", required=True, help="Namespace slug.")
    parser.add_argument(
        "--transcripts-subdir",
        default="whisperx-batch-diarized",
        help="Subdirectory under metadata/<namespace>/transcripts/. Default: whisperx-batch-diarized.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern inside the transcripts subdir. Default: *.json.",
    )
    parser.add_argument(
        "--run-prefix",
        default="reference-match",
        help="Run id prefix under metadata/<namespace>/speakers/runs/. Default: reference-match.",
    )
    parser.add_argument(
        "--allowed-speakers",
        required=True,
        help="Comma-separated speakers to match, for example bob-esponja,lula-molusco.",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Top-k references to average. Default: 2.")
    parser.add_argument("--mean-weight", type=float, default=0.4, help="Mean embedding weight. Default: 0.4.")
    parser.add_argument("--topk-weight", type=float, default=0.6, help="Top-k score weight. Default: 0.6.")
    parser.add_argument("--accept-threshold", type=float, default=0.72, help="Match threshold. Default: 0.72.")
    parser.add_argument("--review-threshold", type=float, default=0.62, help="Review threshold. Default: 0.62.")
    parser.add_argument("--min-margin", type=float, default=0.08, help="Minimum score margin. Default: 0.08.")
    parser.add_argument("--min-duration", type=float, default=1.2, help="Minimum duration. Default: 1.2.")
    parser.add_argument("--max-duration", type=float, default=12.0, help="Maximum duration. Default: 12.0.")
    parser.add_argument("--force", action="store_true", help="Overwrite extracted clips.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them.")
    return parser.parse_args()


def transcripts_dir(namespace: str, subdir: str) -> Path:
    path = METADATA_DIR / namespace / "transcripts" / subdir
    if not path.is_dir():
        sys.exit(f"Transcripts directory not found: {path}")
    return path


def extracted_audio_path(namespace: str, episode_id: str) -> Path:
    path = EPISODES_DIR / namespace / "extracted-audio" / f"{episode_id}.wav"
    if not path.is_file():
        sys.exit(f"Extracted audio not found for {episode_id}: {path}")
    return path


def run_command(command: list[str], dry_run: bool) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def main() -> int:
    args = parse_args()
    json_files = sorted(transcripts_dir(args.namespace, args.transcripts_subdir).glob(args.pattern))
    if not json_files:
        sys.exit("No diarized JSON files found.")

    failures = 0
    for json_path in json_files:
        episode_id = json_path.stem
        run_id = f"{args.run_prefix}-{episode_id}"
        run_root = METADATA_DIR / args.namespace / "speakers" / "runs" / run_id
        match_csv = run_root / "speaker-matches.csv"
        target_audio = extracted_audio_path(args.namespace, episode_id)

        match_command = [
            "python3",
            str(MATCH_SCRIPT),
            "--namespace",
            args.namespace,
            "--target-json",
            str(json_path),
            "--target-audio",
            str(target_audio),
            "--output-csv",
            str(match_csv),
            "--allowed-speakers",
            args.allowed_speakers,
            "--top-k",
            str(args.top_k),
            "--mean-weight",
            str(args.mean_weight),
            "--topk-weight",
            str(args.topk_weight),
            "--accept-threshold",
            str(args.accept_threshold),
            "--review-threshold",
            str(args.review_threshold),
            "--min-margin",
            str(args.min_margin),
            "--min-duration",
            str(args.min_duration),
            "--max-duration",
            str(args.max_duration),
        ]
        if run_command(match_command, args.dry_run) != 0:
            failures += 1
            print(f"matching failed for {episode_id}", file=sys.stderr)
            break

        extract_command = [
            "python3",
            str(EXTRACT_SCRIPT),
            "--namespace",
            args.namespace,
            "--episode-id",
            episode_id,
            "--target-json",
            str(json_path),
            "--match-csv",
            str(match_csv),
            "--run-id",
            run_id,
            "--min-duration",
            str(args.min_duration),
            "--max-duration",
            str(args.max_duration),
        ]
        if args.force:
            extract_command.append("--force")
        if run_command(extract_command, args.dry_run) != 0:
            failures += 1
            print(f"matched extraction failed for {episode_id}", file=sys.stderr)
            break

    if failures:
        return 1

    print(f"processed {len(json_files)} diarized episodes for namespace {args.namespace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
