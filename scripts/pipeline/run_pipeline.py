#!/usr/bin/env python3
"""Run a simple sequence of dataset jobs."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.registry import JOBS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sequence of pipeline jobs for a dataset execution."
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--namespace",
        dest="namespace",
        help="Namespace slug under episodes/, for example 'dragonball'.",
    )
    target_group.add_argument(
        "--series",
        dest="namespace",
        help="Deprecated alias for --namespace.",
    )
    target_group.add_argument(
        "--input",
        help="Single input file to process.",
    )
    parser.add_argument(
        "--jobs",
        required=True,
        help="Comma-separated job list, for example 'extract-audio'.",
    )
    parser.add_argument(
        "--url",
        help="Source URL for jobs that download remote media.",
    )
    parser.add_argument(
        "--episode-id",
        help="Episode or source basename for compatible jobs.",
    )
    parser.add_argument(
        "--workspace",
        help="Workspace folder under episodes/<namespace>/raw/ for compatible jobs.",
    )
    parser.add_argument(
        "--json-path",
        help="Path to a JSON artifact for compatible jobs.",
    )
    parser.add_argument(
        "--name",
        help="Optional fixed basename for compatible jobs.",
    )
    parser.add_argument(
        "--download-format",
        default="bv*+ba/b",
        help="yt-dlp format selector for download jobs. Default: bv*+ba/b",
    )
    parser.add_argument(
        "--speaker",
        help="Speaker slug for speaker-specific runs.",
    )
    parser.add_argument(
        "--diarized-speaker",
        help="WhisperX diarized speaker label for compatible jobs, for example SPEAKER_03.",
    )
    parser.add_argument(
        "--run-id",
        help="Run identifier for speaker-specific outputs.",
    )
    parser.add_argument(
        "--format",
        default="wav",
        choices=["wav", "mp3", "flac"],
        help="Output audio format for compatible jobs. Default: wav.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Output sample rate in Hz for compatible jobs. Default: 16000.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of output channels for compatible jobs. Default: 1.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        help="Chunk size in seconds for compatible jobs.",
    )
    parser.add_argument(
        "--limit-seconds",
        type=int,
        help="Maximum number of seconds to process from the start for compatible jobs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite outputs when supported by a job.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each job command without executing it.",
    )
    parser.add_argument(
        "--write-info-json",
        action="store_true",
        help="Write yt-dlp metadata JSON for compatible jobs.",
    )
    parser.add_argument(
        "--write-thumbnail",
        action="store_true",
        help="Write thumbnails for compatible jobs.",
    )
    parser.add_argument(
        "--language",
        help="Default language for compatible manifest jobs, for example 'pt-BR'.",
    )
    return parser.parse_args()


def parse_jobs(raw_jobs: str) -> list[str]:
    jobs = [job.strip() for job in raw_jobs.split(",") if job.strip()]
    if not jobs:
        sys.exit("At least one job must be provided in --jobs.")
    unknown = [job for job in jobs if job not in JOBS]
    if unknown:
        available = ", ".join(sorted(JOBS))
        sys.exit(f"Unknown job(s): {', '.join(unknown)}. Available jobs: {available}")
    return jobs


def validate_args(args: argparse.Namespace) -> None:
    if bool(args.speaker) != bool(args.run_id):
        sys.exit("Use --speaker and --run-id together.")


def run_job(job_name: str, args: argparse.Namespace) -> int:
    job = JOBS[job_name]
    command = job.build_command(args)
    print(f"[job:{job_name}] {shlex.join(command)}")
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def main() -> int:
    args = parse_args()
    validate_args(args)
    failures = 0

    for job_name in parse_jobs(args.jobs):
        exit_code = run_job(job_name, args)
        if exit_code != 0:
            failures += 1
            print(f"job failed: {job_name} (exit code {exit_code})", file=sys.stderr)
            break

    if failures:
        return 1

    print("pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
