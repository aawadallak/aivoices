#!/usr/bin/env python3
"""Run clip quality scoring across matched-clip runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"
SCRIPT_PATH = Path(__file__).resolve().parent / "score_matched_clips.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score matched clips across multiple run ids.")
    parser.add_argument("--namespace", "--series", dest="namespace", required=True, help="Namespace slug.")
    parser.add_argument(
        "--run-prefix",
        default="reference-match-square-spongebob-episode-",
        help="Run id prefix to scan. Default matches current square-spongebob reference-match runs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def run_ids(namespace: str, prefix: str) -> list[str]:
    root = METADATA_DIR / namespace / "speakers" / "runs"
    return sorted(path.name for path in root.glob(f"{prefix}*") if path.is_dir())


def episode_id_from_run_id(run_id: str) -> str:
    if "square-spongebob-episode-" in run_id:
        return run_id.split("reference-match-")[-1]
    return run_id.removeprefix("reference-match-")


def main() -> int:
    args = parse_args()
    runs = run_ids(args.namespace, args.run_prefix)
    if not runs:
        sys.exit("No matching run ids found.")

    for run_id in runs:
        episode_id = episode_id_from_run_id(run_id)
        command = [
            "python3",
            str(SCRIPT_PATH),
            "--namespace",
            args.namespace,
            "--run-id",
            run_id,
            "--episode-id",
            episode_id,
        ]
        print(" ".join(command))
        if args.dry_run:
            continue
        code = subprocess.run(command, check=False).returncode
        if code != 0:
            return code

    print(f"processed {len(runs)} matched runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
