#!/usr/bin/env python3
"""Validate an existing reference clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from make_reference_clip import ensure_tools, validate_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an existing reference clip.")
    parser.add_argument("--input", required=True, help="Reference clip path.")
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
    parser.add_argument("--json", action="store_true", help="Print the validation report as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_tools()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        sys.exit(f"Input file not found: {input_path}")

    report = validate_clip(input_path, args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"approved: {report['approved']}")
        print(f"duration_sec: {report['duration_sec']}")
        print(f"leading_silence_sec: {report['leading_silence_sec']}")
        print(f"trailing_silence_sec: {report['trailing_silence_sec']}")
        print(f"speech_ratio: {report['speech_ratio']}")
        print(f"checks: {report['checks']}")
    return 0 if report["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
