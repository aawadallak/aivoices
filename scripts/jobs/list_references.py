#!/usr/bin/env python3
"""List registered speaker references."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List registered speaker references.")
    parser.add_argument("--namespace", "--series", dest="namespace", required=True, help="Namespace slug.")
    parser.add_argument("--speaker", required=True, help="Speaker identifier.")
    parser.add_argument("--status", choices=["approved", "pending"], help="Optional status filter.")
    parser.add_argument("--kind", help="Optional kind filter.")
    parser.add_argument("--reference-id", help="Optional exact reference_id filter.")
    parser.add_argument("--json", action="store_true", help="Print as JSON.")
    return parser.parse_args()


def catalog_path(namespace: str, speaker: str) -> Path:
    return METADATA_DIR / namespace / "speakers" / "references" / speaker / "references.csv"


def main() -> int:
    args = parse_args()
    catalog = catalog_path(args.namespace, args.speaker)
    if not catalog.exists():
        sys.exit(f"Catalog not found: {catalog}")

    with catalog.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if args.status:
        rows = [row for row in rows if row["status"] == args.status]
    if args.kind:
        rows = [row for row in rows if row["kind"] == args.kind]
    if args.reference_id:
        rows = [row for row in rows if row["reference_id"] == args.reference_id]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("no references found")
        return 0

    for row in rows:
        print(f"{row['reference_id']}\t{row['status']}\t{row['kind']}\t{row['file_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
