#!/usr/bin/env python3
"""Register a speaker reference in references.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"
FIELDNAMES = ["reference_id", "speaker", "namespace", "status", "kind", "file_path", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a speaker reference in references.csv.")
    parser.add_argument("--namespace", "--series", dest="namespace", required=True, help="Namespace slug.")
    parser.add_argument("--speaker", required=True, help="Speaker identifier.")
    parser.add_argument("--reference-id", help="Stable reference ID. Auto-generated if omitted.")
    parser.add_argument("--file-path", required=True, help="Reference clip path, relative or absolute.")
    parser.add_argument("--status", default="approved", choices=["approved", "pending"], help="Reference status.")
    parser.add_argument("--kind", default="seed", help="Reference kind. Default: seed.")
    parser.add_argument("--notes", default="", help="Optional note.")
    return parser.parse_args()


def catalog_path(namespace: str, speaker: str) -> Path:
    return METADATA_DIR / namespace / "speakers" / "references" / speaker / "references.csv"


def normalize_file_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path.relative_to(REPO_ROOT))
    return str(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def next_reference_id(namespace: str, speaker: str, rows: list[dict[str, str]]) -> str:
    numbers = []
    prefix = f"{namespace}-{speaker}-ref-"
    for row in rows:
        reference_id = row.get("reference_id", "")
        if reference_id.startswith(prefix):
            suffix = reference_id.removeprefix(prefix)
            if suffix.isdigit():
                numbers.append(int(suffix))
    next_number = max(numbers) + 1 if numbers else 1
    return f"{namespace}-{speaker}-ref-{next_number:03d}"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    file_path = normalize_file_path(args.file_path)
    catalog = catalog_path(args.namespace, args.speaker)
    rows = read_rows(catalog)

    if any(row["file_path"] == file_path for row in rows):
        sys.exit(f"Reference already registered for file_path: {file_path}")

    reference_id = args.reference_id or next_reference_id(args.namespace, args.speaker, rows)
    if any(row["reference_id"] == reference_id for row in rows):
        sys.exit(f"Reference ID already exists: {reference_id}")

    rows.append(
        {
            "reference_id": reference_id,
            "speaker": args.speaker,
            "namespace": args.namespace,
            "status": args.status,
            "kind": args.kind,
            "file_path": file_path,
            "notes": args.notes,
        }
    )
    write_rows(catalog, rows)
    print(f"registered {reference_id} -> {file_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
