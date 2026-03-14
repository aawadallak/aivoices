#!/usr/bin/env python3
"""Register raw source files for a namespace into sources.csv."""

from __future__ import annotations

import argparse
import csv
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
FIELDNAMES = [
    "source_id",
    "namespace",
    "episode_id",
    "file_path",
    "language",
    "status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register raw source files for a namespace into metadata/<namespace>/manifests/sources.csv."
    )
    parser.add_argument(
        "--namespace",
        required=True,
        help="Namespace slug under episodes/, for example 'dragonball'.",
    )
    parser.add_argument(
        "--language",
        default="",
        help="Default language to use for new rows when no language exists yet.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resulting CSV rows without writing the file.",
    )
    return parser.parse_args()


def manifest_path_for(namespace: str) -> Path:
    manifest_dir = METADATA_DIR / namespace / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / "sources.csv"


def relative_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def load_existing_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            normalized = {field: (row.get(field, "") or "") for field in FIELDNAMES}

            # Backward compatibility with the earlier "series" column.
            if not normalized["namespace"]:
                normalized["namespace"] = row.get("series", "") or ""

            rows.append(normalized)
    return rows


def source_files_for(namespace: str) -> list[Path]:
    raw_dir = EPISODES_DIR / namespace / "raw"
    if not raw_dir.is_dir():
        sys.exit(f"Namespace raw directory not found: {raw_dir}")

    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def detect_status(namespace: str, source_path: Path) -> str:
    output_path = EPISODES_DIR / namespace / "extracted-audio" / f"{source_path.stem}.wav"
    return "extracted" if output_path.exists() else "available"


def merge_rows(
    namespace: str,
    source_files: list[Path],
    existing_rows: list[dict[str, str]],
    default_language: str,
) -> list[dict[str, str]]:
    existing_by_path = {row["file_path"]: row for row in existing_rows if row["file_path"]}
    merged: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    for source_path in source_files:
        rel_path = relative_path(source_path)
        seen_paths.add(rel_path)
        existing = existing_by_path.get(rel_path, {})
        episode_id = existing.get("episode_id") or source_path.stem
        source_id = existing.get("source_id") or source_path.stem
        merged.append(
            {
                "source_id": source_id,
                "namespace": namespace,
                "episode_id": episode_id,
                "file_path": rel_path,
                "language": existing.get("language") or default_language,
                "status": detect_status(namespace, source_path),
                "notes": existing.get("notes", ""),
            }
        )

    for row in existing_rows:
        file_path = row.get("file_path", "")
        if not file_path or file_path in seen_paths:
            continue

        status = row.get("status", "")
        if not status or status == "available" or status == "extracted":
            status = "missing"

        merged.append(
            {
                "source_id": row.get("source_id", ""),
                "namespace": row.get("namespace", "") or namespace,
                "episode_id": row.get("episode_id", ""),
                "file_path": file_path,
                "language": row.get("language", ""),
                "status": status,
                "notes": row.get("notes", ""),
            }
        )

    return sorted(merged, key=lambda row: (row["namespace"], row["episode_id"], row["file_path"]))


def write_rows(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    args = parse_args()
    manifest_path = manifest_path_for(args.namespace)
    existing_rows = load_existing_rows(manifest_path)
    rows = merge_rows(
        args.namespace,
        source_files_for(args.namespace),
        existing_rows,
        args.language,
    )

    if args.dry_run:
        print_rows(rows)
    else:
        write_rows(manifest_path, rows)
        print(f"registered {len(rows)} sources in {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
