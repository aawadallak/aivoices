#!/usr/bin/env python3
"""Export a provisional dataset from quality-scored matched clips."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"
DATASETS_DIR = REPO_ROOT / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a provisional dataset from clip-quality.csv files.")
    parser.add_argument("--namespace", "--series", dest="namespace", required=True, help="Namespace slug.")
    parser.add_argument("--run-prefix", default="reference-match-square-spongebob-episode-", help="Run id prefix.")
    parser.add_argument(
        "--allowed-quality",
        default="clean,usable",
        help="Comma-separated accepted labels. Default: clean,usable.",
    )
    parser.add_argument(
        "--speakers",
        default="bob-esponja,lula-molusco",
        help="Comma-separated speakers to export. Default: bob-esponja,lula-molusco.",
    )
    parser.add_argument(
        "--export-name",
        default="provisional-match-v1",
        help="Output subdir under datasets/<namespace>/exports/. Default: provisional-match-v1.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned copies without executing them.")
    return parser.parse_args()


def score_files(namespace: str, prefix: str) -> list[Path]:
    root = METADATA_DIR / namespace / "speakers" / "runs"
    return sorted(root.glob(f"{prefix}*/matched-clips/*/clip-quality.csv"))


def main() -> int:
    args = parse_args()
    allowed_quality = {item.strip() for item in args.allowed_quality.split(",") if item.strip()}
    allowed_speakers = {item.strip() for item in args.speakers.split(",") if item.strip()}
    export_root = DATASETS_DIR / args.namespace / "exports" / args.export_name
    manifest_rows: list[dict[str, str]] = []

    for score_path in score_files(args.namespace, args.run_prefix):
        episode_id = score_path.parent.name
        run_id = score_path.parent.parent.parent.name
        with score_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["matched_speaker"] not in allowed_speakers:
                    continue
                if row["quality_label"] not in allowed_quality:
                    continue
                source_path = REPO_ROOT / row["audio_path"]
                speaker_dir = export_root / "clips" / row["matched_speaker"]
                speaker_dir.mkdir(parents=True, exist_ok=True)
                target_name = f"{episode_id}-{row['matched_speaker']}-clip-{int(row['clip_index']):04d}.wav"
                target_path = speaker_dir / target_name
                print(f"{source_path} -> {target_path}")
                if not args.dry_run:
                    shutil.copy2(source_path, target_path)
                manifest_rows.append(
                    {
                        "episode_id": episode_id,
                        "speaker": row["matched_speaker"],
                        "run_id": run_id,
                        "quality_label": row["quality_label"],
                        "keep_for_reference": row["keep_for_reference"],
                        "audio_path": str(target_path.relative_to(REPO_ROOT)),
                    }
                )

    manifest_path = export_root / "provisional-dataset.csv"
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["episode_id", "speaker", "run_id", "quality_label", "keep_for_reference", "audio_path"],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
    print(f"prepared {len(manifest_rows)} provisional clips under {export_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
