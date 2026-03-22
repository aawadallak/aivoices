#!/usr/bin/env python3
"""Export a speaker-specific dataset in the directory layout expected by RVC v2 training.

Unlike XTTS exports, RVC only needs a directory of WAV files — no transcripts
or metadata CSVs. This job collects curated clips from a provisional export,
filters by speaker and duration, and copies them into an RVC-ready layout.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"
DATASETS_DIR = REPO_ROOT / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RVC-ready dataset (wavs only, no transcripts).")
    parser.add_argument("--namespace", "--series", dest="namespace", required=True, help="Namespace slug.")
    parser.add_argument(
        "--source-export",
        required=True,
        help="Source provisional export under datasets/<namespace>/exports/ containing provisional-dataset.csv.",
    )
    parser.add_argument("--speaker", required=True, help="Single speaker to export.")
    parser.add_argument(
        "--output-name",
        required=True,
        help="Output folder under datasets/<namespace>/exports/export_rvc/.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum clip duration in seconds. Default: 15.0.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum clip duration in seconds. Default: 1.0.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def source_manifest_path(namespace: str, export_name: str) -> Path:
    return DATASETS_DIR / namespace / "exports" / export_name / "provisional-dataset.csv"


def matched_manifest_path(namespace: str, run_id: str, episode_id: str) -> Path:
    return (
        METADATA_DIR
        / namespace
        / "speakers"
        / "runs"
        / run_id
        / "matched-clips"
        / episode_id
        / "matched-clips.csv"
    )


def rvc_output_root(namespace: str, output_name: str) -> Path:
    return DATASETS_DIR / namespace / "exports" / "export_rvc" / output_name


def write_dataset_info(
    path: Path,
    *,
    namespace: str,
    speaker: str,
    dataset_name: str,
    source_export: str,
    wav_count: int,
) -> None:
    import json
    payload = {
        "namespace": namespace,
        "voice": speaker,
        "dataset_name": dataset_name,
        "source_export": source_export,
        "format": "rvc-v2",
        "wav_count": wav_count,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_manifest = source_manifest_path(args.namespace, args.source_export)
    if not source_manifest.is_file():
        raise SystemExit(f"Source provisional manifest not found: {source_manifest}")

    output_root = rvc_output_root(args.namespace, args.output_name)
    wavs_root = output_root / "wavs"
    wavs_root.mkdir(parents=True, exist_ok=True)

    counter = 0
    skipped_duration = 0

    with source_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["speaker"] != args.speaker:
                continue

            episode_id = row["episode_id"]
            run_id = row["run_id"]

            # Look up duration from matched-clips manifest
            matched_path = matched_manifest_path(args.namespace, run_id, episode_id)
            if not matched_path.is_file():
                continue

            source_audio = resolve_repo_path(row["audio_path"])
            stem = source_audio.stem
            clip_index = stem.rsplit("-", 1)[-1].replace("clip-", "")

            duration = None
            with matched_path.open(newline="", encoding="utf-8") as mf:
                for item in csv.DictReader(mf):
                    if item["matched_speaker"] == args.speaker and item["clip_index"] == str(int(clip_index)):
                        duration = float(item["duration_sec"])
                        break

            if duration is None:
                continue
            if duration < args.min_duration or duration > args.max_duration:
                skipped_duration += 1
                continue
            if not source_audio.is_file():
                continue

            counter += 1
            target_name = f"{args.speaker}-{counter:04d}.wav"
            shutil.copy2(source_audio, wavs_root / target_name)

    if counter == 0:
        raise SystemExit("No clips matched the RVC export filters.")

    write_dataset_info(
        output_root / "dataset-info.json",
        namespace=args.namespace,
        speaker=args.speaker,
        dataset_name=args.output_name,
        source_export=args.source_export,
        wav_count=counter,
    )

    print(f"wrote RVC export to {output_root}")
    print(f"speaker={args.speaker} wavs={counter} skipped_duration={skipped_duration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
