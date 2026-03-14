#!/usr/bin/env python3
"""Export a speaker-specific dataset in the CSV layout expected by XTTS pipelines."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"
DATASETS_DIR = REPO_ROOT / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export XTTS-ready metadata_train.csv and metadata_eval.csv.")
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
        help="Output folder under datasets/<namespace>/exports/export_xtts/.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Evaluation split ratio. Default: 0.1.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=11.0,
        help="Maximum clip duration in seconds. Default: 11.0.",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=200,
        help="Maximum text length. Default: 200.",
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


def load_text_lookup(namespace: str, run_id: str, episode_id: str) -> dict[tuple[str, str], str]:
    path = matched_manifest_path(namespace, run_id, episode_id)
    if not path.is_file():
        raise SystemExit(f"Matched clips manifest not found: {path}")
    lookup: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lookup[(row["matched_speaker"], row["clip_index"])] = row["text"].strip()
    return lookup


def xtts_output_root(namespace: str, output_name: str) -> Path:
    return DATASETS_DIR / namespace / "exports" / "export_xtts" / output_name


def write_pipe_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_file", "text", "speaker_name"], delimiter="|")
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_info(
    path: Path,
    *,
    namespace: str,
    speaker: str,
    dataset_name: str,
    source_export: str,
    train_count: int,
    eval_count: int,
) -> None:
    payload = {
        "namespace": namespace,
        "voice": speaker,
        "dataset_name": dataset_name,
        "source_export": source_export,
        "language": "pt-BR",
        "train_count": train_count,
        "eval_count": eval_count,
        "total_count": train_count + eval_count,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    path.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_manifest = source_manifest_path(args.namespace, args.source_export)
    if not source_manifest.is_file():
        raise SystemExit(f"Source provisional manifest not found: {source_manifest}")

    output_root = xtts_output_root(args.namespace, args.output_name)
    wavs_root = output_root / "wavs"
    wavs_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    cached_lookup: dict[tuple[str, str], dict[tuple[str, str], str]] = {}
    counter = 0

    with source_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["speaker"] != args.speaker:
                continue
            episode_id = row["episode_id"]
            run_id = row["run_id"]
            key = (run_id, episode_id)
            if key not in cached_lookup:
                cached_lookup[key] = load_text_lookup(args.namespace, run_id, episode_id)
            source_audio = resolve_repo_path(row["audio_path"])
            stem = source_audio.stem
            clip_index = stem.rsplit("-", 1)[-1].replace("clip-", "")
            text = cached_lookup[key].get((args.speaker, str(int(clip_index))), "").strip()
            if not text:
                continue
            if len(text) > args.max_text_length:
                continue

            manifest_row_duration = None
            matched_manifest = matched_manifest_path(args.namespace, run_id, episode_id)
            with matched_manifest.open(newline="", encoding="utf-8") as mf:
                for item in csv.DictReader(mf):
                    if item["matched_speaker"] == args.speaker and item["clip_index"] == str(int(clip_index)):
                        manifest_row_duration = float(item["duration_sec"])
                        break
            if manifest_row_duration is None or manifest_row_duration > args.max_duration:
                continue

            counter += 1
            target_name = f"{args.speaker}-{counter:04d}.wav"
            target_path = wavs_root / target_name
            shutil.copy2(source_audio, target_path)
            rows.append(
                {
                    "audio_file": f"wavs/{target_name}",
                    "text": text,
                    "speaker_name": args.speaker,
                }
            )

    if not rows:
        raise SystemExit("No rows matched the XTTS export filters.")

    rows.sort(key=lambda item: item["audio_file"])
    eval_count = max(1, math.ceil(len(rows) * args.eval_ratio))
    if eval_count >= len(rows):
        eval_count = max(1, len(rows) // 10)
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]
    if not train_rows:
        raise SystemExit("Not enough rows left for training after eval split.")

    write_pipe_csv(output_root / "metadata_train.csv", train_rows)
    write_pipe_csv(output_root / "metadata_eval.csv", eval_rows)
    write_dataset_info(
        output_root / "dataset-info.json",
        namespace=args.namespace,
        speaker=args.speaker,
        dataset_name=args.output_name,
        source_export=args.source_export,
        train_count=len(train_rows),
        eval_count=len(eval_rows),
    )

    print(f"wrote XTTS export to {output_root}")
    print(f"speaker={args.speaker} train={len(train_rows)} eval={len(eval_rows)} total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
