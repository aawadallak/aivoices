#!/usr/bin/env python3
"""Shared helpers for XTTS training and promotion flows."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "training" / "xtts"
SHARED_MODEL_ROOT = TRAINING_ROOT / "_shared" / "XTTS_v2_original_model_files"
DEFAULT_SMOKE_TEST_FILE = REPO_ROOT / "metadata" / "tts" / "smoke-tests" / "pt-br-default-v1.txt"


@dataclass(frozen=True)
class DatasetSummary:
    dataset_dir: Path
    dataset_name: str
    voice: str
    language: str
    train_rows: int
    eval_rows: int
    speakers: tuple[str, ...]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_pipe_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="|"))


def ensure_required_columns(path: Path, rows: list[dict[str, str]], required: Iterable[str]) -> None:
    if not rows:
        return
    available = set(rows[0].keys())
    missing = [column for column in required if column not in available]
    if missing:
        raise SystemExit(f"{path} missing required columns: {', '.join(missing)}")


def validate_xtts_dataset(dataset_dir: Path, language: str = "pt-BR") -> DatasetSummary:
    dataset_dir = dataset_dir.resolve()
    train_csv = dataset_dir / "metadata_train.csv"
    eval_csv = dataset_dir / "metadata_eval.csv"
    wavs_dir = dataset_dir / "wavs"

    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")
    if not train_csv.is_file():
        raise SystemExit(f"Dataset missing metadata_train.csv: {train_csv}")
    if not eval_csv.is_file():
        raise SystemExit(f"Dataset missing metadata_eval.csv: {eval_csv}")
    if not wavs_dir.is_dir():
        raise SystemExit(f"Dataset missing wavs directory: {wavs_dir}")

    train_rows = read_pipe_csv(train_csv)
    eval_rows = read_pipe_csv(eval_csv)
    ensure_required_columns(train_csv, train_rows, ("audio_file", "text", "speaker_name"))
    ensure_required_columns(eval_csv, eval_rows or train_rows, ("audio_file", "text", "speaker_name"))

    valid_train_rows = 0
    speakers: set[str] = set()
    missing_audio: list[str] = []

    for row in train_rows:
        audio_file = row.get("audio_file", "").strip()
        text = row.get("text", "").strip()
        speaker_name = row.get("speaker_name", "").strip()
        if not audio_file or not text or not speaker_name:
            continue
        audio_path = dataset_dir / audio_file
        if not audio_path.is_file():
            missing_audio.append(audio_file)
            continue
        speakers.add(speaker_name)
        valid_train_rows += 1

    if missing_audio:
        preview = ", ".join(missing_audio[:5])
        raise SystemExit(f"Dataset references missing audio files under {dataset_dir}: {preview}")
    if valid_train_rows == 0:
        raise SystemExit(f"Dataset has no valid train rows: {train_csv}")

    if len(speakers) == 0:
        raise SystemExit(f"Dataset has no speaker_name values: {train_csv}")
    if len(speakers) > 1:
        print(f"[warn] dataset contains multiple speakers: {', '.join(sorted(speakers))}")
    if len(eval_rows) == 0:
        print(f"[warn] eval CSV is empty: {eval_csv}")

    voice = sorted(speakers)[0]
    return DatasetSummary(
        dataset_dir=dataset_dir,
        dataset_name=dataset_dir.name,
        voice=voice,
        language=language,
        train_rows=valid_train_rows,
        eval_rows=len(eval_rows),
        speakers=tuple(sorted(speakers)),
    )


def make_run_id(voice: str, dataset_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slugify(voice)}-{slugify(dataset_name)}-{timestamp}"


def run_root(namespace: str, voice: str, run_id: str) -> Path:
    return TRAINING_ROOT / namespace / voice / run_id


def run_manifest_path(run_dir: Path) -> Path:
    return run_dir / "run.json"


def candidates_manifest_path(run_dir: Path) -> Path:
    return run_dir / "candidates.json"


def promotion_manifest_path(run_dir: Path) -> Path:
    return run_dir / "promotion.json"


def current_model_manifest_path(namespace: str, voice: str) -> Path:
    return TRAINING_ROOT / "models" / namespace / voice / "current.json"


def trainer_dir(run_dir: Path) -> Path:
    return run_dir / "trainer"


def trainer_config_path(run_dir: Path) -> Path:
    return trainer_dir(run_dir) / "config.json"


def smoke_samples_dir(run_dir: Path) -> Path:
    return run_dir / "samples"


def artifacts_dir(run_dir: Path) -> Path:
    return run_dir / "artifacts"


def artifact_dir(run_dir: Path, kind: str) -> Path:
    return artifacts_dir(run_dir) / kind


def checkpoint_sort_key(path: Path) -> tuple[int, float]:
    match = re.search(r"checkpoint_(\d+)\.pth$", path.name)
    step = int(match.group(1)) if match else -1
    return (step, path.stat().st_mtime)


def list_checkpoint_files(training_dir: Path) -> list[Path]:
    candidates = list(training_dir.glob("checkpoint_*.pth"))
    if not candidates:
        candidates = list(training_dir.glob("*/checkpoint_*.pth"))
    return sorted(candidates, key=checkpoint_sort_key)


def best_model_path(training_dir: Path) -> Path | None:
    path = training_dir / "best_model.pth"
    if path.is_file():
        return path
    matches = list(training_dir.glob("*/best_model.pth"))
    return matches[0] if matches else None


def latest_checkpoint_path(training_dir: Path) -> Path | None:
    checkpoints = list_checkpoint_files(training_dir)
    return checkpoints[-1] if checkpoints else None


def load_smoke_test_lines(path: Path | None = None) -> list[str]:
    source = path or DEFAULT_SMOKE_TEST_FILE
    if not source.is_file():
        raise SystemExit(f"Smoke test file not found: {source}")
    return [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
