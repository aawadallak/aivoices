#!/usr/bin/env python3
"""Shared helpers for RVC v2 training flows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "training" / "rvc"
DATASETS_DIR = REPO_ROOT / "datasets"


@dataclass(frozen=True)
class RvcDatasetSummary:
    dataset_dir: Path
    dataset_name: str
    voice: str
    wav_count: int
    total_duration_sec: float


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


def make_run_id(voice: str, dataset_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slugify(voice)}-{slugify(dataset_name)}-{timestamp}"


def run_root(namespace: str, voice: str, run_id: str) -> Path:
    return TRAINING_ROOT / namespace / voice / run_id


def run_manifest_path(run_dir: Path) -> Path:
    return run_dir / "run.json"


def model_dir(run_dir: Path) -> Path:
    return run_dir / "model"


def logs_dir(run_dir: Path) -> Path:
    return run_dir / "logs"


def validate_rvc_dataset(dataset_dir: Path) -> RvcDatasetSummary:
    """Validate an RVC dataset directory (just a wavs/ folder with audio files)."""
    dataset_dir = dataset_dir.resolve()
    wavs_dir = dataset_dir / "wavs"

    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")
    if not wavs_dir.is_dir():
        raise SystemExit(f"Dataset missing wavs directory: {wavs_dir}")

    wav_files = sorted(wavs_dir.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No .wav files found in {wavs_dir}")

    total_duration = 0.0
    try:
        import soundfile as sf
        for wav_path in wav_files:
            info = sf.info(wav_path)
            total_duration += info.duration
    except ImportError:
        pass  # soundfile not available — skip duration calculation

    # Infer voice from first filename (pattern: <voice>-NNNN.wav)
    first_stem = wav_files[0].stem
    match = re.match(r"^(.+)-\d{4}$", first_stem)
    voice = match.group(1) if match else first_stem

    return RvcDatasetSummary(
        dataset_dir=dataset_dir,
        dataset_name=dataset_dir.name,
        voice=voice,
        wav_count=len(wav_files),
        total_duration_sec=round(total_duration, 1),
    )
