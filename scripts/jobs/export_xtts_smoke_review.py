#!/usr/bin/env python3
"""Generate XTTS smoke-test samples and candidate manifests for review."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.pipeline.xtts_common import (
    DEFAULT_SMOKE_TEST_FILE,
    SHARED_MODEL_ROOT,
    best_model_path,
    candidates_manifest_path,
    latest_checkpoint_path,
    list_checkpoint_files,
    load_smoke_test_lines,
    read_json,
    run_manifest_path,
    run_root,
    smoke_samples_dir,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate XTTS smoke-test review artifacts.")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--speaker-wav", required=True, help="Reference WAV for XTTS inference.")
    parser.add_argument("--smoke-test-file", default=str(DEFAULT_SMOKE_TEST_FILE))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-milestone", action="store_true")
    parser.add_argument("--remote-prefix", help="Optional remote run prefix, e.g. r2:aivoices/training/runs.")
    parser.add_argument("--dry-run-upload", action="store_true")
    return parser.parse_args()


def load_checkpoint_step(path: Path) -> int:
    try:
        import torch
    except ImportError:
        return -1
    try:
        state = torch.load(path, map_location="cpu")
    except Exception:
        return -1
    return int(state.get("step", -1))


def select_candidates(training_dir: Path, include_milestone: bool) -> list[dict]:
    candidates: list[dict] = []
    last_path = latest_checkpoint_path(training_dir)
    best_path = best_model_path(training_dir)
    seen: set[str] = set()

    if last_path:
        step = load_checkpoint_step(last_path)
        candidates.append({"checkpoint_id": last_path.stem, "step": step, "kind": "last", "path": last_path})
        seen.add(str(last_path.resolve()))

    if best_path and str(best_path.resolve()) not in seen:
        step = load_checkpoint_step(best_path)
        candidates.append({"checkpoint_id": best_path.stem, "step": step, "kind": "best_loss", "path": best_path})
        seen.add(str(best_path.resolve()))

    if include_milestone:
        checkpoints = list_checkpoint_files(training_dir)
        if len(checkpoints) >= 3:
            candidate = checkpoints[len(checkpoints) // 2]
            if str(candidate.resolve()) not in seen:
                step = load_checkpoint_step(candidate)
                candidates.append({"checkpoint_id": candidate.stem, "step": step, "kind": "milestone", "path": candidate})

    return candidates[:3]


def write_sample_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "text", "audio_file"], delimiter="|")
        writer.writeheader()
        writer.writerows(rows)


def generate_samples(
    *,
    checkpoint_path: Path,
    config_path: Path,
    vocab_path: Path,
    speaker_wav: Path,
    smoke_lines: list[str],
    output_dir: Path,
    device: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torchaudio
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
    except ImportError as exc:
        raise SystemExit("XTTS inference dependencies are missing in the current runtime.") from exc

    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(checkpoint_path), vocab_path=str(vocab_path), use_deepspeed=False)
    if device == "cuda" and torch.cuda.is_available():
        model.cuda()
    else:
        device = "cpu"

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(speaker_wav)])

    rows: list[dict[str, str]] = []
    for index, text in enumerate(smoke_lines, start=1):
        result = model.inference(
            text=text,
            language="pt",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )
        wav_path = output_dir / f"sample-{index:02d}.wav"
        audio = torch.tensor(result["wav"]).unsqueeze(0)
        torchaudio.save(str(wav_path), audio, 24000)
        rows.append({"sample_id": f"sample-{index:02d}", "text": text, "audio_file": wav_path.name})

    manifest_path = output_dir / "sample-manifest.csv"
    write_sample_manifest(manifest_path, rows)
    return manifest_path


def maybe_upload_review(run_dir: Path, remote_prefix: str, namespace: str, voice: str, run_id: str, dry_run: bool) -> None:
    if shutil.which("rclone") is None:
        raise SystemExit("rclone not found in PATH")
    destination = f"{remote_prefix.rstrip('/')}/{namespace}/{voice}/{run_id}"
    commands = [
        ["rclone", "copyto", str(run_manifest_path(run_dir)), f"{destination}/run.json", "-P"],
        ["rclone", "copyto", str(candidates_manifest_path(run_dir)), f"{destination}/candidates.json", "-P"],
        ["rclone", "copy", str(smoke_samples_dir(run_dir)), f"{destination}/samples", "-P", "--create-empty-src-dirs"],
    ]
    for cmd in commands:
        if dry_run:
            cmd.append("--dry-run")
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    run_dir = run_root(args.namespace, args.voice, args.run_id)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    run_manifest = read_json(run_manifest_path(run_dir))
    training_dir = run_dir / "trainer"
    config_path = training_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Trainer config not found: {config_path}")
    vocab_path = SHARED_MODEL_ROOT / "vocab.json"
    if not vocab_path.is_file():
        raise SystemExit(f"Shared vocab.json not found: {vocab_path}")

    smoke_lines = load_smoke_test_lines(Path(args.smoke_test_file))
    speaker_wav = Path(args.speaker_wav).expanduser().resolve()
    if not speaker_wav.is_file():
        raise SystemExit(f"Speaker WAV not found: {speaker_wav}")

    selected = select_candidates(training_dir, args.include_milestone)
    if not selected:
        raise SystemExit(f"No candidate checkpoints found under {training_dir}")

    candidates_payload = {
        "run_id": run_manifest["run_id"],
        "generated_at": utc_now_iso(),
        "candidates": [],
    }

    base_samples_dir = smoke_samples_dir(run_dir)
    for candidate in selected:
        sample_dir = base_samples_dir / candidate["checkpoint_id"]
        sample_manifest = generate_samples(
            checkpoint_path=candidate["path"],
            config_path=config_path,
            vocab_path=vocab_path,
            speaker_wav=speaker_wav,
            smoke_lines=smoke_lines,
            output_dir=sample_dir,
            device=args.device,
        )
        candidates_payload["candidates"].append(
            {
                "checkpoint_id": candidate["checkpoint_id"],
                "step": candidate["step"],
                "kind": candidate["kind"],
                "sample_dir": str(sample_dir.relative_to(run_dir)),
                "sample_manifest": str(sample_manifest.relative_to(run_dir)),
                "metrics": {},
                "status": "pending",
            }
        )

    write_json(candidates_manifest_path(run_dir), candidates_payload)
    print(f"wrote candidates manifest to {candidates_manifest_path(run_dir)}")

    if args.remote_prefix:
        maybe_upload_review(run_dir, args.remote_prefix, args.namespace, args.voice, args.run_id, args.dry_run_upload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
