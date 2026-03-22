#!/usr/bin/env python3
"""Generate hybrid XTTS+RVC smoke-test samples for review.

For each XTTS checkpoint candidate:
  1. Generate raw XTTS audio from smoke-test sentences
  2. Pass each sample through the trained RVC model
  3. Save both raw (xtts/) and converted (rvc/) samples side-by-side

The output follows the same candidates.json pattern as export_xtts_smoke_review.py.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder has been deprecated.*")
warnings.filterwarnings("ignore", message=".*StreamingMediaEncoder has been deprecated.*")
warnings.filterwarnings("ignore", message=".*torchaudio.load_with_torchcodec.*")
warnings.filterwarnings("ignore", message=".*torchaudio.save_with_torchcodec.*")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.pipeline.xtts_common import (
    DEFAULT_SMOKE_TEST_FILE,
    SHARED_MODEL_ROOT,
    best_model_path,
    candidates_manifest_path as xtts_candidates_manifest_path,
    latest_checkpoint_path,
    list_checkpoint_files,
    load_smoke_test_lines,
    read_json,
    run_manifest_path as xtts_run_manifest_path,
    run_root as xtts_run_root,
    smoke_samples_dir as xtts_smoke_samples_dir,
    utc_now_iso,
    write_json,
)
from scripts.pipeline.rvc_common import (
    TRAINING_ROOT as RVC_TRAINING_ROOT,
    model_dir as rvc_model_dir,
    read_json as rvc_read_json,
    run_root as rvc_run_root,
)
from scripts.pipeline.rvc_inference import convert_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hybrid XTTS+RVC smoke-test samples."
    )
    # XTTS run
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--xtts-run-id", required=True, help="XTTS training run ID.")
    parser.add_argument("--speaker-wav", required=True, help="Reference WAV for XTTS conditioning.")

    # RVC model
    parser.add_argument("--rvc-run-id", required=True, help="RVC training run ID.")
    parser.add_argument("--rvc-namespace", help="RVC namespace (defaults to --namespace).")
    parser.add_argument("--rvc-voice", help="RVC voice (defaults to --voice).")

    # RVC inference params
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "crepe"])
    parser.add_argument("--f0-up-key", type=int, default=0, help="Pitch shift in semitones.")
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--rms-mix-rate", type=float, default=0.25)
    parser.add_argument("--protect", type=float, default=0.33)

    # Smoke test
    parser.add_argument("--smoke-test-file", default=str(DEFAULT_SMOKE_TEST_FILE))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-milestone", action="store_true")

    # Upload
    parser.add_argument(
        "--remote-prefix", default="r2:aivoices/training/runs",
        help="Remote run prefix for upload.",
    )
    parser.add_argument("--no-upload", action="store_true")
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
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "text", "xtts_audio", "rvc_audio"],
            delimiter="|",
        )
        writer.writeheader()
        writer.writerows(rows)


def generate_xtts_samples(
    *,
    checkpoint_path: Path,
    config_path: Path,
    vocab_path: Path,
    speaker_wav: Path,
    smoke_lines: list[str],
    output_dir: Path,
    device: str,
) -> list[tuple[Path, str]]:
    """Generate raw XTTS samples. Returns list of (wav_path, text)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torchaudio
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(checkpoint_path),
        vocab_path=str(vocab_path),
        use_deepspeed=False,
    )
    if device == "cuda" and torch.cuda.is_available():
        model.cuda()
    else:
        device = "cpu"

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[str(speaker_wav)]
    )

    results: list[tuple[Path, str]] = []
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
        results.append((wav_path, text))

    # Free GPU memory before RVC inference
    del model, gpt_cond_latent, speaker_embedding
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def convert_samples_rvc(
    *,
    xtts_samples: list[tuple[Path, str]],
    rvc_model_path: Path,
    rvc_index_path: Path | None,
    output_dir: Path,
    f0_method: str,
    f0_up_key: int,
    index_rate: float,
    filter_radius: int,
    rms_mix_rate: float,
    protect: float,
) -> list[Path]:
    """Pass each XTTS sample through RVC. Returns list of converted wav paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    converted: list[Path] = []

    for xtts_wav, _text in xtts_samples:
        rvc_wav = output_dir / xtts_wav.name
        print(f"  [rvc] converting {xtts_wav.name}")
        convert_audio(
            model_path=rvc_model_path,
            index_path=rvc_index_path,
            input_path=xtts_wav,
            output_path=rvc_wav,
            f0_method=f0_method,
            f0_up_key=f0_up_key,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )
        converted.append(rvc_wav)

    return converted


def resolve_rvc_model(rvc_run_dir: Path) -> tuple[Path, Path | None]:
    """Find the RVC generator .pth and FAISS index in a run's model/ dir."""
    mdir = rvc_model_dir(rvc_run_dir)
    if not mdir.is_dir():
        raise SystemExit(f"RVC model directory not found: {mdir}")

    generators = sorted(mdir.glob("G_*.pth"))
    if not generators:
        raise SystemExit(f"No RVC generator checkpoint found in {mdir}")
    model_path = generators[-1]  # latest epoch

    index_files = list(mdir.glob("added_*.index"))
    index_path = index_files[0] if index_files else None

    return model_path, index_path


def maybe_upload_review(
    run_dir: Path,
    remote_prefix: str,
    namespace: str,
    voice: str,
    run_id: str,
    dry_run: bool,
) -> None:
    if shutil.which("rclone") is None:
        raise SystemExit("rclone not found in PATH")
    # Upload under a hybrid-specific prefix
    destination = f"{remote_prefix.rstrip('/')}/{namespace}/{voice}/{run_id}/hybrid"
    commands = [
        ["rclone", "copyto", str(run_dir / "hybrid-candidates.json"), f"{destination}/hybrid-candidates.json", "-P"],
        ["rclone", "copy", str(run_dir / "hybrid-samples"), f"{destination}/hybrid-samples", "-P", "--create-empty-src-dirs"],
    ]
    for cmd in commands:
        if dry_run:
            cmd.append("--dry-run")
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    # --- Resolve XTTS run ---
    xtts_run_dir = xtts_run_root(args.namespace, args.voice, args.xtts_run_id)
    if not xtts_run_dir.is_dir():
        raise SystemExit(f"XTTS run directory not found: {xtts_run_dir}")
    xtts_manifest = read_json(xtts_run_manifest_path(xtts_run_dir))

    training_dir = xtts_run_dir / "trainer"
    config_path = training_dir / "config.json"
    if not config_path.is_file():
        nested = list(training_dir.glob("*/config.json"))
        if nested:
            config_path = nested[0]
        else:
            raise SystemExit(f"Trainer config not found: {config_path}")

    vocab_path = SHARED_MODEL_ROOT / "vocab.json"
    if not vocab_path.is_file():
        raise SystemExit(f"Shared vocab.json not found: {vocab_path}")

    speaker_wav = Path(args.speaker_wav).expanduser().resolve()
    if not speaker_wav.is_file():
        raise SystemExit(f"Speaker WAV not found: {speaker_wav}")

    smoke_lines = load_smoke_test_lines(Path(args.smoke_test_file))

    # --- Resolve RVC model ---
    rvc_namespace = args.rvc_namespace or args.namespace
    rvc_voice = args.rvc_voice or args.voice
    rvc_run_dir = rvc_run_root(rvc_namespace, rvc_voice, args.rvc_run_id)
    if not rvc_run_dir.is_dir():
        raise SystemExit(f"RVC run directory not found: {rvc_run_dir}")

    rvc_model_path, rvc_index_path = resolve_rvc_model(rvc_run_dir)
    print(f"[hybrid] XTTS run: {xtts_run_dir.name}")
    print(f"[hybrid] RVC model: {rvc_model_path.name}" + (f" + {rvc_index_path.name}" if rvc_index_path else ""))

    # --- Select XTTS checkpoint candidates ---
    selected = select_candidates(training_dir, args.include_milestone)
    if not selected:
        raise SystemExit(f"No candidate checkpoints found under {training_dir}")

    # --- Generate samples ---
    base_samples_dir = xtts_run_dir / "hybrid-samples"
    candidates_payload = {
        "run_id": xtts_manifest["run_id"],
        "rvc_run_id": args.rvc_run_id,
        "rvc_model": rvc_model_path.name,
        "rvc_index": rvc_index_path.name if rvc_index_path else None,
        "rvc_params": {
            "index_rate": args.index_rate,
            "f0_method": args.f0_method,
            "f0_up_key": args.f0_up_key,
            "filter_radius": args.filter_radius,
            "rms_mix_rate": args.rms_mix_rate,
            "protect": args.protect,
        },
        "generated_at": utc_now_iso(),
        "candidates": [],
    }

    for candidate in selected:
        checkpoint_id = candidate["checkpoint_id"]
        candidate_dir = base_samples_dir / checkpoint_id
        xtts_dir = candidate_dir / "xtts"
        rvc_dir = candidate_dir / "rvc"

        print(f"\n[hybrid] checkpoint: {checkpoint_id} (kind={candidate['kind']})")

        # Step 1: Generate XTTS samples
        print(f"  [xtts] generating {len(smoke_lines)} samples...")
        xtts_samples = generate_xtts_samples(
            checkpoint_path=candidate["path"],
            config_path=config_path,
            vocab_path=vocab_path,
            speaker_wav=speaker_wav,
            smoke_lines=smoke_lines,
            output_dir=xtts_dir,
            device=args.device,
        )

        # Step 2: Convert through RVC
        print(f"  [rvc] converting {len(xtts_samples)} samples...")
        rvc_outputs = convert_samples_rvc(
            xtts_samples=xtts_samples,
            rvc_model_path=rvc_model_path,
            rvc_index_path=rvc_index_path,
            output_dir=rvc_dir,
            f0_method=args.f0_method,
            f0_up_key=args.f0_up_key,
            index_rate=args.index_rate,
            filter_radius=args.filter_radius,
            rms_mix_rate=args.rms_mix_rate,
            protect=args.protect,
        )

        # Write sample manifest for this checkpoint
        manifest_rows: list[dict[str, str]] = []
        for (xtts_wav, text), rvc_wav in zip(xtts_samples, rvc_outputs):
            sample_id = xtts_wav.stem
            manifest_rows.append({
                "sample_id": sample_id,
                "text": text,
                "xtts_audio": f"xtts/{xtts_wav.name}",
                "rvc_audio": f"rvc/{rvc_wav.name}",
            })

        manifest_path = candidate_dir / "sample-manifest.csv"
        write_sample_manifest(manifest_path, manifest_rows)

        candidates_payload["candidates"].append({
            "checkpoint_id": checkpoint_id,
            "step": candidate["step"],
            "kind": candidate["kind"],
            "sample_dir": str(candidate_dir.relative_to(xtts_run_dir)),
            "sample_manifest": str(manifest_path.relative_to(xtts_run_dir)),
            "xtts_sample_dir": str(xtts_dir.relative_to(xtts_run_dir)),
            "rvc_sample_dir": str(rvc_dir.relative_to(xtts_run_dir)),
            "metrics": {},
            "status": "pending",
        })

    # Write hybrid candidates manifest
    hybrid_candidates_path = xtts_run_dir / "hybrid-candidates.json"
    write_json(hybrid_candidates_path, candidates_payload)
    print(f"\nwrote hybrid candidates manifest to {hybrid_candidates_path}")

    # Upload
    if args.remote_prefix and not args.no_upload:
        maybe_upload_review(
            xtts_run_dir,
            args.remote_prefix,
            args.namespace,
            args.voice,
            args.xtts_run_id,
            args.dry_run_upload,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
