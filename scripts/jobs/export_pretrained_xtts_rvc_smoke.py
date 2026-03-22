#!/usr/bin/env python3
"""Generate smoke-test samples using pretrained XTTS-v2 (coqui/XTTS-v2) + RVC.

Uses the stock XTTS-v2 model from HuggingFace to generate pt-BR audio,
then passes each sample through a trained RVC model. No XTTS training run
is required — only an RVC run.

Output layout:
  training/rvc/<ns>/<voice>/<rvc-run-id>/
    smoke-pretrained-xtts/
      candidates.json
      xtts/sample-01.wav
      xtts/sample-02.wav
      rvc/sample-01.wav
      rvc/sample-02.wav
      sample-manifest.csv
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
    load_smoke_test_lines,
    utc_now_iso,
    write_json,
)
from scripts.pipeline.rvc_common import (
    model_dir as rvc_model_dir,
    run_root as rvc_run_root,
)
from scripts.pipeline.rvc_inference import convert_audio


XTTS_HF_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test: pretrained XTTS-v2 (coqui/XTTS-v2) → RVC."
    )
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--voice", required=True)
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
    parser.add_argument("--language", default="pt", help="XTTS language code. Default: pt.")
    parser.add_argument("--device", default="cuda")

    # Upload
    parser.add_argument(
        "--remote-prefix", default="r2:aivoices/training/runs",
        help="Remote prefix for upload.",
    )
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--dry-run-upload", action="store_true")

    return parser.parse_args()


def write_sample_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "text", "xtts_audio", "rvc_audio"],
            delimiter="|",
        )
        writer.writeheader()
        writer.writerows(rows)


def generate_pretrained_xtts_samples(
    *,
    speaker_wav: Path,
    smoke_lines: list[str],
    output_dir: Path,
    language: str,
    device: str,
) -> list[tuple[Path, str]]:
    """Generate samples using the stock XTTS-v2 from HuggingFace (coqui/XTTS-v2)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torchaudio
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.manage import ModelManager

    # Download / locate the pretrained XTTS-v2 model
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model(XTTS_HF_MODEL)
    model_dir = Path(model_path).parent

    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(model_dir), use_deepspeed=False)

    if device == "cuda" and torch.cuda.is_available():
        model.cuda()
    else:
        device = "cpu"

    print(f"  [xtts] loaded pretrained XTTS-v2 on {device}")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[str(speaker_wav)]
    )

    results: list[tuple[Path, str]] = []
    for index, text in enumerate(smoke_lines, start=1):
        result = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )
        wav_path = output_dir / f"sample-{index:02d}.wav"
        audio = torch.tensor(result["wav"]).unsqueeze(0)
        torchaudio.save(str(wav_path), audio, 24000)
        results.append((wav_path, text))
        print(f"  [xtts] sample-{index:02d}.wav ({len(text)} chars)")

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
    """Pass each XTTS sample through RVC."""
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
    model_path = generators[-1]

    index_files = list(mdir.glob("added_*.index"))
    index_path = index_files[0] if index_files else None

    return model_path, index_path


def maybe_upload_review(
    output_dir: Path,
    candidates_path: Path,
    remote_prefix: str,
    namespace: str,
    voice: str,
    rvc_run_id: str,
    dry_run: bool,
) -> None:
    if shutil.which("rclone") is None:
        raise SystemExit("rclone not found in PATH")
    destination = f"{remote_prefix.rstrip('/')}/{namespace}/{voice}/{rvc_run_id}/smoke-pretrained-xtts"
    commands = [
        ["rclone", "copyto", str(candidates_path), f"{destination}/candidates.json", "-P"],
        ["rclone", "copy", str(output_dir / "xtts"), f"{destination}/xtts", "-P", "--create-empty-src-dirs"],
        ["rclone", "copy", str(output_dir / "rvc"), f"{destination}/rvc", "-P", "--create-empty-src-dirs"],
        ["rclone", "copyto", str(output_dir / "sample-manifest.csv"), f"{destination}/sample-manifest.csv", "-P"],
    ]
    for cmd in commands:
        if dry_run:
            cmd.append("--dry-run")
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

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
    print(f"[smoke] XTTS: pretrained coqui/XTTS-v2 (language={args.language})")
    print(f"[smoke] RVC:  {rvc_model_path.name}" + (f" + {rvc_index_path.name}" if rvc_index_path else ""))
    print(f"[smoke] speaker ref: {speaker_wav.name}")
    print(f"[smoke] {len(smoke_lines)} sentences")

    # --- Output directory lives under the RVC run ---
    output_dir = rvc_run_dir / "smoke-pretrained-xtts"
    xtts_dir = output_dir / "xtts"
    rvc_dir = output_dir / "rvc"

    # Step 1: Generate XTTS samples with pretrained model
    print(f"\n[xtts] generating {len(smoke_lines)} samples with pretrained XTTS-v2...")
    xtts_samples = generate_pretrained_xtts_samples(
        speaker_wav=speaker_wav,
        smoke_lines=smoke_lines,
        output_dir=xtts_dir,
        language=args.language,
        device=args.device,
    )

    # Step 2: Convert through RVC
    print(f"\n[rvc] converting {len(xtts_samples)} samples...")
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

    # Write sample manifest
    manifest_rows: list[dict[str, str]] = []
    for (xtts_wav, text), rvc_wav in zip(xtts_samples, rvc_outputs):
        manifest_rows.append({
            "sample_id": xtts_wav.stem,
            "text": text,
            "xtts_audio": f"xtts/{xtts_wav.name}",
            "rvc_audio": f"rvc/{rvc_wav.name}",
        })

    manifest_path = output_dir / "sample-manifest.csv"
    write_sample_manifest(manifest_path, manifest_rows)

    # Write candidates.json
    candidates_payload = {
        "xtts_model": "coqui/XTTS-v2 (pretrained)",
        "xtts_language": args.language,
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
        "speaker_wav": speaker_wav.name,
        "generated_at": utc_now_iso(),
        "sample_count": len(smoke_lines),
        "status": "pending",
    }
    candidates_path = output_dir / "candidates.json"
    write_json(candidates_path, candidates_payload)

    print(f"\nwrote {len(smoke_lines)} sample pairs to {output_dir}")
    print(f"  xtts/ — raw pretrained XTTS-v2 output")
    print(f"  rvc/  — XTTS → RVC converted")
    print(f"  candidates.json + sample-manifest.csv")

    # Upload
    if args.remote_prefix and not args.no_upload:
        maybe_upload_review(
            output_dir,
            candidates_path,
            args.remote_prefix,
            args.namespace,
            args.voice,
            args.rvc_run_id,
            args.dry_run_upload,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
