#!/usr/bin/env python3
"""Run WhisperX diarization in batch for all extracted audio in a namespace."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODES_DIR = REPO_ROOT / "episodes"
METADATA_DIR = REPO_ROOT / "metadata"
EXPORT_SAMPLES_SCRIPT = Path(__file__).resolve().parent / "export_speaker_samples.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WhisperX diarization for every extracted audio file in a namespace."
    )
    parser.add_argument("--namespace", "--series", dest="namespace", required=True, help="Namespace slug.")
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="Glob pattern inside episodes/<namespace>/extracted-audio/. Default: *.wav.",
    )
    parser.add_argument(
        "--output-subdir",
        default="whisperx-batch-diarized",
        help="Subdirectory under metadata/<namespace>/transcripts/. Default: whisperx-batch-diarized.",
    )
    parser.add_argument(
        "--review-run-id",
        default="whisperx-batch-review",
        help="Run id prefix for exported speaker samples. Default: whisperx-batch-review.",
    )
    parser.add_argument("--language", default="pt", help="Language passed to WhisperX. Default: pt.")
    parser.add_argument("--model", default="large-v3", help="WhisperX model name. Default: large-v3.")
    parser.add_argument("--device", default="cpu", help="WhisperX device. Default: cpu.")
    parser.add_argument(
        "--compute-type",
        default="float32",
        help="WhisperX compute type. Default: float32.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="WhisperX batch size. Default: 8.")
    parser.add_argument("--max-speakers", type=int, default=5, help="Max speakers for diarization. Default: 5.")
    parser.add_argument(
        "--export-samples",
        action="store_true",
        help="Also export review samples for each diarized episode.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSON outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them.")
    return parser.parse_args()


def ensure_tools() -> None:
    if shutil.which("whisperx") is None:
        sys.exit("whisperx was not found in PATH.")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found in PATH.")


def extracted_audio_files(namespace: str, pattern: str) -> list[Path]:
    root = EPISODES_DIR / namespace / "extracted-audio"
    if not root.is_dir():
        sys.exit(f"Extracted audio directory not found: {root}")
    return sorted(root.glob(pattern))


def whisperx_output_json(namespace: str, output_subdir: str, episode_id: str) -> Path:
    out_dir = METADATA_DIR / namespace / "transcripts" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{episode_id}.json"


def whisperx_command(args: argparse.Namespace, audio_path: Path, output_dir: Path) -> list[str]:
    hf_token = Path("/dev/null")  # marker to avoid accidental token persistence in source
    del hf_token
    token = __import__("os").environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is required in the environment for WhisperX diarization.")
    return [
        "whisperx",
        str(audio_path),
        "--model",
        args.model,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--compute_type",
        args.compute_type,
        "--language",
        args.language,
        "--vad_method",
        "silero",
        "--diarize",
        "--speaker_embeddings",
        "--max_speakers",
        str(args.max_speakers),
        "--hf_token",
        token,
        "--output_dir",
        str(output_dir),
        "--output_format",
        "json",
    ]


def export_samples_command(namespace: str, episode_id: str, json_path: Path, run_id: str, force: bool) -> list[str]:
    command = [
        "python3",
        str(EXPORT_SAMPLES_SCRIPT),
        "--namespace",
        namespace,
        "--episode-id",
        episode_id,
        "--json-path",
        str(json_path),
        "--run-id",
        run_id,
    ]
    if force:
        command.append("--force")
    return command


def run_command(command: list[str], dry_run: bool) -> int:
    display = list(command)
    if "--hf_token" in display:
        token_index = display.index("--hf_token") + 1
        if token_index < len(display):
            display[token_index] = "<redacted>"
    print(" ".join(display))
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def main() -> int:
    args = parse_args()
    ensure_tools()
    audio_files = extracted_audio_files(args.namespace, args.pattern)
    if not audio_files:
        sys.exit(f"No files matched {args.pattern!r} in episodes/{args.namespace}/extracted-audio/")

    failures = 0
    for audio_path in audio_files:
        episode_id = audio_path.stem
        output_json = whisperx_output_json(args.namespace, args.output_subdir, episode_id)
        output_dir = output_json.parent

        if output_json.exists() and not args.force:
            print(f"skip {output_json} (already exists)")
        else:
            code = run_command(whisperx_command(args, audio_path, output_dir), args.dry_run)
            if code != 0:
                failures += 1
                print(f"whisperx failed for {episode_id}", file=sys.stderr)
                break

        if args.export_samples:
            review_run_id = f"{args.review_run_id}-{episode_id}"
            code = run_command(
                export_samples_command(
                    namespace=args.namespace,
                    episode_id=episode_id,
                    json_path=output_json,
                    run_id=review_run_id,
                    force=args.force,
                ),
                args.dry_run,
            )
            if code != 0:
                failures += 1
                print(f"speaker sample export failed for {episode_id}", file=sys.stderr)
                break

    if failures:
        return 1

    print(f"processed {len(audio_files)} extracted audio files for namespace {args.namespace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
