#!/usr/bin/env python3
"""Train XTTS from a portable dataset package."""

from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder has been deprecated.*")
warnings.filterwarnings("ignore", message=".*torchaudio.load_with_torchcodec.*")
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.pipeline.xtts_common import make_run_id, run_root, validate_xtts_dataset
from scripts.pipeline.xtts_training import train_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XTTS from a dataset-dir package.")
    parser.add_argument("--namespace", required=True, help="Namespace slug.")
    parser.add_argument("--dataset-dir", required=True, help="XTTS package directory.")
    parser.add_argument("--voice", help="Voice slug. Defaults to the single speaker_name found in the dataset.")
    parser.add_argument("--run-id", help="Explicit run ID. Default: auto-generated.")
    parser.add_argument("--language", default="pt-BR")
    parser.add_argument("--output-root", help="Override base training root directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-audio-seconds", type=int, default=11)
    parser.add_argument("--max-text-chars", type=int, default=200)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--save-step", type=int, default=1000)
    parser.add_argument("--plot-step", type=int, default=10**9)
    parser.add_argument("--log-model-step", type=int, default=10**9)
    parser.add_argument("--batch-group-size", type=int, default=48)
    parser.add_argument("--num-loader-workers", type=int, default=8)
    parser.add_argument("--num-eval-loader-workers", type=int, default=8)
    parser.add_argument("--restore-path", help="Advanced mode: restore from a specific checkpoint.")
    parser.add_argument("--continue-path", help="Resume from an existing trainer directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    summary = validate_xtts_dataset(dataset_dir, language=args.language)
    voice = args.voice or summary.voice
    run_id = args.run_id or make_run_id(voice, summary.dataset_name)

    if args.output_root:
        run_dir = Path(args.output_root).expanduser().resolve() / args.namespace / voice / run_id
    else:
        run_dir = run_root(args.namespace, voice, run_id)

    train_run(
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        language=args.language,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_audio_seconds=args.max_audio_seconds,
        max_text_chars=args.max_text_chars,
        precision=args.precision,
        mixed_precision=not args.no_mixed_precision,
        save_step=args.save_step,
        plot_step=args.plot_step,
        log_model_step=args.log_model_step,
        batch_group_size=args.batch_group_size,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_eval_loader_workers,
        restore_path=args.restore_path,
        continue_path=args.continue_path,
    )
    print(f"training run ready at {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
