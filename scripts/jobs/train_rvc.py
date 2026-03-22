#!/usr/bin/env python3
"""Train RVC v2 from a dataset directory of WAV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.pipeline.rvc_common import make_run_id, run_root, validate_rvc_dataset
from scripts.pipeline.rvc_training import train_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RVC v2 from a dataset directory.")
    parser.add_argument("--namespace", required=True, help="Namespace slug.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing wavs/ for training.")
    parser.add_argument("--voice", help="Voice slug. Defaults to voice inferred from filenames.")
    parser.add_argument("--run-id", help="Explicit run ID. Default: auto-generated.")
    parser.add_argument("--output-root", help="Override base training root directory.")
    parser.add_argument("--applio-dir", help="Path to Applio installation. Default: auto-detect.")
    parser.add_argument("--sample-rate", type=int, default=48000, choices=[32000, 40000, 48000])
    parser.add_argument("--total-epoch", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every-epoch", type=int, default=50)
    parser.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "crepe"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    summary = validate_rvc_dataset(dataset_dir)
    voice = args.voice or summary.voice
    run_id = args.run_id or make_run_id(voice, summary.dataset_name)

    if args.output_root:
        run_dir = Path(args.output_root).expanduser().resolve() / args.namespace / voice / run_id
    else:
        run_dir = run_root(args.namespace, voice, run_id)

    train_run(
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        voice=voice,
        applio_dir=args.applio_dir,
        sample_rate=args.sample_rate,
        total_epoch=args.total_epoch,
        batch_size=args.batch_size,
        save_every_epoch=args.save_every_epoch,
        f0_method=args.f0_method,
    )
    print(f"training run ready at {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
