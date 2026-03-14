#!/usr/bin/env python3
"""Promote XTTS run artifacts based on a promotion manifest."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.pipeline.xtts_common import (
    SHARED_MODEL_ROOT,
    artifact_dir,
    best_model_path,
    candidates_manifest_path,
    current_model_manifest_path,
    latest_checkpoint_path,
    promotion_manifest_path,
    read_json,
    run_manifest_path,
    run_root,
    trainer_dir,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote best/last artifacts for an XTTS run.")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--promotion-json", help="Override promotion manifest path.")
    parser.add_argument("--remote-prefix", help="Optional remote run prefix, e.g. r2:aivoices/training/runs.")
    parser.add_argument("--models-remote-prefix", help="Optional remote models prefix, e.g. r2:aivoices/training/models.")
    parser.add_argument("--dry-run-upload", action="store_true")
    return parser.parse_args()


def copy_artifact_bundle(
    *,
    run_dir: Path,
    kind: str,
    checkpoint_path: Path,
    step: int,
    dataset_name: str,
    language: str,
    notes: str,
) -> Path:
    source_config = trainer_dir(run_dir) / "config.json"
    source_vocab = SHARED_MODEL_ROOT / "vocab.json"
    if not source_config.is_file():
        raise SystemExit(f"Trainer config not found: {source_config}")
    if not source_vocab.is_file():
        raise SystemExit(f"Shared vocab.json not found: {source_vocab}")

    target_dir = artifact_dir(run_dir, kind)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_checkpoint = target_dir / checkpoint_path.name
    shutil.copy2(checkpoint_path, target_checkpoint)
    shutil.copy2(source_config, target_dir / "config.json")
    shutil.copy2(source_vocab, target_dir / "vocab.json")

    manifest = {
        "run_id": run_dir.name,
        "voice": run_dir.parent.name,
        "kind": kind,
        "step": step,
        "checkpoint_file": checkpoint_path.name,
        "dataset_name": dataset_name,
        "language": language,
        "promoted_at": utc_now_iso() if kind == "best" else "",
        "created_at": utc_now_iso(),
        "resume_mode": "continue_path" if kind == "last" else "",
        "notes": notes,
    }
    write_json(target_dir / "manifest.json", manifest)
    return target_dir


def maybe_upload(path: Path, destination: str, dry_run: bool) -> None:
    if shutil.which("rclone") is None:
        raise SystemExit("rclone not found in PATH")
    cmd = ["rclone", "copy", str(path), destination, "-P", "--create-empty-src-dirs"]
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
    promotion_path = Path(args.promotion_json).expanduser().resolve() if args.promotion_json else promotion_manifest_path(run_dir)
    if not promotion_path.is_file():
        raise SystemExit(f"Promotion manifest not found: {promotion_path}")
    promotion = read_json(promotion_path)

    candidates = read_json(candidates_manifest_path(run_dir))
    candidate_map = {item["checkpoint_id"]: item for item in candidates.get("candidates", [])}
    promoted_id = promotion["promote_checkpoint"]
    if promoted_id not in candidate_map:
        raise SystemExit(f"promote_checkpoint {promoted_id} not found in candidates.json")

    training_dir = trainer_dir(run_dir)
    best_candidate_file = training_dir / f"{promoted_id}.pth"
    if not best_candidate_file.is_file():
        fallback = best_model_path(training_dir)
        if fallback and fallback.stem == promoted_id:
            best_candidate_file = fallback
    if not best_candidate_file.is_file():
        raise SystemExit(f"Promoted checkpoint file not found for candidate {promoted_id}")

    last_candidate_file = latest_checkpoint_path(training_dir)
    if last_candidate_file is None:
        raise SystemExit(f"No latest checkpoint found under {training_dir}")

    best_dir = copy_artifact_bundle(
        run_dir=run_dir,
        kind="best",
        checkpoint_path=best_candidate_file,
        step=int(candidate_map[promoted_id].get("step", -1)),
        dataset_name=run_manifest["dataset_name"],
        language=run_manifest["language"],
        notes=promotion.get("notes", ""),
    )
    last_dir = copy_artifact_bundle(
        run_dir=run_dir,
        kind="last",
        checkpoint_path=last_candidate_file,
        step=-1,
        dataset_name=run_manifest["dataset_name"],
        language=run_manifest["language"],
        notes="latest checkpoint for resume",
    )

    current_payload = {
        "voice": args.voice,
        "run_id": args.run_id,
        "checkpoint_kind": "best",
        "artifact_path": f"training/runs/{args.namespace}/{args.voice}/{args.run_id}/artifacts/best",
        "dataset_name": run_manifest["dataset_name"],
        "language": run_manifest["language"],
        "promoted_at": utc_now_iso(),
        "notes": promotion.get("notes", ""),
    }
    current_path = current_model_manifest_path(args.namespace, args.voice)
    write_json(current_path, current_payload)
    print(f"wrote {current_path}")

    if args.remote_prefix:
        base = f"{args.remote_prefix.rstrip('/')}/{args.namespace}/{args.voice}/{args.run_id}/artifacts"
        maybe_upload(best_dir, f"{base}/best", args.dry_run_upload)
        if promotion.get("keep_last", True):
            maybe_upload(last_dir, f"{base}/last", args.dry_run_upload)

    if args.models_remote_prefix:
        destination = f"{args.models_remote_prefix.rstrip('/')}/{args.namespace}/{args.voice}"
        maybe_upload(current_path.parent, destination, args.dry_run_upload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
