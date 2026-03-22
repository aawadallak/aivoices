#!/usr/bin/env python3
"""RVC v2 training engine for repository-native training runs.

Wraps the Applio/RVC training pipeline with run tracking, manifest
generation, and consistent directory layout.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

from scripts.pipeline.rvc_common import (
    TRAINING_ROOT,
    logs_dir,
    model_dir,
    run_manifest_path,
    utc_now_iso,
    validate_rvc_dataset,
    write_json,
)


PRETRAINED_ROOT = TRAINING_ROOT / "_shared" / "pretrained_v2"
HUBERT_ROOT = TRAINING_ROOT / "_shared" / "hubert"
RMVPE_ROOT = TRAINING_ROOT / "_shared" / "rmvpe"

CONTENTVEC_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

PRETRAINED_URLS = {
    "f0G48k.pth": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth",
    "f0D48k.pth": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return
    print(f"[rvc] downloading {url}")
    urllib.request.urlretrieve(url, dest)


def ensure_pretrained_models() -> dict[str, Path]:
    """Download all pretrained models if not already cached."""
    paths: dict[str, Path] = {}

    contentvec_path = HUBERT_ROOT / "hubert_base.pt"
    _download(CONTENTVEC_URL, contentvec_path)
    paths["contentvec"] = contentvec_path

    rmvpe_path = RMVPE_ROOT / "rmvpe.pt"
    _download(RMVPE_URL, rmvpe_path)
    paths["rmvpe"] = rmvpe_path

    for name, url in PRETRAINED_URLS.items():
        dest = PRETRAINED_ROOT / name
        _download(url, dest)
        paths[name] = dest

    return paths


def prepare_run_manifest(
    *,
    run_dir: Path,
    voice: str,
    dataset_name: str,
    dataset_uri: str,
    sample_rate: int,
    total_epoch: int,
    status: str,
    notes: str = "",
) -> dict:
    existing = {}
    manifest_path = run_manifest_path(run_dir)
    if manifest_path.is_file():
        import json
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest = {
        "run_id": run_dir.name,
        "voice": voice,
        "dataset_name": dataset_name,
        "dataset_uri": dataset_uri,
        "sample_rate": sample_rate,
        "total_epoch": total_epoch,
        "started_at": existing.get("started_at", utc_now_iso()),
        "finished_at": utc_now_iso() if status in {"completed", "failed"} else existing.get("finished_at", ""),
        "status": status,
        "notes": notes or existing.get("notes", ""),
    }
    write_json(manifest_path, manifest)
    return manifest


def _find_applio(applio_dir: str | None) -> Path:
    """Locate the Applio installation directory."""
    if applio_dir:
        p = Path(applio_dir).expanduser().resolve()
        if p.is_dir():
            return p
        raise SystemExit(f"Applio directory not found: {p}")

    candidates = [
        Path("/workspace/Applio"),
        Path.home() / "Applio",
        Path("/opt/Applio"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise SystemExit(
        "Applio installation not found. Set --applio-dir or install Applio at /workspace/Applio"
    )


def _symlink_pretrained(applio_path: Path, pretrained: dict[str, Path]) -> None:
    """Symlink shared pretrained models into Applio's expected asset directories."""
    hubert_dir = applio_path / "assets" / "hubert"
    hubert_dir.mkdir(parents=True, exist_ok=True)
    hubert_target = hubert_dir / "hubert_base.pt"
    if not hubert_target.exists():
        hubert_target.symlink_to(pretrained["contentvec"])

    rmvpe_dir = applio_path / "assets" / "rmvpe"
    rmvpe_dir.mkdir(parents=True, exist_ok=True)
    rmvpe_target = rmvpe_dir / "rmvpe.pt"
    if not rmvpe_target.exists():
        rmvpe_target.symlink_to(pretrained["rmvpe"])

    # RMVPE predictor model — extract.py loads from rvc/models/predictors/rmvpe.pt
    predictor_dir = applio_path / "rvc" / "models" / "predictors"
    predictor_dir.mkdir(parents=True, exist_ok=True)
    predictor_target = predictor_dir / "rmvpe.pt"
    if not predictor_target.exists():
        predictor_target.symlink_to(pretrained["rmvpe"])

    pv2_dir = applio_path / "assets" / "pretrained_v2"
    pv2_dir.mkdir(parents=True, exist_ok=True)
    for name in ("f0G48k.pth", "f0D48k.pth"):
        target = pv2_dir / name
        if not target.exists():
            target.symlink_to(pretrained[name])


def train_run(
    *,
    dataset_dir: Path,
    run_dir: Path,
    voice: str,
    applio_dir: str | None = None,
    sample_rate: int = 48000,
    total_epoch: int = 400,
    batch_size: int = 8,
    save_every_epoch: int = 50,
    f0_method: str = "rmvpe",
) -> Path:
    """Run the full RVC v2 training pipeline via Applio CLI scripts."""
    summary = validate_rvc_dataset(dataset_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    applio_path = _find_applio(applio_dir)
    pretrained = ensure_pretrained_models()
    _symlink_pretrained(applio_path, pretrained)

    prepare_run_manifest(
        run_dir=run_dir,
        voice=voice,
        dataset_name=summary.dataset_name,
        dataset_uri=str(dataset_dir),
        sample_rate=sample_rate,
        total_epoch=total_epoch,
        status="running",
    )

    wavs_dir = dataset_dir / "wavs"
    rvc_logs = applio_path / "logs" / voice
    rvc_logs.mkdir(parents=True, exist_ok=True)

    num_cpu = str(__import__("multiprocessing").cpu_count())
    exp_dir = str(rvc_logs)

    try:
        # Step 1: Preprocess
        # positional: experiment_dir input_root sample_rate num_processes
        #             cut_preprocess process_effects noise_reduction
        #             reduction_strength chunk_len overlap_len normalization_mode
        print(f"[rvc] preprocessing {summary.wav_count} files at {sample_rate}Hz")
        subprocess.run(
            [
                "python", str(applio_path / "rvc" / "train" / "preprocess" / "preprocess.py"),
                exp_dir,           # experiment_directory
                str(wavs_dir),     # input_root
                str(sample_rate),  # sample_rate
                num_cpu,           # num_processes
                "Skip",            # cut_preprocess (Skip: clips already sliced)
                "false",           # process_effects
                "false",           # noise_reduction
                "0.7",             # reduction_strength
                "0.5",             # chunk_len
                "0.3",             # overlap_len
                "Loudness",        # normalization_mode
            ],
            cwd=str(applio_path),
            check=True,
        )

        # Step 2: Extract features (f0 + embeddings in one script)
        # positional: exp_dir f0_method num_processes gpus sample_rate
        #             embedder_model [embedder_model_custom] [include_mutes]
        print(f"[rvc] extracting f0 ({f0_method}) + ContentVec features")
        subprocess.run(
            [
                "python", str(applio_path / "rvc" / "train" / "extract" / "extract.py"),
                exp_dir,           # exp_dir
                f0_method,         # f0_method
                num_cpu,           # num_processes
                "0",               # gpus (0 = first GPU)
                str(sample_rate),  # sample_rate
                "contentvec",      # embedder_model
                "None",            # embedder_model_custom
                "2",               # include_mutes
            ],
            cwd=str(applio_path),
            check=True,
        )

        # Step 3: Train
        # positional: model_name save_every_epoch total_epoch pretrainG pretrainD
        #             gpus batch_size sample_rate save_only_latest save_every_weights
        #             cache_data_in_gpu overtraining_detector overtraining_threshold
        #             cleanup vocoder checkpointing
        print(f"[rvc] training {total_epoch} epochs, batch_size={batch_size}")
        subprocess.run(
            [
                "python", str(applio_path / "rvc" / "train" / "train.py"),
                voice,                          # model_name
                str(save_every_epoch),           # save_every_epoch
                str(total_epoch),                # total_epoch
                str(pretrained["f0G48k.pth"]),   # pretrainG
                str(pretrained["f0D48k.pth"]),   # pretrainD
                "0",                             # gpus
                str(batch_size),                 # batch_size
                str(sample_rate),                # sample_rate
                "false",                         # save_only_latest
                "true",                          # save_every_weights
                "false",                         # cache_data_in_gpu
                "false",                         # overtraining_detector
                "50",                            # overtraining_threshold
                "false",                         # cleanup
                "Default",                       # vocoder
                "true",                          # checkpointing
            ],
            cwd=str(applio_path),
            check=True,
        )

        # Step 4: Generate FAISS index
        # positional: exp_dir index_algorithm
        print("[rvc] generating FAISS retrieval index")
        subprocess.run(
            [
                "python", str(applio_path / "rvc" / "train" / "process" / "extract_index.py"),
                exp_dir,    # exp_dir
                "Auto",     # index_algorithm
            ],
            cwd=str(applio_path),
            check=True,
        )

    except subprocess.CalledProcessError as exc:
        prepare_run_manifest(
            run_dir=run_dir,
            voice=voice,
            dataset_name=summary.dataset_name,
            dataset_uri=str(dataset_dir),
            sample_rate=sample_rate,
            total_epoch=total_epoch,
            status="failed",
            notes=f"Step failed with exit code {exc.returncode}",
        )
        raise

    # Collect outputs into run directory
    out_model = model_dir(run_dir)
    out_logs = logs_dir(run_dir)
    out_model.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    # Copy best generator checkpoint
    generator_candidates = sorted(rvc_logs.glob("G_*.pth"))
    if generator_candidates:
        best_g = generator_candidates[-1]
        shutil.copy2(best_g, out_model / best_g.name)

    # Copy FAISS index
    index_candidates = list(rvc_logs.glob(f"added_*.index"))
    if index_candidates:
        shutil.copy2(index_candidates[0], out_model / index_candidates[0].name)

    # Copy total_fea.npy if present
    total_fea = rvc_logs / "total_fea.npy"
    if total_fea.is_file():
        shutil.copy2(total_fea, out_model / "total_fea.npy")

    prepare_run_manifest(
        run_dir=run_dir,
        voice=voice,
        dataset_name=summary.dataset_name,
        dataset_uri=str(dataset_dir),
        sample_rate=sample_rate,
        total_epoch=total_epoch,
        status="completed",
    )
    return run_dir
