#!/usr/bin/env python3
"""XTTS training engine for repository-native training runs."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from scripts.pipeline.xtts_common import (
    SHARED_MODEL_ROOT,
    trainer_dir,
    trainer_config_path,
    run_manifest_path,
    utc_now_iso,
    validate_xtts_dataset,
    write_json,
)


XTTS_BASE_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main"
MODEL_FILES = ("dvae.pth", "mel_stats.pth", "vocab.json", "model.pth", "config.json")


def ensure_xtts_model_files(model_root: Path) -> dict[str, Path]:
    model_root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for file_name in MODEL_FILES:
        path = model_root / file_name
        if not path.is_file():
            url = f"{XTTS_BASE_URL}/{file_name}"
            print(f"[train] downloading {url}")
            urllib.request.urlretrieve(url, path)
        resolved[file_name] = path
    return resolved


def prepare_run_manifest(
    *,
    run_dir: Path,
    voice: str,
    dataset_name: str,
    dataset_uri: str,
    language: str,
    status: str,
    notes: str = "",
) -> dict:
    existing = {}
    manifest_path = run_manifest_path(run_dir)
    if manifest_path.is_file():
        existing = {
            **existing,
            **__import__("json").loads(manifest_path.read_text(encoding="utf-8")),
        }
    manifest = {
        "run_id": run_dir.name,
        "voice": voice,
        "dataset_name": dataset_name,
        "dataset_uri": dataset_uri,
        "language": language,
        "started_at": existing.get("started_at", utc_now_iso()),
        "finished_at": utc_now_iso() if status in {"completed", "failed"} else existing.get("finished_at", ""),
        "status": status,
        "notes": notes or existing.get("notes", ""),
    }
    write_json(manifest_path, manifest)
    return manifest


def train_run(
    *,
    dataset_dir: Path,
    run_dir: Path,
    language: str = "pt-BR",
    epochs: int = 10,
    batch_size: int = 4,
    grad_accum: int = 8,
    max_audio_seconds: int = 11,
    max_text_chars: int = 200,
    precision: str = "fp16",
    mixed_precision: bool = True,
    save_step: int = 1000,
    plot_step: int = 10**9,
    log_model_step: int = 10**9,
    batch_group_size: int = 48,
    num_loader_workers: int = 8,
    num_eval_loader_workers: int = 8,
    restore_path: str | None = None,
    continue_path: str | None = None,
) -> Path:
    summary = validate_xtts_dataset(dataset_dir, language=language)
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer_output_dir = trainer_dir(run_dir)
    trainer_output_dir.mkdir(parents=True, exist_ok=True)

    prepare_run_manifest(
        run_dir=run_dir,
        voice=summary.voice,
        dataset_name=summary.dataset_name,
        dataset_uri=str(dataset_dir),
        language=language,
        status="running",
    )

    try:
        import torch
        from trainer import Trainer, TrainerArgs
        from trainer.logging.dummy_logger import DummyLogger

        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.xtts_config import XttsAudioConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
        from TTS.tts.utils.speakers import SpeakerManager
    except ImportError as exc:
        raise SystemExit(
            "XTTS training dependencies are missing. Install Coqui TTS and trainer in the training runtime."
        ) from exc

    model_files = ensure_xtts_model_files(SHARED_MODEL_ROOT)

    train_csv = dataset_dir / "metadata_train.csv"
    eval_csv = dataset_dir / "metadata_eval.csv"

    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="xtts_dataset",
        path=str(dataset_dir),
        meta_file_train=train_csv.name,
        meta_file_val=eval_csv.name,
        language=language,
    )

    max_audio_length = int(max_audio_seconds * 22050)
    max_conditioning_length = min(int(4 * 22050), max_audio_length)
    min_conditioning_length = min(int(2 * 22050), max_conditioning_length)

    model_args = GPTArgs(
        max_conditioning_length=max_conditioning_length,
        min_conditioning_length=min_conditioning_length,
        debug_loading_failures=False,
        max_wav_length=max_audio_length,
        max_text_length=max_text_chars,
        mel_norm_file=str(model_files["mel_stats.pth"]),
        dvae_checkpoint=str(model_files["dvae.pth"]),
        xtts_checkpoint=str(model_files["model.pth"]),
        tokenizer_file=str(model_files["vocab.json"]),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    config = GPTTrainerConfig(
        mixed_precision=bool(mixed_precision and torch.cuda.is_available()),
        precision=precision if torch.cuda.is_available() else "fp32",
        use_grad_scaler=bool(mixed_precision and torch.cuda.is_available()),
        epochs=epochs,
        output_path=str(trainer_output_dir),
        model_args=model_args,
        run_name=run_dir.name,
        project_name="aivoices-xtts",
        run_description="XTTS training run",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=16000),
        batch_size=batch_size,
        batch_group_size=batch_group_size,
        eval_batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        num_eval_loader_workers=num_eval_loader_workers,
        eval_split_max_size=256,
        print_step=50,
        plot_step=plot_step,
        log_model_step=log_model_step,
        save_step=save_step,
        save_n_checkpoints=4,
        save_checkpoints=True,
        print_eval=False,
        run_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2, "foreach": False},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
        max_text_len=max_text_chars,
        max_audio_len=max_audio_length,
    )

    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.speaker_manager = speaker_manager
    config.num_speakers = speaker_manager.num_speakers

    # Monkey-patch XttsArgs so debug_loading_failures always exists, even
    # when the trainer reloads config from a checkpoint that lacks the field.
    from TTS.tts.layers.xtts.trainer.gpt_trainer import XttsArgs
    if not hasattr(XttsArgs, "debug_loading_failures"):
        XttsArgs.debug_loading_failures = False

    model = GPTTrainer.init_from_config(config)

    trainer_args_kwargs = {
        "restore_path": restore_path,
        "skip_train_epoch": False,
        "start_with_eval": False,
        "grad_accum_steps": grad_accum,
    }
    if continue_path:
        trainer_args_kwargs["continue_path"] = continue_path

    trainer = Trainer(
        TrainerArgs(**trainer_args_kwargs),
        config,
        output_path=str(trainer_output_dir),
        dashboard_logger=DummyLogger(),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.update_training_dashboard_logger = lambda *args, **kwargs: None

    try:
        trainer.fit()
    except Exception:
        prepare_run_manifest(
            run_dir=run_dir,
            voice=summary.voice,
            dataset_name=summary.dataset_name,
            dataset_uri=str(dataset_dir),
            language=language,
            status="failed",
        )
        raise

    config_path = trainer_config_path(run_dir)
    if not config_path.is_file():
        fallback = Path(trainer.output_path) / "config.json"
        if fallback.is_file():
            shutil.copy2(fallback, config_path)

    prepare_run_manifest(
        run_dir=run_dir,
        voice=summary.voice,
        dataset_name=summary.dataset_name,
        dataset_uri=str(dataset_dir),
        language=language,
        status="completed",
    )
    return run_dir
