#!/usr/bin/env python3
"""RVC v2 inference helper — converts a source WAV through a trained RVC model.

This module provides a Python-native inference path that loads the RVC
generator directly (no Applio subprocess required). It uses the same
ContentVec + RMVPE + FAISS pipeline that Applio uses under the hood.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _load_contentvec(hubert_path: str, device: str):
    """Load ContentVec / HuBERT model for feature extraction."""
    from fairseq import checkpoint_utils

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path], suffix=""
    )
    model = models[0].to(device)
    model.eval()
    return model


def _extract_f0_rmvpe(audio: np.ndarray, sr: int, rmvpe_path: str, device: str) -> np.ndarray:
    """Extract pitch (f0) using RMVPE."""
    import torch

    # Use torchcrepe as fallback-safe import; RMVPE is preferred
    try:
        from rvc.lib.rmvpe import RMVPE
        rmvpe = RMVPE(rmvpe_path, device=device)
        f0 = rmvpe.infer_from_audio(torch.from_numpy(audio).float().to(device), thred=0.03)
    except ImportError:
        # Fallback: use pyworld harvest
        import pyworld as pw
        audio_f64 = audio.astype(np.float64)
        f0, _ = pw.harvest(audio_f64, sr, f0_floor=50, f0_ceil=1100, frame_period=10)
    return f0


def convert_audio(
    *,
    model_path: str | Path,
    index_path: str | Path | None = None,
    input_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    f0_method: str = "rmvpe",
    f0_up_key: int = 0,
    index_rate: float = 0.75,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    hubert_path: str | None = None,
    rmvpe_path: str | None = None,
) -> Path:
    """Convert a single audio file through an RVC v2 model.

    Uses Applio's inference pipeline if available, otherwise falls back to
    a subprocess call to Applio's CLI.
    """
    import subprocess
    import shutil

    model_path = Path(model_path)
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try Applio CLI inference (most reliable approach)
    applio_path = _find_applio()
    if applio_path:
        return _convert_via_applio_cli(
            applio_path=applio_path,
            model_path=model_path,
            index_path=Path(index_path) if index_path else None,
            input_path=input_path,
            output_path=output_path,
            f0_method=f0_method,
            f0_up_key=f0_up_key,
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )

    raise SystemExit(
        "Applio installation not found. Install Applio at /workspace/Applio or set APPLIO_DIR."
    )


def _find_applio() -> Optional[Path]:
    """Locate Applio installation."""
    import os

    env_dir = os.environ.get("APPLIO_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p

    for candidate in [Path("/workspace/Applio"), Path.home() / "Applio", Path("/opt/Applio")]:
        if candidate.is_dir():
            return candidate
    return None


def _convert_via_applio_cli(
    *,
    applio_path: Path,
    model_path: Path,
    index_path: Optional[Path],
    input_path: Path,
    output_path: Path,
    f0_method: str,
    f0_up_key: int,
    index_rate: float,
    filter_radius: int,
    resample_sr: int,
    rms_mix_rate: float,
    protect: float,
) -> Path:
    """Run RVC inference via Applio's CLI script."""
    import subprocess

    cmd = [
        "python", str(applio_path / "rvc" / "infer" / "infer.py"),
        "--model_path", str(model_path),
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--f0_method", f0_method,
        "--f0_up_key", str(f0_up_key),
        "--index_rate", str(index_rate),
        "--filter_radius", str(filter_radius),
        "--resample_sr", str(resample_sr),
        "--rms_mix_rate", str(rms_mix_rate),
        "--protect", str(protect),
    ]
    if index_path and index_path.is_file():
        cmd.extend(["--index_path", str(index_path)])

    subprocess.run(cmd, cwd=str(applio_path), check=True)

    if not output_path.is_file():
        raise SystemExit(f"RVC inference did not produce output at {output_path}")
    return output_path
