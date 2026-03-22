#!/usr/bin/env python3
"""RVC v2 inference helper — converts a source WAV through a trained RVC model.

Uses Applio's VoiceConverter class directly via Python import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _find_applio() -> Optional[Path]:
    """Locate Applio installation."""
    env_dir = os.environ.get("APPLIO_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p
    for candidate in [Path("/workspace/Applio"), Path.home() / "Applio", Path("/opt/Applio")]:
        if candidate.is_dir():
            return candidate
    return None


def _ensure_applio_on_path() -> Path:
    """Add Applio to sys.path and return its directory."""
    applio_path = _find_applio()
    if applio_path is None:
        raise SystemExit(
            "Applio installation not found. Install at /workspace/Applio or set APPLIO_DIR."
        )
    applio_str = str(applio_path)
    if applio_str not in sys.path:
        sys.path.insert(0, applio_str)
    # Applio scripts use os.getcwd() for relative paths — ensure cwd is Applio root
    os.chdir(applio_str)
    return applio_path


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
    **kwargs,
) -> Path:
    """Convert a single audio file through an RVC v2 model using Applio's VoiceConverter."""
    model_path = Path(model_path)
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_cwd = os.getcwd()
    try:
        _ensure_applio_on_path()

        from rvc.infer.infer import VoiceConverter

        converter = VoiceConverter()
        converter.convert_audio(
            audio_input_path=str(input_path),
            audio_output_path=str(output_path),
            model_path=str(model_path),
            index_path=str(index_path) if index_path else "",
            pitch=f0_up_key,
            f0_method=f0_method,
            index_rate=index_rate,
            volume_envelope=rms_mix_rate,
            protect=protect,
            hop_length=128,
            split_audio=False,
            f0_autotune=False,
            embedder_model="contentvec",
            clean_audio=False,
            export_format="WAV",
            resample_sr=resample_sr,
            sid=0,
        )
    finally:
        os.chdir(original_cwd)

    if not output_path.is_file():
        raise SystemExit(f"RVC inference did not produce output at {output_path}")
    return output_path
