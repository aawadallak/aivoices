#!/usr/bin/env python3
"""Score extracted matched clips with lightweight quality heuristics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import webrtcvad


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"

FRAME_MS = 30
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score matched clips with VAD, loudness, overlap, and music heuristics."
    )
    parser.add_argument("--namespace", "--series", dest="namespace", required=True, help="Namespace slug.")
    parser.add_argument("--run-id", required=True, help="Run id under metadata/<namespace>/speakers/runs/.")
    parser.add_argument("--episode-id", required=True, help="Episode basename without extension.")
    parser.add_argument("--target-json", help="Optional WhisperX JSON path. Auto-detected when omitted.")
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="WebRTC VAD aggressiveness. Default: 3.",
    )
    parser.add_argument("--min-speech-ratio", type=float, default=0.45, help="Reject below this speech ratio.")
    parser.add_argument("--min-snr-db", type=float, default=4.0, help="Reject below this estimated SNR.")
    parser.add_argument("--usable-snr-db", type=float, default=8.0, help="Mark usable below this SNR.")
    parser.add_argument("--min-rms-dbfs", type=float, default=-32.0, help="Reject below this RMS dBFS.")
    parser.add_argument("--usable-rms-dbfs", type=float, default=-29.0, help="Mark usable below this RMS.")
    parser.add_argument("--max-music-score", type=float, default=0.76, help="Reject above this music score.")
    parser.add_argument("--usable-music-score", type=float, default=0.64, help="Mark usable above this score.")
    parser.add_argument("--max-edge-silence-sec", type=float, default=0.45, help="Mark usable above this silence.")
    parser.add_argument(
        "--max-reference-duration",
        type=float,
        default=6.0,
        help="Maximum clip duration recommended for references.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def run_root(namespace: str, run_id: str) -> Path:
    return METADATA_DIR / namespace / "speakers" / "runs" / run_id


def matched_root(namespace: str, run_id: str, episode_id: str) -> Path:
    return run_root(namespace, run_id) / "matched-clips" / episode_id


def matched_manifest_path(namespace: str, run_id: str, episode_id: str) -> Path:
    return matched_root(namespace, run_id, episode_id) / "matched-clips.csv"


def speaker_matches_path(namespace: str, run_id: str) -> Path:
    return run_root(namespace, run_id) / "speaker-matches.csv"


def autodetect_target_json(namespace: str, episode_id: str) -> Path:
    candidates = sorted((METADATA_DIR / namespace / "transcripts").glob(f"*/{episode_id}.json"))
    if not candidates:
        raise SystemExit(f"Could not auto-detect transcript JSON for {episode_id}.")
    return candidates[0]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def frame_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def vad_metrics(samples: np.ndarray, sample_rate: int, aggressiveness: int) -> dict[str, float]:
    if sample_rate != 16000:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sample_rate * FRAME_MS / 1000.0)
    usable = len(samples) - (len(samples) % frame_size)
    if usable <= 0:
        return {
            "speech_ratio": 0.0,
            "leading_silence_sec": 0.0,
            "trailing_silence_sec": 0.0,
            "speech_rms": 0.0,
            "noise_rms": 0.0,
        }

    samples = samples[:usable]
    frames = samples.reshape(-1, frame_size)
    speech_flags: list[bool] = []
    speech_samples: list[np.ndarray] = []
    noise_samples: list[np.ndarray] = []
    for frame in frames:
        is_speech = vad.is_speech(frame_bytes(frame), sample_rate)
        speech_flags.append(is_speech)
        if is_speech:
            speech_samples.append(frame)
        else:
            noise_samples.append(frame)

    first_speech = next((i for i, flag in enumerate(speech_flags) if flag), len(speech_flags))
    last_speech = next((i for i, flag in enumerate(reversed(speech_flags)) if flag), len(speech_flags))
    speech_ratio = sum(speech_flags) / len(speech_flags)
    speech_rms = float(np.sqrt(np.mean(np.square(np.concatenate(speech_samples))))) if speech_samples else 0.0
    noise_rms = float(np.sqrt(np.mean(np.square(np.concatenate(noise_samples))))) if noise_samples else 0.0
    return {
        "speech_ratio": speech_ratio,
        "leading_silence_sec": first_speech * FRAME_MS / 1000.0,
        "trailing_silence_sec": last_speech * FRAME_MS / 1000.0,
        "speech_rms": speech_rms,
        "noise_rms": noise_rms,
    }


def dbfs(value: float) -> float:
    if value <= 0:
        return -120.0
    return 20.0 * math.log10(value + EPS)


def snr_db(speech_rms: float, noise_rms: float) -> float:
    if speech_rms <= 0:
        return -120.0
    if noise_rms <= 0:
        return 60.0
    return 20.0 * math.log10((speech_rms + EPS) / (noise_rms + EPS))


def music_metrics(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    if len(samples) == 0:
        return {
            "music_score": 0.0,
            "harmonic_ratio": 0.0,
            "spectral_flatness": 0.0,
            "centroid_ratio": 0.0,
        }
    if sample_rate != 16000:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    harmonic, _ = librosa.effects.hpss(samples)
    total_rms = float(np.sqrt(np.mean(np.square(samples)))) + EPS
    harmonic_rms = float(np.sqrt(np.mean(np.square(harmonic))))
    harmonic_ratio = min(1.0, harmonic_rms / total_rms)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=samples)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=samples, sr=sample_rate)))
    centroid_ratio = min(1.0, centroid / (sample_rate / 2.0))
    music_score = max(
        0.0,
        min(1.0, (0.55 * harmonic_ratio) + (0.25 * (1.0 - flatness)) + (0.20 * centroid_ratio)),
    )
    return {
        "music_score": music_score,
        "harmonic_ratio": harmonic_ratio,
        "spectral_flatness": flatness,
        "centroid_ratio": centroid_ratio,
    }


def overlap_metrics(
    current_start: float,
    current_end: float,
    current_diarized: str,
    segments: list[dict],
) -> dict[str, float | str]:
    overlap_sec = 0.0
    nearest_gap = math.inf
    overlap_speaker = ""
    for segment in segments:
        diarized = str(segment.get("speaker", ""))
        if not diarized or diarized == current_diarized:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        overlap = max(0.0, min(current_end, end) - max(current_start, start))
        if overlap > overlap_sec:
            overlap_sec = overlap
            overlap_speaker = diarized
        gap = min(abs(current_start - end), abs(start - current_end))
        nearest_gap = min(nearest_gap, gap)
    if nearest_gap is math.inf:
        nearest_gap = 999.0
    return {
        "overlap_sec": overlap_sec,
        "nearest_other_gap_sec": nearest_gap,
        "overlap_speaker": overlap_speaker,
    }


def diarized_mapping(match_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], str]:
    mapping: dict[tuple[str, str, str], str] = {}
    for row in match_rows:
        key = (
            row.get("start_sec", "").strip(),
            row.get("end_sec", "").strip(),
            row.get("matched_speaker", "").strip(),
        )
        mapping[key] = row.get("target_diarized_speaker", "").strip()
    return mapping


def classify_clip(
    args: argparse.Namespace,
    duration_sec: float,
    speech_ratio: float,
    leading_silence_sec: float,
    trailing_silence_sec: float,
    rms_dbfs_value: float,
    snr_db_value: float,
    music_score: float,
    overlap_sec: float,
    nearest_other_gap_sec: float,
) -> tuple[str, bool, bool, list[str]]:
    reasons: list[str] = []
    label = "clean"

    if speech_ratio < args.min_speech_ratio:
        label = "reject"
        reasons.append("low_speech_ratio")
    elif speech_ratio < 0.62:
        label = "usable"
        reasons.append("borderline_speech_ratio")

    if rms_dbfs_value < args.min_rms_dbfs:
        label = "reject"
        reasons.append("too_quiet")
    elif rms_dbfs_value < args.usable_rms_dbfs and label != "reject":
        label = "usable"
        reasons.append("low_loudness")

    if snr_db_value < args.min_snr_db:
        label = "reject"
        reasons.append("low_snr")
    elif snr_db_value < args.usable_snr_db and label != "reject":
        label = "usable"
        reasons.append("borderline_snr")

    if music_score > args.max_music_score:
        label = "reject"
        reasons.append("strong_music")
    elif music_score > args.usable_music_score and label != "reject":
        label = "usable"
        reasons.append("music_present")

    if overlap_sec > 0.0:
        label = "reject"
        reasons.append("speaker_overlap")
    elif nearest_other_gap_sec < 0.12 and label != "reject":
        label = "usable"
        reasons.append("speaker_turn_close")

    if leading_silence_sec > args.max_edge_silence_sec or trailing_silence_sec > args.max_edge_silence_sec:
        if label != "reject":
            label = "usable"
        reasons.append("edge_silence")

    keep_for_dataset = label != "reject"
    keep_for_reference = (
        label == "clean"
        and 1.5 <= duration_sec <= args.max_reference_duration
        and speech_ratio >= 0.70
        and snr_db_value >= 16.0
        and music_score <= 0.45
        and leading_silence_sec <= 0.24
        and trailing_silence_sec <= 0.24
        and overlap_sec <= 0.0
    )
    return label, keep_for_dataset, keep_for_reference, reasons


def main() -> int:
    args = parse_args()
    manifest_path = matched_manifest_path(args.namespace, args.run_id, args.episode_id)
    if not manifest_path.is_file():
        raise SystemExit(f"Matched clips manifest not found: {manifest_path}")

    target_json_path = resolve_path(args.target_json) if args.target_json else autodetect_target_json(args.namespace, args.episode_id)
    match_csv_path = speaker_matches_path(args.namespace, args.run_id)
    if not match_csv_path.is_file():
        raise SystemExit(f"Speaker match CSV not found: {match_csv_path}")

    matched_rows = load_csv_rows(manifest_path)
    match_rows = load_csv_rows(match_csv_path)
    target_json = json.loads(target_json_path.read_text())
    diarized_by_key = diarized_mapping(match_rows)
    transcript_segments = target_json.get("segments", [])

    output_rows: list[dict[str, str]] = []
    for row in matched_rows:
        clip_path = resolve_path(row["audio_path"])
        audio, sample_rate = sf.read(str(clip_path))
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        duration_sec = float(row["duration_sec"])
        clip_rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        clip_peak = float(np.max(np.abs(audio))) if len(audio) else 0.0

        vad = vad_metrics(audio, sample_rate, args.vad_aggressiveness)
        music = music_metrics(audio, sample_rate)

        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        key = (f"{start_sec:.3f}", f"{end_sec:.3f}", row["matched_speaker"].strip())
        diarized_speaker = diarized_by_key.get(key, "")
        overlap = overlap_metrics(start_sec, end_sec, diarized_speaker, transcript_segments)
        label, keep_for_dataset, keep_for_reference, reasons = classify_clip(
            args=args,
            duration_sec=duration_sec,
            speech_ratio=float(vad["speech_ratio"]),
            leading_silence_sec=float(vad["leading_silence_sec"]),
            trailing_silence_sec=float(vad["trailing_silence_sec"]),
            rms_dbfs_value=dbfs(clip_rms),
            snr_db_value=snr_db(float(vad["speech_rms"]), float(vad["noise_rms"])),
            music_score=float(music["music_score"]),
            overlap_sec=float(overlap["overlap_sec"]),
            nearest_other_gap_sec=float(overlap["nearest_other_gap_sec"]),
        )

        output_rows.append(
            {
                "matched_speaker": row["matched_speaker"],
                "clip_index": row["clip_index"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "duration_sec": row["duration_sec"],
                "text": row["text"],
                "audio_path": row["audio_path"],
                "diarized_speaker": diarized_speaker,
                "speech_ratio": f"{float(vad['speech_ratio']):.4f}",
                "leading_silence_sec": f"{float(vad['leading_silence_sec']):.3f}",
                "trailing_silence_sec": f"{float(vad['trailing_silence_sec']):.3f}",
                "rms_dbfs": f"{dbfs(clip_rms):.3f}",
                "peak_dbfs": f"{dbfs(clip_peak):.3f}",
                "snr_estimate_db": f"{snr_db(float(vad['speech_rms']), float(vad['noise_rms'])):.3f}",
                "music_score": f"{float(music['music_score']):.4f}",
                "harmonic_ratio": f"{float(music['harmonic_ratio']):.4f}",
                "spectral_flatness": f"{float(music['spectral_flatness']):.4f}",
                "centroid_ratio": f"{float(music['centroid_ratio']):.4f}",
                "overlap_sec": f"{float(overlap['overlap_sec']):.3f}",
                "nearest_other_gap_sec": f"{float(overlap['nearest_other_gap_sec']):.3f}",
                "overlap_speaker": str(overlap["overlap_speaker"]),
                "quality_label": label,
                "keep_for_dataset": "yes" if keep_for_dataset else "no",
                "keep_for_reference": "yes" if keep_for_reference else "no",
                "reasons": ",".join(reasons),
            }
        )

    output_path = matched_root(args.namespace, args.run_id, args.episode_id) / "clip-quality.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()) if output_rows else [])
        if output_rows:
            writer.writeheader()
            writer.writerows(output_rows)

    clean = sum(1 for row in output_rows if row["quality_label"] == "clean")
    usable = sum(1 for row in output_rows if row["quality_label"] == "usable")
    reject = sum(1 for row in output_rows if row["quality_label"] == "reject")
    print(f"wrote {len(output_rows)} clip quality rows to {output_path}")
    print(f"quality distribution: clean={clean} usable={usable} reject={reject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
