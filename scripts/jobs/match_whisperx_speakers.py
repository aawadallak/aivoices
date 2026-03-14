#!/usr/bin/env python3
"""Match diarized WhisperX speakers against canonical speakers using speaker embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match target WhisperX speaker embeddings against a reference episode speaker map."
    )
    parser.add_argument("--reference-json", help="Reference WhisperX JSON path.")
    parser.add_argument("--reference-map", help="Reference speaker-map.csv path.")
    parser.add_argument("--target-json", required=True, help="Target WhisperX JSON path.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Similarity threshold to mark a match as accepted. Default: 0.45.",
    )
    parser.add_argument(
        "--allowed-speakers",
        help="Optional comma-separated list of canonical speakers to consider during matching.",
    )
    parser.add_argument(
        "--reference-root",
        help="Optional references root. If set with --target-audio, runs segment-level matching from approved references.",
    )
    parser.add_argument(
        "--target-audio",
        help="Optional target audio path. Required with --reference-root for segment-level matching.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.8,
        help="Minimum segment duration for segment-level matching. Default: 0.8.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum segment duration for segment-level matching. Default: 15.0.",
    )
    return parser.parse_args()


def load_json(path_str: str) -> dict:
    path = Path(path_str).expanduser().resolve()
    return json.loads(path.read_text())


def load_map(path_str: str) -> list[dict[str, str]]:
    path = Path(path_str).expanduser().resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def average(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("No vectors to average.")
    length = len(vectors[0])
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(length)]


def canonical_embeddings(
    reference_json: dict,
    speaker_map_rows: list[dict[str, str]],
    allowed_speakers: set[str] | None,
) -> dict[str, list[float]]:
    ref_embeddings = reference_json.get("speaker_embeddings", {})
    grouped: dict[str, list[list[float]]] = {}

    for row in speaker_map_rows:
        diarized = row.get("diarized_speaker", "")
        canonical = row.get("canonical_speaker", "")
        if not diarized or not canonical:
            continue
        if allowed_speakers is not None and canonical not in allowed_speakers:
            continue
        emb = ref_embeddings.get(diarized)
        if not emb:
            continue
        grouped.setdefault(canonical, []).append(emb)

    return {speaker: average(vectors) for speaker, vectors in grouped.items()}


def write_rows(path_str: str, rows: list[dict[str, str]]) -> None:
    path = Path(path_str).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "target_diarized_speaker",
            "start_sec",
            "end_sec",
            "duration_sec",
            "matched_speaker",
            "similarity",
            "status",
            "text",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def approved_reference_paths(reference_root: Path, allowed_speakers: set[str] | None) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for speaker_dir in sorted(reference_root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name
        if allowed_speakers is not None and speaker not in allowed_speakers:
            continue
        catalog = speaker_dir / "references.csv"
        if not catalog.is_file():
            continue
        with catalog.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "approved":
                    continue
                file_path = row.get("file_path", "").strip()
                if not file_path:
                    continue
                grouped.setdefault(speaker, []).append(resolve_path(file_path))
    return grouped


def embed_file(encoder: VoiceEncoder, path: Path) -> list[float]:
    wav, sample_rate = sf.read(str(path))
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    return encoder.embed_utterance(preprocess_wav(wav, source_sr=sample_rate)).tolist()


def reference_embeddings_from_catalog(
    reference_root: Path,
    allowed_speakers: set[str] | None,
) -> dict[str, list[float]]:
    encoder = VoiceEncoder()
    grouped = approved_reference_paths(reference_root, allowed_speakers)
    embeddings: dict[str, list[float]] = {}
    for speaker, paths in grouped.items():
        vectors = [embed_file(encoder, path) for path in paths if path.is_file()]
        if vectors:
            embeddings[speaker] = average(vectors)
    return embeddings


def segment_level_rows(
    target_json: dict,
    target_audio: Path,
    canonical: dict[str, list[float]],
    threshold: float,
    min_duration: float,
    max_duration: float,
) -> list[dict[str, str]]:
    encoder = VoiceEncoder()
    wav, sample_rate = sf.read(str(target_audio))
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)

    rows: list[dict[str, str]] = []
    for segment in target_json.get("segments", []):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        duration = end - start
        text = " ".join((segment.get("text") or "").split())
        diarized = segment.get("speaker", "")
        if not diarized or not text:
            continue
        if duration < min_duration or duration > max_duration:
            continue

        start_frame = max(0, int(start * sample_rate))
        end_frame = min(len(wav), int(end * sample_rate))
        if end_frame <= start_frame:
            continue

        clip_wav = wav[start_frame:end_frame]
        if len(clip_wav) < sample_rate:
            continue

        embedding = encoder.embed_utterance(preprocess_wav(clip_wav, source_sr=sample_rate)).tolist()
        scored = [
            (speaker, cosine_similarity(embedding, can_emb))
            for speaker, can_emb in canonical.items()
        ]
        speaker, similarity = max(scored, key=lambda item: item[1])
        rows.append(
            {
                "target_diarized_speaker": diarized,
                "start_sec": f"{start:.3f}",
                "end_sec": f"{end:.3f}",
                "duration_sec": f"{duration:.3f}",
                "matched_speaker": speaker,
                "similarity": f"{similarity:.6f}",
                "status": "matched" if similarity >= threshold else "review",
                "text": text[:160],
            }
        )
    return rows


def main() -> int:
    target_json = load_json(args.target_json)
    allowed_speakers = None
    if args.allowed_speakers:
        allowed_speakers = {speaker.strip() for speaker in args.allowed_speakers.split(",") if speaker.strip()}

    if args.reference_root or args.target_audio:
        if not (args.reference_root and args.target_audio):
            raise SystemExit("--reference-root and --target-audio must be used together.")
        canonical = reference_embeddings_from_catalog(resolve_path(args.reference_root), allowed_speakers)
        if not canonical:
            raise SystemExit("No approved reference embeddings were found.")
        rows = segment_level_rows(
            target_json=target_json,
            target_audio=resolve_path(args.target_audio),
            canonical=canonical,
            threshold=args.threshold,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    else:
        if not (args.reference_json and args.reference_map):
            raise SystemExit("--reference-json and --reference-map are required unless using --reference-root with --target-audio.")
        reference_json = load_json(args.reference_json)
        speaker_map_rows = load_map(args.reference_map)
        canonical = canonical_embeddings(reference_json, speaker_map_rows, allowed_speakers)
        target_embeddings = target_json.get("speaker_embeddings", {})

        rows = []
        for target_speaker, emb in sorted(target_embeddings.items()):
            scored = [
                (speaker, cosine_similarity(emb, can_emb))
                for speaker, can_emb in canonical.items()
            ]
            speaker, similarity = max(scored, key=lambda item: item[1])
            rows.append(
                {
                    "target_diarized_speaker": target_speaker,
                    "start_sec": "",
                    "end_sec": "",
                    "duration_sec": "",
                    "matched_speaker": speaker,
                    "similarity": f"{similarity:.6f}",
                    "status": "matched" if similarity >= args.threshold else "review",
                    "text": "",
                }
            )

    write_rows(args.output_csv, rows)
    print(f"wrote {len(rows)} speaker matches to {Path(args.output_csv).resolve()}")
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main())
