#!/usr/bin/env python3
"""Conservatively match diarized segments against approved speaker references."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = REPO_ROOT / "metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match diarized segments against approved references with conservative thresholds."
    )
    parser.add_argument("--namespace", "--series", dest="namespace", required=True, help="Namespace slug.")
    parser.add_argument("--target-json", required=True, help="Target WhisperX JSON path.")
    parser.add_argument("--target-audio", required=True, help="Target extracted audio path.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--allowed-speakers",
        required=True,
        help="Comma-separated canonical speakers to consider, for example bob-esponja,lula-molusco.",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Number of best reference scores to average. Default: 2.")
    parser.add_argument(
        "--mean-weight",
        type=float,
        default=0.4,
        help="Weight for mean-speaker embedding score. Default: 0.4.",
    )
    parser.add_argument(
        "--topk-weight",
        type=float,
        default=0.6,
        help="Weight for top-k reference score. Default: 0.6.",
    )
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=0.72,
        help="Minimum combined score for matched. Default: 0.72.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.62,
        help="Minimum combined score for review. Default: 0.62.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.08,
        help="Minimum score margin between best and second speaker for matched. Default: 0.08.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.2,
        help="Minimum segment duration in seconds. Default: 1.2.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=12.0,
        help="Maximum segment duration in seconds. Default: 12.0.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_json(path_str: str) -> dict:
    path = resolve_path(path_str)
    return json.loads(path.read_text())


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


def approved_reference_paths(namespace: str, speakers: set[str]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    root = METADATA_DIR / namespace / "speakers" / "references"
    for speaker in sorted(speakers):
        catalog = root / speaker / "references.csv"
        if not catalog.is_file():
            continue
        with catalog.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "approved":
                    continue
                file_path = row.get("file_path", "").strip()
                if file_path:
                    grouped.setdefault(speaker, []).append(resolve_path(file_path))
    return grouped


def load_reference_bank(namespace: str, speakers: set[str]) -> dict[str, dict[str, object]]:
    encoder = VoiceEncoder()
    bank: dict[str, dict[str, object]] = {}
    grouped = approved_reference_paths(namespace, speakers)
    for speaker, paths in grouped.items():
        embeddings: list[list[float]] = []
        for path in paths:
            if not path.is_file():
                continue
            wav, sample_rate = sf.read(str(path))
            if getattr(wav, "ndim", 1) > 1:
                wav = wav.mean(axis=1)
            embedding = encoder.embed_utterance(preprocess_wav(wav, source_sr=sample_rate)).tolist()
            embeddings.append(embedding)
        if embeddings:
            bank[speaker] = {
                "paths": paths,
                "embeddings": embeddings,
                "mean_embedding": average(embeddings),
            }
    return bank


def score_speaker(
    segment_embedding: list[float],
    speaker_bank: dict[str, object],
    top_k: int,
    mean_weight: float,
    topk_weight: float,
) -> tuple[float, float, float]:
    mean_score = cosine_similarity(segment_embedding, speaker_bank["mean_embedding"])  # type: ignore[index]
    all_scores = [
        cosine_similarity(segment_embedding, embedding)
        for embedding in speaker_bank["embeddings"]  # type: ignore[index]
    ]
    top_scores = sorted(all_scores, reverse=True)[: max(1, top_k)]
    topk_score = sum(top_scores) / len(top_scores)
    combined = (mean_weight * mean_score) + (topk_weight * topk_score)
    return combined, mean_score, topk_score


def decision_reason(
    combined: float,
    margin: float,
    accept_threshold: float,
    review_threshold: float,
    min_margin: float,
) -> tuple[str, str]:
    if combined >= accept_threshold and margin >= min_margin:
        return "matched", "high_confidence"
    if combined >= accept_threshold and margin < min_margin:
        return "review", "low_margin"
    if combined >= review_threshold:
        return "review", "below_accept_threshold"
    return "reject", "low_score"


def write_rows(path_str: str, rows: list[dict[str, str]]) -> None:
    path = resolve_path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_diarized_speaker",
                "start_sec",
                "end_sec",
                "duration_sec",
                "matched_speaker",
                "best_score",
                "best_mean_score",
                "best_topk_score",
                "second_speaker",
                "second_score",
                "margin",
                "status",
                "reason",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    target_json = load_json(args.target_json)
    target_audio = resolve_path(args.target_audio)
    if not target_audio.is_file():
        raise SystemExit(f"Target audio not found: {target_audio}")

    speakers = {speaker.strip() for speaker in args.allowed_speakers.split(",") if speaker.strip()}
    if len(speakers) < 2:
        raise SystemExit("Use at least two speakers in --allowed-speakers.")

    bank = load_reference_bank(args.namespace, speakers)
    missing = sorted(speakers - set(bank))
    if missing:
        raise SystemExit(f"No approved references found for: {', '.join(missing)}")

    encoder = VoiceEncoder()
    wav, sample_rate = sf.read(str(target_audio))
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)

    rows: list[dict[str, str]] = []
    for segment in target_json.get("segments", []):
        diarized = segment.get("speaker", "")
        text = " ".join((segment.get("text") or "").split())
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        duration = end - start

        if not diarized:
            continue
        if not text:
            continue
        if duration < args.min_duration or duration > args.max_duration:
            continue

        start_frame = max(0, int(start * sample_rate))
        end_frame = min(len(wav), int(end * sample_rate))
        if end_frame <= start_frame:
            continue

        clip_wav = wav[start_frame:end_frame]
        if len(clip_wav) < int(sample_rate * args.min_duration):
            continue

        segment_embedding = encoder.embed_utterance(preprocess_wav(clip_wav, source_sr=sample_rate)).tolist()
        scored = []
        for speaker, speaker_bank in bank.items():
            combined, mean_score, topk_score = score_speaker(
                segment_embedding=segment_embedding,
                speaker_bank=speaker_bank,
                top_k=args.top_k,
                mean_weight=args.mean_weight,
                topk_weight=args.topk_weight,
            )
            scored.append((speaker, combined, mean_score, topk_score))

        scored.sort(key=lambda item: item[1], reverse=True)
        best_speaker, best_score, best_mean_score, best_topk_score = scored[0]
        second_speaker, second_score, _, _ = scored[1]
        margin = best_score - second_score
        status, reason = decision_reason(
            combined=best_score,
            margin=margin,
            accept_threshold=args.accept_threshold,
            review_threshold=args.review_threshold,
            min_margin=args.min_margin,
        )

        rows.append(
            {
                "target_diarized_speaker": diarized,
                "start_sec": f"{start:.3f}",
                "end_sec": f"{end:.3f}",
                "duration_sec": f"{duration:.3f}",
                "matched_speaker": best_speaker,
                "best_score": f"{best_score:.6f}",
                "best_mean_score": f"{best_mean_score:.6f}",
                "best_topk_score": f"{best_topk_score:.6f}",
                "second_speaker": second_speaker,
                "second_score": f"{second_score:.6f}",
                "margin": f"{margin:.6f}",
                "status": status,
                "reason": reason,
                "text": text[:160],
            }
        )

    write_rows(args.output_csv, rows)
    print(f"wrote {len(rows)} segment decisions to {resolve_path(args.output_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
