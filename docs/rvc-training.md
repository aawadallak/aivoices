# RVC v2 Training

This document covers training RVC v2 (Retrieval-based Voice Conversion) as a complement or alternative to the existing XTTS TTS pipeline.

## What Is RVC v2

RVC v2 is a voice conversion model. It transforms existing audio from one voice into another — it does not generate speech from text.

**Architecture:**
- **Feature extraction:** ContentVec (recommended) or HuBERT base encodes speech into speaker-independent representations. ContentVec is a fine-tuned HuBERT variant optimized for separating content from speaker identity — it produces better voice similarity than vanilla HuBERT
- **Pitch extraction:** RMVPE (Robust Model for Pitch Estimation in Variable Environments) replaces older Harvest/PM methods
- **Generator:** VITS-based model converts features into target-speaker audio
- **Retrieval augmentation:** FAISS index of training features is queried at inference to improve voice similarity

**Upstream repos:**
- [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [IAHispano/Applio](https://github.com/IAHispano/Applio) (actively maintained fork with CLI support)

## RVC v2 vs XTTS

| Aspect | XTTS v2 | RVC v2 |
|--------|---------|--------|
| **Type** | Text-to-speech | Voice conversion |
| **Input** | Text + speaker reference WAV | Source audio (any voice) |
| **Output** | Generated speech in target voice | Re-voiced source audio |
| **Transcripts needed** | Yes (pipe-delimited CSV) | No |
| **Training data** | 10-40 min with transcripts | 10-40 min clean audio only |
| **Training time** | Hours to days | Minutes to hours |
| **Model size** | ~1.8 GB | ~50-100 MB |
| **Best for** | New dialogue, narration, chatbots | Dubbing, singing, voice correction |
| **Multilingual** | Yes (zero-shot) | Per-language training |

## Dataset Requirements

RVC v2 trains directly from audio files. No transcripts or metadata CSVs are needed.

- **Minimum:** 10 minutes of clean speech
- **Recommended:** 20-40 minutes
- **Single-speaker only** — one model per voice
- **Format:** mono WAV, 16-bit (any sample rate — internally resampled to 40kHz or 48kHz)
- **Quality:** clean speech without background noise, music, or overlapping voices

The `clean`-quality clips already curated for XTTS exports are ideal candidates. The existing `datasets/<namespace>/exports/export_xtts/<export-name>/wavs/` directory can serve as the input audio pool directly.

## Dataset Preparation

Collect target speaker wavs into an RVC-ready directory. No metadata CSV required.

**Proposed layout:**

```text
datasets/<namespace>/exports/export_rvc/<voice>/
  wavs/
    <voice>-0001.wav
    <voice>-0002.wav
    ...
  dataset-info.json
```

**Reusing existing XTTS clips:**

```bash
# Copy curated wavs from an existing XTTS export
mkdir -p datasets/<namespace>/exports/export_rvc/<voice>/wavs

cp datasets/<namespace>/exports/export_xtts/<dataset-name>/wavs/*.wav \
   datasets/<namespace>/exports/export_rvc/<voice>/wavs/
```

Filter out clips shorter than 1 second — very short clips add noise to training:

```bash
# Remove clips under 1 second (requires soxi from sox)
for f in datasets/<namespace>/exports/export_rvc/<voice>/wavs/*.wav; do
  duration=$(soxi -D "$f" 2>/dev/null)
  if (( $(echo "$duration < 1.0" | bc -l) )); then
    rm "$f"
  fi
done
```

## Training Setup on Runpod

RVC v2 trains fast. A single A40, A100, or H100 is sufficient. Even consumer GPUs (RTX 3090/4090) work well.

**Base image:** same as XTTS training — `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`

**Recommended tool: Applio** — fork-friendly, CLI support, actively maintained.

### Install Applio

```bash
cd /workspace
git clone https://github.com/IAHispano/Applio.git
cd Applio
pip install -r requirements.txt
```

### Download Pretrained Models

RVC v2 needs pretrained base models for feature extraction and pitch estimation.

```bash
cd /workspace/Applio

# ContentVec (recommended feature extractor — fine-tuned HuBERT for voice conversion)
mkdir -p assets/hubert
wget -O assets/hubert/hubert_base.pt \
  https://huggingface.co/contentvec/contentvec/resolve/main/checkpoint_best_legacy_500.pt
# Note: Applio expects the file at assets/hubert/hubert_base.pt regardless of which extractor you use.
# If you prefer vanilla HuBERT instead (not recommended):
#   wget -O assets/hubert/hubert_base.pt \
#     https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt

# RMVPE (pitch extraction)
mkdir -p assets/rmvpe
wget -O assets/rmvpe/rmvpe.pt \
  https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt

# RVC v2 pretrained generator + discriminator (48kHz, f0)
mkdir -p assets/pretrained_v2
wget -O assets/pretrained_v2/f0G48k.pth \
  https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth
wget -O assets/pretrained_v2/f0D48k.pth \
  https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth
```

## Training Workflow

### 1. Preprocess

Resample and slice audio into training segments.

```bash
python rvc/train/preprocess/preprocess.py \
  --model_name <voice> \
  --dataset_path /workspace/datasets/<voice>/wavs \
  --sample_rate 48000
```

### 2. Extract Features

Extract pitch (f0) and ContentVec embeddings.

```bash
# Pitch extraction with RMVPE
python rvc/train/extract/extract_f0_rmvpe.py \
  --model_name <voice>

# ContentVec feature extraction
python rvc/train/extract/extract_feature_print.py \
  --model_name <voice>
```

### 3. Train

Typical settings: 200-500 epochs, batch size 8-16, save every 50 epochs.

```bash
python rvc/train/train.py \
  --model_name <voice> \
  --sample_rate 48000 \
  --total_epoch 300 \
  --batch_size 8 \
  --save_every_epoch 50 \
  --pretrained_G assets/pretrained_v2/f0G48k.pth \
  --pretrained_D assets/pretrained_v2/f0D48k.pth
```

Training loss should decrease steadily. Overfitting typically appears after 400-500 epochs — listen to intermediate checkpoints.

### 4. Generate FAISS Index

Build the retrieval index from extracted features. This enables the retrieval augmentation at inference.

```bash
python rvc/train/process/extract_index.py \
  --model_name <voice>
```

### 5. Verify Outputs

Confirm these artifacts exist:

```text
logs/<voice>/
  G_300.pth                    # generator checkpoint (the model)
  D_300.pth                    # discriminator checkpoint (not needed for inference)
  added_<voice>.index          # FAISS retrieval index
  total_fea.npy                # extracted feature vectors
```

## Training Output Layout

Follow the existing namespace convention:

```text
training/rvc/<namespace>/<voice>/<run-id>/
  run.json
  model/
    G_<epoch>.pth
    added_<voice>.index
  logs/
    train.log
```

`run.json` should record: namespace, voice, epoch count, sample rate, dataset source, timestamp. Same schema spirit as XTTS `run.json`.

## Inference

Convert a source audio file to the target voice.

```bash
python rvc/infer/infer.py \
  --model_path training/rvc/<namespace>/<voice>/<run-id>/model/G_300.pth \
  --index_path training/rvc/<namespace>/<voice>/<run-id>/model/added_<voice>.index \
  --input_path /path/to/source-audio.wav \
  --output_path /path/to/converted-output.wav \
  --f0_method rmvpe \
  --index_rate 0.75 \
  --filter_radius 3 \
  --resample_sr 0 \
  --rms_mix_rate 0.25 \
  --protect 0.33
```

### Key Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `index_rate` | 0.0–1.0 | 0.75 | Retrieval strength. Higher = more target voice character |
| `f0_method` | rmvpe, crepe | rmvpe | Pitch extraction algorithm. RMVPE is best for speech |
| `protect` | 0.0–0.5 | 0.33 | Protects voiceless consonants from artifacts. Lower = more protection |
| `rms_mix_rate` | 0.0–1.0 | 0.25 | Envelope mixing. 0 = source loudness, 1 = target loudness |
| `filter_radius` | 0–7 | 3 | Median filter on pitch. Higher = smoother pitch but less expression |

Start with `index_rate=0.75` and adjust — this is the most impactful parameter.

## Hybrid Pipeline: XTTS + RVC v2

The combination uses XTTS for text-to-speech generation, then RVC v2 to refine the voice timbre.

**Pipeline:** text → XTTS inference → intermediate WAV → RVC inference → final WAV

**When this helps:**
- XTTS captures prosody and language well but the voice timbre is imperfect
- RVC corrects the timbre while preserving XTTS prosody and intonation

**When this does not help:**
- XTTS output already sounds correct
- XTTS output has artifacts (clicks, garbled segments) that RVC will amplify

**Automated smoke review (recommended):**

```bash
# Generates XTTS samples from all checkpoint candidates, then converts each through RVC
python scripts/jobs/export_hybrid_smoke_review.py \
  --namespace <namespace> --voice <voice> \
  --xtts-run-id <xtts-run-id> \
  --rvc-run-id <rvc-run-id> \
  --speaker-wav metadata/<namespace>/speaker-references/<voice>-reference-01-seed.wav \
  --index-rate 0.75 \
  --include-milestone
```

**Manual single-file conversion:**

```bash
# Step 1: Generate with XTTS (using an existing smoke review sample or custom text)
# Step 2: Convert with RVC
python rvc/infer/infer.py \
  --model_path training/rvc/<namespace>/<voice>/<run-id>/model/G_300.pth \
  --index_path training/rvc/<namespace>/<voice>/<run-id>/model/added_<voice>.index \
  --input_path /tmp/xtts-raw.wav \
  --output_path /tmp/final.wav \
  --f0_method rmvpe \
  --index_rate 0.75
```

## Integration with aivoices Pipeline

RVC artifacts follow the same `<namespace>/<voice>` convention used throughout the project.

**Dataset reuse:** existing curated clips from XTTS exports can be used directly — no re-curation needed.

**R2 transport:** RVC models are small (~50-100 MB). They can be published/fetched with the same rclone pattern used for XTTS datasets:

```bash
# Upload
rclone copy training/rvc/<namespace>/<voice>/<run-id>/model/ \
  r2:aivoices/training/rvc/<namespace>/<voice>/<run-id>/model/

# Download
rclone copy r2:aivoices/training/rvc/<namespace>/<voice>/<run-id>/model/ \
  training/rvc/<namespace>/<voice>/<run-id>/model/
```

**Implemented jobs:**
- `scripts/jobs/export_rvc_dataset.py` — collect clean wavs into an RVC-ready directory
- `scripts/jobs/train_rvc.py` — wrapper around RVC training with run tracking and `run.json` generation
- `scripts/jobs/fetch_rvc_dataset.py` / `publish_rvc_dataset.py` — R2 dataset transport
- `scripts/jobs/export_hybrid_smoke_review.py` — hybrid XTTS+RVC smoke test

**Hybrid smoke review:** generates XTTS samples from each checkpoint candidate, then converts each through the trained RVC model. Outputs both raw XTTS and RVC-converted samples side-by-side for A/B comparison.

```bash
python scripts/jobs/export_hybrid_smoke_review.py \
  --namespace <ns> --voice <voice> \
  --xtts-run-id <xtts-run-id> \
  --rvc-run-id <rvc-run-id> \
  --speaker-wav <reference.wav> \
  --index-rate 0.75
```

Output layout per XTTS checkpoint:

```text
training/xtts/<ns>/<voice>/<xtts-run-id>/
  hybrid-candidates.json
  hybrid-samples/
    <checkpoint-id>/
      xtts/sample-01.wav     # raw XTTS
      xtts/sample-02.wav
      rvc/sample-01.wav      # XTTS → RVC
      rvc/sample-02.wav
      sample-manifest.csv    # links both side-by-side
```

## Notes

- RVC v2 works best with clean, noise-free audio. The existing clip-quality scoring pipeline is directly beneficial.
- For pt-BR content, RMVPE handles Portuguese intonation well.
- `index_rate` tuning is the most impactful inference knob — start at 0.75 and adjust by ear.
- Trim silence from training clips. RVC does not handle long silences or non-speech segments well.
- Model files are small (~50-100 MB for the generator + index). They can be stored in R2 or even committed to git if needed.
- Training 300 epochs on 30 minutes of audio typically takes 15-30 minutes on an H100.
- Save checkpoints every 50 epochs and audition intermediate results — overfitting degrades quality noticeably.
- ContentVec generally produces better voice similarity than HuBERT base. Use ContentVec unless you have a specific reason not to.
