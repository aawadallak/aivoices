# Setup

## Purpose

This document records the environment setup required to prepare and process audio datasets in this repository.

## Current Local Environment

- Repository root: `/home/awadallak/aivoices`
- Virtual environment path: `/home/awadallak/aivoices/.venv-dataset`
- Source media root: `/home/awadallak/aivoices/episodes`

## System Packages

Install the base system packages with:

```bash
sudo apt update
sudo apt install -y ffmpeg python3 python3-venv python3-pip git
```

## Python Environment

Create and activate the virtual environment with:

```bash
python3 -m venv /home/awadallak/aivoices/.venv-dataset
source /home/awadallak/aivoices/.venv-dataset/bin/activate
python -m pip install --upgrade "pip<24.1" setuptools wheel
```

## Verified Python Baseline

This workspace is currently working with:

- `Python 3.11.15`
- `pip 24.0`

Confirm the interpreter version before reinstalling on another machine:

```bash
python --version
pip --version
```

If the interpreter is Python 3.13, recreate the virtual environment with Python 3.11 before retrying the install.

## Current Working ML Stack

The following packages are currently installed in `.venv-dataset` and import successfully:

- `torch 2.8.0+cu128`
- `torchaudio 2.8.0`
- `torchvision 0.23.0`
- `whisperx 3.8.2`
- `pyannote.audio 4.0.4`
- `resemblyzer 0.1.4`
- `librosa 0.11.0`
- `soundfile 0.13.1`
- `pydub 0.25.1`

`resemblyzer` also required keeping:

- `setuptools 80.10.2`

because newer `setuptools` removed `pkg_resources`, which `resemblyzer` still imports.

## Install Command Used

After the environment was already on Python 3.11 with a working `torch` baseline, this command was used:

```bash
source /home/awadallak/aivoices/.venv-dataset/bin/activate
pip install whisperx pyannote.audio resemblyzer pydub soundfile librosa
pip install torchvision==0.23.0
pip install "setuptools<81"
```

## Important Compatibility Notes

- installing `whisperx 3.8.2` upgraded the environment from `torch 2.5.1+cu121` to `torch 2.8.0+cu128`
- `torchvision 0.20.1+cu121` became incompatible after that upgrade and had to be replaced with `torchvision 0.23.0`
- `resemblyzer` failed until `setuptools` was pinned below `81`

## Current Warnings And Limits

- reported torch build is `2.8.0+cu128`
- GPU availability is currently intermittent on this host:
  - some validation passes detect the RTX 3060 Ti correctly
  - other passes fail with `nvidia-smi: Failed to initialize NVML: Unknown Error`
  - the latest validation in `.venv-dataset` returned `torch.cuda.is_available() == False`
- `pyannote.audio` imports successfully but warns that `torchcodec` is not correctly usable in this environment
- because of that warning, prefer feeding `pyannote.audio` audio that was already extracted with `ffmpeg` instead of relying on built-in decoding

## Historical Failure

The original install command used:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

That earlier attempt failed with:

```text
ERROR: Could not find a version that satisfies the requirement torch
ERROR: No matching distribution found for torch
```

That failure no longer reflects the current state of the workspace, but it is documented here because it can happen again on a fresh machine with the wrong interpreter or wheel combination.
