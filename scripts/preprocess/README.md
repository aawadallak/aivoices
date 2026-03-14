# Preprocess Scripts

## `extract_audio.py`

Extracts audio from files in `episodes/<series>/raw/` into `episodes/<series>/extracted-audio/`.

If `--speaker` and `--run-id` are provided together, the output is redirected to:

`metadata/<series>/speakers/runs/<run-id>/<speaker>/extracted-audio/`

Examples:

```bash
python scripts/preprocess/extract_audio.py --series chaves
python scripts/preprocess/extract_audio.py --series dragonball --format mp3
python scripts/preprocess/extract_audio.py --input episodes/chaves/raw/example.mp4 --dry-run
python scripts/preprocess/extract_audio.py --series chaves --speaker seu-madruga --run-id 2026-03-13-pass-01
```

Default output settings are tuned for speech workflows:

- format: `wav`
- sample rate: `16000`
- channels: `1`

Notes:

- use `--speaker` and `--run-id` together
- use speaker slugs in lowercase with hyphens
