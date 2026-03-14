# Metadata

Structured data about the source and processed media lives here.

Suggested structure:

- `metadata/<namespace>/transcripts/`: text or subtitle-derived transcripts
- `metadata/<namespace>/speakers/references/<speaker>/`: reusable reference samples per speaker
- `metadata/<namespace>/speakers/references/<speaker>/references.csv`: catalog of available references with IDs
- `metadata/<namespace>/speakers/runs/`: diarization and speaker-extraction run outputs
- `metadata/<namespace>/speakers/runs/<run-id>/<speaker>/extracted-clips/`: clips already extracted for a specific speaker
- `metadata/<namespace>/manifests/`: clip indexes and dataset manifests
- `metadata/<namespace>/qa/`: notes about filtering, exclusions, or quality checks

Metadata should describe both raw inputs and generated dataset assets.
