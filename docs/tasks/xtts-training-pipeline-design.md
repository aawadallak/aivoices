# XTTS Training Pipeline Design

## Status

Proposed

## Goal

Design a first-party XTTS training pipeline inside this repository, using the existing dataset preparation flow and the best ideas from the external training code already reviewed.

This task is for design and decision-making only. Do not implement the training pipeline yet.

## Why This Exists

The repository already covers:

- source ingestion
- audio extraction
- diarization
- reference matching
- clip QA
- provisional dataset export
- XTTS-ready dataset export

What is still missing is the training layer that consumes those XTTS exports directly and produces repeatable fine-tune runs.

## Inputs We Already Have

- XTTS-ready exports under `datasets/<namespace>/exports/export_xtts/<export-name>/`
- `metadata_train.csv`
- `metadata_eval.csv`
- `wavs/`
- approved speaker references under `metadata/<namespace>/speakers/references/`
- stable namespace model across `episodes/`, `metadata/`, and `datasets/`

## Deployment Reality

Dataset preparation usually runs on bare metal in this repository.

Training usually runs later on ephemeral cloud GPU VMs. In practice, this means:

- the repository code may be cloned on the VM
- the dataset itself should be fetched separately
- repeated manual environment rebuilds are expensive
- dataset transport should not depend on the full local workspace being present

The design should optimize for this operational model.

Current preferred execution mode:

- container-first for training on ephemeral cloud VMs
- direct host execution only as fallback for debugging or emergency cases
- `Python 3.11.9` as the current operational baseline for first training runs

## Reference Material Already Reviewed

Two external code paths were reviewed and should inform the design:

1. A `train_gpt(...)` engine with:
- dataset validation
- text-length filtering
- eval fallback
- checkpoint restore / resume logic
- XTTS base file download
- Coqui `formatter="coqui"` dataset config
- `GPTTrainerConfig`, `SpeakerManager`, and `Trainer`

2. A CLI wrapper with:
- explicit argparse surface
- training hyperparameters
- restore / resume options
- max audio and max text controls

The direction is to combine the stronger engine ideas from the first with the operational usability of the second.

## Desired Outcome

Define a repository-native XTTS training workflow with:

- one obvious entrypoint
- stable output layout
- documented storage behavior
- resume support
- compatibility with the XTTS exports already generated here
- portability to ephemeral remote GPU VMs

## Proposed Scope

Design the following pieces:

- training engine module
- CLI entrypoint
- dataset fetch contract for remote VMs
- output directory layout
- checkpoint and resume behavior
- environment and dependency expectations
- logging policy
- storage and git policy for training outputs
- minimum validation performed before training starts

## Proposed File Layout

This is a starting point for discussion, not a final decision:

```text
scripts/jobs/train_xtts.py
scripts/pipeline/xtts_training.py
docs/training.md
training/xtts/<namespace>/<speaker>/<run-id>/
```

The training input should be a portable XTTS package, not a dependency on the full repository tree.

## Current Direction

These points now have a preferred direction and should be treated as the working baseline for the rest of the design discussion.

### Dataset Distribution

- XTTS datasets are prepared on bare metal
- XTTS datasets are published to Cloudflare R2
- R2 public access is the preferred distribution method for remote pre-training fetches
- `scp` is fallback only

Current preferred remote prefix convention:

- `training/datasets/<namespace>/<voice>/<dataset-name>/`
- `training/runs/<namespace>/<voice>/<run-id>/`
- `training/models/<namespace>/<voice>/current.json`

This keeps namespace isolation explicit and avoids collisions between similarly named voices from different content groups.

### Training Input Contract

The canonical trainer input should be:

- `--dataset-dir /path/to/xtts-package`

The first version should not require local resolution through:

- `--namespace`
- `--speaker`
- `--source-export`

Those values can still exist as metadata, but the trainer should consume a self-contained package after it has already been downloaded.

### Remote VM Workflow

Preferred flow:

1. clone the repository for code
2. fetch the XTTS package from R2
3. run the trainer against `--dataset-dir`

This keeps:

- git for code
- R2 for datasets
- training outputs outside git

### Checkpoint Retention

Checkpoint handling should be promotion-based, not full-history export.

Working policy:

- keep frequent checkpoints locally during the run for resume safety
- export only a small promoted set remotely
- never upload the full checkpoint history by default

Preferred retained outputs:

- `best`
- `last`
- sample audios used for evaluation
- run metadata and config

Do not promote:

- all intermediate checkpoints
- large local logs or temporary trainer state unless explicitly needed

### Smoke Test Is Mandatory

Standardized inference samples are a required part of checkpoint promotion.

Working policy:

- every comparable training run uses the same fixed smoke-test phrases
- candidate checkpoints must generate smoke-test audio before promotion
- checkpoint promotion is based on both training signals and listening evaluation
- a checkpoint without smoke-test outputs is not eligible to become the promoted `best`
- smoke-test review is expected to happen outside the training VM

Because the training VMs are not the place where final listening review happens, the workflow should explicitly support:

- generating smoke samples on the VM
- uploading those samples to R2
- reviewing them remotely
- promoting the chosen checkpoint after external review

The smoke-test phrase set should be versioned in the repository so that comparisons stay stable across runs.

Current proposed canonical file:

- `metadata/tts/smoke-tests/pt-br-default-v1.txt`

The first version should target a small, fixed set such as:

- 5 to 10 phrases
- same phrases for every run of the same language
- broad enough to expose pronunciation, pacing, punctuation, and stability issues

### Promotion Manifests

Promotion should be driven by small explicit manifest files, not by ad-hoc terminal instructions.

The current design direction is to use three manifests:

#### `run.json`

Written by the training run itself.

Purpose:

- identify the run
- record dataset and environment context
- point to smoke-test configuration

Suggested fields:

- `run_id`
- `voice`
- `dataset_name`
- `dataset_uri`
- `language`
- `started_at`
- `finished_at`
- `status`
- `notes`

Current preferred `run_id` format:

- `<voice>-<dataset-name>-<yyyymmdd-hhmmss>`

Example:

- `bob-esponja-bob-esponja-clean-v1-20260314-154500`

Current preferred first-version posture:

- keep `run.json` intentionally lean
- only include fields needed to identify the run and its dataset context
- defer richer execution metadata until a real consumer needs it

### Pre-Training Validation Baseline

The first version should run a short mandatory preflight before any GPU-heavy training starts.

Fail hard on:

- missing `dataset-dir`
- missing `metadata_train.csv`
- missing `metadata_eval.csv`
- missing `wavs/`
- malformed CSV
- missing required columns
- zero valid train rows
- referenced audio files that do not exist

Warn, but do not necessarily fail, on:

- empty eval CSV
- unusually long text
- unusually long audio clips
- sample-rate inconsistency
- more than one speaker label in a run that is expected to be single-speaker

Required CSV columns for the first version:

- `audio_file`
- `text`
- `speaker_name`

The goal is to catch obvious packaging errors before burning GPU time.

#### `candidates.json`

Written by the training/export-review stage.

Purpose:

- list candidate checkpoints for listening review
- describe which smoke samples belong to each checkpoint

Suggested fields:

- `run_id`
- `generated_at`
- `candidates`

Each candidate entry should include:

- `checkpoint_id`
- `step`
- `kind`
- `sample_dir`
- `sample_manifest`
- `metrics`
- `status`

Typical `kind` values:

- `best_loss`
- `last`
- `milestone`

Typical `status` values:

- `pending`
- `approved`
- `rejected`

Current preferred candidate policy:

- always include `last`
- always include `best_loss` when it differs from `last`
- optionally include one intermediate checkpoint when there is a clear reason
- cap the review set at `3` candidates per run

This keeps smoke generation, upload cost, and human review manageable.

#### `promotion.json`

Written after external review.

Purpose:

- tell the promote step which checkpoint becomes `best`
- keep an explicit audit trail of the human decision

Suggested fields:

- `run_id`
- `promote_checkpoint`
- `keep_last`
- `reviewed_by`
- `reviewed_at`
- `notes`

The training VM or a follow-up promotion job should read `promotion.json` and then publish the final promoted artifacts.

### Remote Run Layout

The current preferred direction is to keep all run-level review and promotion files together under the same `run-id` prefix in R2.

Suggested layout:

```text
training/runs/<voice>/<run-id>/
  run.json
  candidates.json
  promotion.json
  samples/
  artifacts/
    last/
    best/
```

This keeps:

- run metadata
- review inputs
- promotion decision
- promoted artifacts

in one place.

To avoid unnecessary storage duplication, the current preferred direction is:

- keep `best` and `last` only under the run prefix
- do not duplicate `last` into a separate models area
- do not duplicate checkpoints into a promoted-models area by default
- use a lightweight model manifest to point at the promoted run artifact instead

### Promotion Trigger

Promotion should happen by explicit command, not automatically when `promotion.json` appears.

Reasoning:

- avoids accidental promotion
- makes review and approval auditable
- allows correction of `promotion.json` before any irreversible publish step
- works better with ephemeral VMs and separate follow-up jobs

### Models Area

The current preferred direction is for `training/models/` to remain lightweight.

Instead of copying checkpoints again, it should store a small manifest that points to the promoted artifact inside the corresponding run.

Suggested shape:

```text
training/models/<voice>/
  current.json
```

Example intent:

- `current.json` identifies the currently promoted run
- `current.json` points to the `artifacts/best/` path under that run
- actual checkpoint payload remains stored only once, under `training/runs/<voice>/<run-id>/artifacts/best/`

Suggested `current.json` fields:

- `voice`
- `run_id`
- `checkpoint_kind`
- `artifact_path`
- `dataset_name`
- `language`
- `promoted_at`
- `notes`

### Open Technical Validation: Resume Artifact Requirements

This item must be confirmed before implementation is considered complete.

Current source-based reading from the Coqui Trainer code:

- checkpoint files written by the trainer include:
  - `config`
  - `model`
  - `optimizer`
  - `scaler`
  - `step`
  - `epoch`
- trainer restore logic loads:
  - model state
  - optimizer state
  - scaler state when present
  - step and epoch
- `continue_path` expects a training directory and reads:
  - `config.json` from that directory
  - the latest checkpoint from that directory
- scheduler state does not appear to be restored explicitly in the Trainer code path reviewed so far

Working implication for design:

- `last/` almost certainly needs the checkpoint file itself
- `last/` likely also needs a compatible `config.json`
- optimizer and scaler state appear to live inside the checkpoint payload, not in separate files
- scheduler behavior still needs explicit empirical verification during implementation

Current working assumption:

- `best/` can remain a minimal promoted artifact for inference and evaluation
- `last/` must be treated as a resume artifact and validated with a real interrupted-and-resumed training run before its contract is finalized

Current preferred resume direction:

- use `continue_path` as the primary resume mechanism
- treat `restore_path` as an advanced override, not the default workflow

Reasoning:

- `continue_path` matches the operational idea of resuming a run directory
- it reduces the chance of missing companion files such as `config.json`
- it fits better with the idea that `artifacts/last/` should behave like a portable resume bundle

Working implication:

- `artifacts/last/` should be designed around reliable `continue_path` usage
- `restore_path` can still be exposed for special cases, but it should not define the main artifact contract

Working distinction:

- `resume` means continuing the same run from `last`
- `derived run` means starting a new run from a previous checkpoint

Example:

- train for `10` epochs, then continue to `30` or `50` with the same run intent: `resume`
- start a new experiment from a previous checkpoint with changed settings or a different training plan: `derived run`

Current provisional `artifacts/last/` layout:

```text
artifacts/last/
  config.json
  checkpoint_<step>.pth
  manifest.json
```

The current baseline assumption is that a flat layout is safer until a real implementation proves that a nested layout is equally compatible with the chosen Trainer flow.

Suggested `manifest.json` fields:

- `run_id`
- `voice`
- `kind`
- `step`
- `checkpoint_file`
- `dataset_name`
- `language`
- `resume_mode`
- `created_at`
- `notes`

Current provisional `artifacts/best/` layout:

```text
artifacts/best/
  config.json
  checkpoint_<step>.pth
  manifest.json
```

The current preferred direction is to keep the external shape of `best/` similar to `last/`, while keeping the meaning different:

- `last/` exists for resume continuity
- `best/` exists for promotion, evaluation, and downstream consumption

Suggested `manifest.json` fields for `best/`:

- `run_id`
- `voice`
- `kind`
- `step`
- `checkpoint_file`
- `dataset_name`
- `language`
- `promoted_at`
- `notes`

## Questions To Resolve

1. Where should the training engine live?
- `scripts/pipeline/xtts_training.py`
- `src/...`
- another structure

2. What should be the canonical CLI?
- `scripts/jobs/train_xtts.py`
- one command per run
- pipeline-driven orchestration

3. What should the minimum required CLI arguments be?
- `--dataset-dir`
- `--output-dir`
- `--language`
- `--run-id` or auto-generated equivalent

4. How should runs be named?
- user-supplied `run-id`
- auto-generated timestamp
- speaker plus dataset slug

5. What should happen if `metadata_eval.csv` is empty?
- fail hard
- reuse train as eval
- auto-resplit

6. Should the trainer consume only XTTS-ready packages, or also allow direct train/eval CSV paths as an advanced mode?

7. Should the pipeline support one speaker per run only, or multispeaker later?

8. Where should base XTTS model files live locally?
- shared cache outside outputs
- per-run copy
- dedicated repository-level cache directory

9. What should be stored remotely after training?
- checkpoints
- final promoted checkpoint only
- config plus best checkpoint
- logs or no logs

10. Should we design for container-first training later, even if the first version still runs directly on the VM host?
Current preferred answer:

- no
- the first implementation should already target container-first execution

11. Where should the canonical smoke-test phrase files live?
- `metadata/tts/smoke-tests/`
- `scripts/assets/`
- another location

12. Which checkpoints should generate smoke-test outputs automatically?
- every save step
- best-loss candidates only
- end-of-run shortlist such as `best`, `last`, and one intermediate candidate

13. What exact remote prefix conventions should we use for:
- datasets
- runs
- promoted models

14. Do we want `current.json` to stay minimal, or add extra duplicated context later only if a real consumer needs it?

15. After an implementation exists, can we prove resume correctness with a real interruption test using the chosen XTTS training flow?

## Design Constraints

- keep raw media, bulk clips, and large outputs out of git
- prefer one obvious entrypoint per responsibility
- training outputs must be reproducible from stored exports and documented settings
- paths and conventions must align with existing namespace structure
- output directories must be safe to sync to remote storage later
- remote transfer costs matter
- ephemeral VMs make environment rebuild and checkpoint sprawl expensive
- checkpoint promotion must be audibly reviewable, not loss-only
- final listening review may happen outside the training VM

## Non-Goals

- implementing the trainer now
- choosing final hyperparameters for every future run
- adding Demucs or other enhancement logic to training itself
- building a multi-stage experiment tracking platform

## Acceptance Criteria

This task is done when we have agreement on:

- canonical file layout
- CLI contract
- dataset distribution contract
- output layout
- resume behavior
- checkpoint promotion policy
- mandatory smoke-test policy
- validation behavior
- dependency strategy
- storage policy for checkpoints and logs
- follow-up implementation task split

## Follow-Up

After this design is agreed, create implementation tasks for:

- training engine
- CLI entrypoint
- docs
- storage helpers for training outputs

Current decomposition:

- [`docs/tasks/xtts-training-runtime.md`](/home/awadallak/aivoices/docs/tasks/xtts-training-runtime.md)
- [`docs/tasks/xtts-training-trainer-core.md`](/home/awadallak/aivoices/docs/tasks/xtts-training-trainer-core.md)
- [`docs/tasks/xtts-training-smoke-review.md`](/home/awadallak/aivoices/docs/tasks/xtts-training-smoke-review.md)
- [`docs/tasks/xtts-training-promotion.md`](/home/awadallak/aivoices/docs/tasks/xtts-training-promotion.md)
- [`docs/tasks/xtts-training-dataset-fetch.md`](/home/awadallak/aivoices/docs/tasks/xtts-training-dataset-fetch.md)
- [`docs/tasks/xtts-training-resume-validation.md`](/home/awadallak/aivoices/docs/tasks/xtts-training-resume-validation.md)
