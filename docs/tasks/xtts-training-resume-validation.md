# XTTS Resume Validation

## Status

Planned

## Goal

Empirically validate the resume contract for XTTS training before finalizing the `artifacts/last/` format.

## Why This Exists

Source inspection suggests that the Trainer checkpoint contains model, optimizer, scaler, step, and epoch state, and that `continue_path` expects a directory containing `config.json` plus the latest checkpoint.

That is not enough to treat resume as solved. We need a real interruption-and-resume test.

## Scope

- run a real interrupted training session
- resume it from the proposed `artifacts/last/` layout
- verify that training continues correctly
- verify whether scheduler behavior is acceptable

## Success Criteria

- resumed run starts from the expected step and epoch
- optimizer and scaler state resume correctly
- no missing-file assumptions exist beyond the chosen artifact contract
- scheduler behavior is either verified or documented with a concrete caveat

## Notes

- this task is a blocker for declaring the `last/` artifact contract fully stable
