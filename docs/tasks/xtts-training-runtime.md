# XTTS Training Runtime

## Status

Planned

## Goal

Create the runtime layer for XTTS training on ephemeral cloud GPU VMs, using a container-first approach.

## Scope

- define the training container layout
- define required host/runtime assumptions for Runpod
- document how the VM should start the training container
- keep direct host execution as fallback only

## Deliverables

- training container definition
- runtime docs for cloud VMs
- minimal bootstrap contract for fetching dataset and starting training

## Must Support

- container-first execution
- dataset fetch from public R2 before training
- mounting writable output directories for run artifacts
- compatibility with the training entrypoint used by this repo
- `Python 3.11.9` as the current operational baseline

## Notes

- this task is about runtime and environment, not the trainer logic itself
- avoid coupling the runtime to the full local workspace layout
