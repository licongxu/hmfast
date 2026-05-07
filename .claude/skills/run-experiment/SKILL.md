---
name: run-experiment
description: >-
  Plans and launches an hmfast experiment (training, inference, or scan) on the
  cluster with safe resource limits. Use when the user wants to run something
  that costs nontrivial compute, especially on GPU.
---

# Run experiment

Follow `.claude/rules/cluster-usage.md` and, for autonomous sessions, `.claude/rules/autonomous-operation.md`.

## Pre-flight

1. **Identify the device.** GPU partition? Which one? Quick `nvidia-smi` check on the target node.
2. **Estimate memory.** Sum array sizes in floats; multiply by `8` bytes (x64 default). Confirm it fits with headroom for JAX's default 75% preallocation. **Show the arithmetic.**
3. **Estimate runtime.** Run a tiny version (small `N`, few steps) and extrapolate. If the extrapolation is unclear, ask the user before launching the full run.
4. **Pick a seed and config.** Save both into the output directory.

## Launch

- Always via **SLURM** if estimated runtime > 10 min.
- Job script must set `--time`, `--mem`, `--gres=gpu:1` (when GPU is needed), and `--output=logs/%x-%j.out`.
- Activate the env inside the job script:
  ```bash
  source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
  ```
- Outputs go under `/scratch/scratch-lxu/...` (working) or `/rds/rds-lxu/...` (persistent), never inside the repo.

## During and after

- Stream the log: `tail -f logs/<jobname>-<jobid>.out`.
- On finish, verify saved arrays are finite and have the expected shapes.
- Write a `run.json` manifest with: git hash, full config, seed, `MaxRSS`, wall time, output paths.

## Stop conditions

If during the run you see OOM, repeated NaNs, or runtime exceeding the estimate by more than `2x`, **stop and report** before resubmitting. Do not blindly retry with the same parameters.
