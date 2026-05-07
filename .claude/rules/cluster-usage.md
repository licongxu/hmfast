---
description: HPC cluster, GPU, memory, and runtime constraints for hmfast work
---

# Cluster usage and hardware constraints

These rules are **hard constraints** for any compute Claude launches on behalf of the user.

## Resource priorities

1. **GPU first.** If a code path has a JAX (or PyTorch) implementation, run it on GPU. Do not silently fall back to CPU because a GPU is busy. Either queue the job or stop and ask.
2. **Bound memory.** Before launching anything that allocates `>~ 4 GiB`, estimate peak memory in floats and confirm it fits on the target device with headroom. Show the arithmetic in your reply. JAX preallocates ~75% of GPU memory by default; lower `XLA_PYTHON_CLIENT_MEM_FRACTION` only when you have to share the device.
3. **Bound runtime.** Any single command that may run **longer than 10 minutes** must be submitted as a SLURM job with explicit `--time` and `--mem`, not run interactively in a shell tool call.

## SLURM defaults

- Submit to a GPU partition when GPUs are needed: `--partition=ampere` (or the project's GPU partition), `--gres=gpu:1`.
- Always set `--time`, `--mem`, `--cpus-per-task`, and `--output=logs/%x-%j.out`.
- Use `sbatch --test-only` first when the script is new.
- Activate the environment **inside** the job script, not in `.bashrc`:
  ```bash
  source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
  ```

## Forbidden actions

- Do **not** start training or sampling runs without checkpointing.
- Do **not** allocate `O(N^2)` arrays for large `N` without an explicit memory check first.
- Do **not** run `pytest -m slow`, full emulator retraining, or sampler chains on the login node.
- Do **not** run multiple GPU processes that race for the same device. Use `CUDA_VISIBLE_DEVICES` to be explicit.
- Do **not** write large outputs into the repository directory.

## Storage

- **Working outputs:** `/scratch/scratch-lxu/...` (fast, not persistent).
- **Persistent outputs:** `/rds/rds-lxu/...`.
- Always include the full output path in any "job done" report.

## Sanity checks before declaring a job done

- Search the SLURM log for `OOM`, `Segmentation fault`, `CUDA_ERROR_OUT_OF_MEMORY`, and silent NaNs in saved arrays.
- Save a **manifest** with: hmfast git hash, full config, seed, runtime, peak memory (`MaxRSS`), output paths.
