---
name: debug-hpc-job
description: >-
  Diagnoses a failed or stuck SLURM/HPC job for hmfast (OOM, timeout, NaNs,
  CUDA errors, missing outputs). Use when the user asks "why did my job fail"
  or shows a SLURM log.
---

# Debug HPC job

## Triage

1. Locate the SLURM output: `logs/<jobname>-<jobid>.out`. Read it from the bottom.
2. Pull SLURM accounting:
   ```bash
   sacct -j <jobid> --format=JobID,State,ExitCode,MaxRSS,Elapsed,ReqMem,Timelimit
   ```
3. Match the failure signature:
   - `OOM` / `Killed` / `MaxRSS ~ ReqMem`: host memory pressure.
   - `DUE TO TIME LIMIT`: wall time exhausted.
   - `CUDA_ERROR_OUT_OF_MEMORY`: GPU memory pressure (often JAX 75% preallocation collision).
   - `NaN` or `inf` in saved arrays: numerics, not infrastructure.

## Fixes

- **Host memory:** reduce batch size, chunk along the largest axis, or request more `--mem` in the job script.
- **GPU memory:** lower `XLA_PYTHON_CLIENT_MEM_FRACTION` (last resort), reduce batch, or move parts to CPU only when JAX cannot fit.
- **Time:** add checkpointing; resubmit from the latest checkpoint. Re-estimate runtime with the small-N method from the `run-experiment` skill.
- **GPU OOM with multiple jobs:** set `CUDA_VISIBLE_DEVICES`. Do not let two jobs share one GPU silently.
- **NaNs:** insert a `jax.debug.print` or assert at the suspected step; bisect to the operation that first produces a non-finite value. Common culprits: log of a non-positive quantity, division by `sigma8 - sigma8`, integration outside emulator range.

## Before resubmitting

- Confirm the fix in a short interactive run on a tiny problem.
- Update the manifest with what failed and what you changed, so the next run can be reasoned about.
- If the same root cause appears twice in a row, **stop and ask the user** (see `.claude/rules/autonomous-operation.md`).
