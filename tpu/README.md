# `tpu/` – Cloud TPU smoke tests and benchmarks

Self-contained scripts for exercising the `hmfast` pipeline on Cloud TPU VMs
(verified on `lxu-persistent`, v6e, `jax 0.6.2 + libtpu`). Everything here is
runtime/CI tooling – nothing is imported by the `hmfast` package itself.

## Files

| File | Purpose |
| --- | --- |
| `test_tsz.py` | End-to-end smoke test: builds the GNFW–tSZ Cl pipeline, times one compile + one execute, and saves `tsz_spectrum_<backend>.png` next to itself. Driven by `../../tpu_submission/sync_and_run.sh`. |
| `benchmark_cpu_vs_tpu.py` | Single-backend timing + correctness harness (compile + N=5 timed runs of `cl_1h + cl_2h`, plus full `cl_1h`/`cl_2h` arrays and a few `pk_1h`/`pk_2h` spot points). Writes a JSON result file. Invoked twice by the wrapper below. |
| `run_benchmark.sh` | Wrapper that runs the benchmark on CPU then TPU (separate processes so each picks the right dtype) and: (i) prints a side-by-side timing table with the wall-time speed-up; (ii) runs per-ℓ correctness checks (max/median relative diff vs CPU baseline, finite-positivity, `pk_*` spot consistency) with explicit PASS/FAIL; (iii) writes `tsz_cpu_vs_tpu.png` overlaying both backends with a residual panel. JSON outputs land in `bench_results/`. Exits non-zero if any correctness check fails. |
| `bench_results/` | Per-run JSON outputs (git-ignored). |
| `tsz_spectrum_*.png`, `tsz_cpu_vs_tpu.png` | Generated plot artifacts (git-ignored). |

## Quick start (on a TPU VM)

```bash
# One-shot tSZ smoke + plot
PYTHONPATH=~/hmfast/src HMFAST_JAX_ENABLE_X64=0 JAX_PLATFORMS=tpu \
    python3 tpu/test_tsz.py

# CPU vs TPU timing table
tpu/run_benchmark.sh
```

The remote pipeline shipped from a laptop is `~/tpu_submission/sync_and_run.sh`
(outside this repo); it `tar | ssh`s the source tree to the TPU VM and runs
`tpu/test_tsz.py` automatically.
