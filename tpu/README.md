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
| `profile_stages.py` | Stage-by-stage timing of the tSZ Cl pipeline (HMF, bias, background, profile `u_k`, `pk_1h`, `pk_2h`, full `cl_*`). Used to identify TPU bottlenecks. |
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

## Current TPU vs CPU numbers (lxu-persistent, TPU v6 lite, jax 0.6.2)

Workload: full `cl_1h + cl_2h` Limber projection for the tSZ tracer with
`Nell=50, Nm=64, Nz=32`.

| backend | dtype   | compile  | min       | median    |
|---------|---------|----------|-----------|-----------|
| CPU     | float64 | 4.1 s    | 91 ms     | 100 ms    |
| TPU     | float32 | 10.4 s   | 49 ms     | 49 ms     |

→ ~2× speedup vs CPU, 11/11 correctness checks PASS (max relative diff vs
CPU float64 baseline: 1.86% on `cl_1h`, 0.77% on `cl_2h`).

## Where the TPU time goes (per `profile_stages.py`)

| stage                              | median wall-time |
|------------------------------------|------------------|
| HMF, bias, concentration           | ≤ 1 ms each      |
| background quantities              | ~1 ms each       |
| `u_k(k, m, [z0])` — one z slice    | **1.3 ms**       |
| `u_k` vmap over 32 z slices        | **53 ms**        |
| `pk_1h`/`pk_2h` for one z slice    | ~2 ms            |
| `cl_1h + cl_2h` (jit'd together)   | **49 ms**        |

The dominant cost is the **per-z vmap over `profile.u_k(...)`** inside
`cl_1h` / `cl_2h`. Each call runs a separate mcfit Hankel transform of
shape `(Nm, 1, 1024)`; vmap-over-z replays this 32 times rather than
batching into one `(Nm, 32, 1024)` FFT. XLA does not auto-fuse it because
the per-z slice contains data-dependent quantities (chi(z), d_A(z),
prefactors) that change between iterations.

### Acceleration paths (not yet implemented)

1. **Eliminate the outer vmap-over-z** (~1.5–2× expected). Refactor
   `profile.u_k(k, m, z)` so the caller can pass a 2D `k=(Nl, Nz)` or
   so `cl_1h`/`cl_2h` pass the multipole grid directly, letting the
   profile do *one* Hankel transform on a `(Nm, Nz, Nx)` integrand.
   Touches every concrete profile; do it carefully.

2. **Batch across cosmologies / MCMC samples**. The fundamental
   mismatch is that this single-Cl workload has only ~100k array
   elements, which doesn't saturate the TPU MXU. Calling
   `jax.vmap(cl_1h, in_axes=(cosmo_pytree, ...))` over a batch of 32–
   256 cosmology samples should give near-linear speedup because the
   batched ops fill the MXU. This is the right path for chain runs.

3. **Shrink the mcfit pad**. mcfit pads to next-power-of-two ≈ 1024.
   For tSZ accuracy we may tolerate 512; expect ~30–40% Hankel speedup.

4. **bfloat16 for sub-operations**. JAX 0.6.2 supports bfloat16 on
   TPU's MXU. Most of `u_k` doesn't need full float32; a targeted
   `.astype(jnp.bfloat16)` cast on the Hankel integrand could give
   significant speedup, with a correctness re-check via this
   benchmark.

5. **`jax.profiler.trace`**. Capture an HLO/TensorBoard trace to
   confirm dispatch-overhead dominance and find any other small
   kernels that can be fused.
