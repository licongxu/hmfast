"""Time the tSZ Cl pipeline on a single backend and write the result to JSON.

Designed to be invoked twice (once with ``JAX_PLATFORMS=cpu`` and once with
``JAX_PLATFORMS=tpu``) by ``tpu/run_benchmark.sh``, which then prints a
side-by-side table. Keeping this as a single-backend process avoids JAX's
"backend selected at import time" footgun and lets ``hmfast.jax_platform``
pick the correct dtype for whichever platform is active.

Usage
-----
    JAX_PLATFORMS=cpu python3 tpu/benchmark_cpu_vs_tpu.py --out cpu.json
    JAX_PLATFORMS=tpu python3 tpu/benchmark_cpu_vs_tpu.py --out tpu.json

Each invocation:
  * warms the JIT cache with one untimed call,
  * runs N_RUNS timed executions of ``cl_1h + cl_2h`` (full Limber pipeline),
  * records compile-time, mean/median/min run-time, mean Cl (sanity), and the
    backend + device label.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer

N_WARMUP = 1
N_RUNS = 5

# Grid is shared by both backends so the comparison is apples-to-apples.
ELL = (10.0, 10000.0, 50)   # logspace(log10(10), log10(10000), 50)
M = (10.0, 15.0, 64)        # logspace(10, 15, 64) in M_sun
Z = (0.01, 3.0, 32)         # linspace


def _device_label() -> str:
    devs = jax.devices()
    if not devs:
        return "unknown"
    d = devs[0]
    kind = getattr(d, "device_kind", "")
    return f"{d.platform}:{kind}" if kind else d.platform


def benchmark():
    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)
    tsz = tSZTracer(profile=GNFWPressureProfile())

    m = jnp.logspace(M[0], M[1], M[2])
    z = jnp.linspace(Z[0], Z[1], Z[2])
    ell = jnp.logspace(jnp.log10(ELL[0]), jnp.log10(ELL[1]), ELL[2])

    @jax.jit
    def compute_tsz_cl(hm, tsz, ell, m, z):
        cl1h = hm.cl_1h(tsz, tsz, l=ell, m=m, z=z)
        cl2h = hm.cl_2h(tsz, tsz, l=ell, m=m, z=z)
        return cl1h, cl2h

    # Compile + warm cache
    t0 = time.perf_counter()
    cl1h, cl2h = compute_tsz_cl(hm, tsz, ell, m, z)
    cl1h.block_until_ready()
    cl2h.block_until_ready()
    compile_seconds = time.perf_counter() - t0

    # Extra warmup runs (post-compile) to flush any one-off caching.
    for _ in range(N_WARMUP):
        c1, c2 = compute_tsz_cl(hm, tsz, ell, m, z)
        c1.block_until_ready()
        c2.block_until_ready()

    # Timed runs.
    run_times = []
    for _ in range(N_RUNS):
        t = time.perf_counter()
        c1, c2 = compute_tsz_cl(hm, tsz, ell, m, z)
        c1.block_until_ready()
        c2.block_until_ready()
        run_times.append(time.perf_counter() - t)

    # Correctness probes: full Cl arrays + a few non-projected sanity points.
    cl1h_np = np.asarray(cl1h)
    cl2h_np = np.asarray(cl2h)
    cl_tot = cl1h_np + cl2h_np

    # Spot checks against pk_* at a couple representative (k, z) values.
    pk1h = hm.pk_1h(tsz, tsz, k=jnp.array([0.01, 0.1, 1.0]), m=m, z=jnp.array([0.5]))
    pk2h = hm.pk_2h(tsz, tsz, k=jnp.array([0.01, 0.1, 1.0]), m=m, z=jnp.array([0.5]))

    return {
        "backend": jax.default_backend(),
        "device": _device_label(),
        "dtype": str(cl1h.dtype),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "n_ell": int(ell.shape[0]),
        "n_m": int(m.shape[0]),
        "n_z": int(z.shape[0]),
        "compile_seconds": compile_seconds,
        "n_runs": N_RUNS,
        "run_seconds_min": min(run_times),
        "run_seconds_median": statistics.median(run_times),
        "run_seconds_mean": statistics.mean(run_times),
        "run_seconds_max": max(run_times),
        "run_seconds_all": run_times,
        # Correctness payload.
        "ell": np.asarray(ell).tolist(),
        "cl1h": cl1h_np.tolist(),
        "cl2h": cl2h_np.tolist(),
        "cl_mean": float(np.mean(cl_tot)),
        "cl1h_all_finite": bool(np.all(np.isfinite(cl1h_np))),
        "cl2h_all_finite": bool(np.all(np.isfinite(cl2h_np))),
        "cl1h_all_positive": bool(np.all(cl1h_np > 0)),
        "cl2h_all_positive": bool(np.all(cl2h_np > 0)),
        "pk1h_spot_k_0p01_0p1_1p0_z_0p5": np.asarray(pk1h).flatten().tolist(),
        "pk2h_spot_k_0p01_0p1_1p0_z_0p5": np.asarray(pk2h).flatten().tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Path to write the JSON result.")
    args = p.parse_args()

    print(f"[{jax.default_backend()}] starting benchmark ({_device_label()})", flush=True)
    result = benchmark()
    print(f"[{result['backend']}] compile={result['compile_seconds']:.4f}s "
          f"run(median)={result['run_seconds_median']*1000:.3f}ms "
          f"mean_cl={result['cl_mean']:.4e}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
