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

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer

N_WARMUP = 1
N_RUNS = 5


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

    m = jnp.logspace(10, 15, 64)
    z = jnp.linspace(0.01, 3.0, 32)
    ell = jnp.logspace(jnp.log10(10), jnp.log10(10000), 50)

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

    # Extra warmup runs (post-compile) to flush any one-off caching
    for _ in range(N_WARMUP):
        c1, c2 = compute_tsz_cl(hm, tsz, ell, m, z)
        c1.block_until_ready()
        c2.block_until_ready()

    # Timed runs
    run_times = []
    for _ in range(N_RUNS):
        t = time.perf_counter()
        c1, c2 = compute_tsz_cl(hm, tsz, ell, m, z)
        c1.block_until_ready()
        c2.block_until_ready()
        run_times.append(time.perf_counter() - t)

    cl_tot = cl1h + cl2h
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
        "cl_mean": float(jnp.mean(cl_tot)),
        "cl1h_all_positive": bool(jnp.all(cl1h > 0)),
        "cl2h_all_positive": bool(jnp.all(cl2h > 0)),
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
