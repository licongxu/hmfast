"""
hmfast benchmark suite.

Measures runtime for key halo-model operations across tracers and grid sizes.
Outputs results as JSON for reproducibility.

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --quick    # fast mode for CI
    python benchmarks/run_all.py --output benchmarks/results/baseline.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import (
    GNFWPressureProfile, NFWMatterProfile, B12PressureProfile,
    B16DensityProfile, Z07GalaxyHODProfile, S12CIBProfile,
)
from hmfast.tracers import tSZTracer, CMBLensingTracer, kSZTracer, GalaxyHODTracer, GalaxyLensingTracer, CIBTracer


def get_git_info():
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
        git_branch = "unknown"
    return git_hash, git_branch


def get_device_info():
    devices = jax.devices()
    info = []
    for d in devices:
        info.append({"device_kind": d.device_kind, "id": d.id})
    return info


def time_fn(fn, *args, warmup=1, repeats=5, **kwargs):
    """Time a function with warmup and repeated runs."""
    # Warmup (includes JIT compilation)
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        result.block_until_ready()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        result.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "warmup": warmup,
        "repeats": repeats,
    }


def benchmark_init():
    """Benchmark HaloModel initialization."""
    t0 = time.perf_counter()
    cosmo = Cosmology(emulator_set="lcdm:v1")
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    hm = HaloModel(cosmology=cosmo)
    t3 = time.perf_counter()

    return {
        "cosmology_init_s": t1 - t0,
        "halomodel_init_s": t3 - t2,
        "total_init_s": t3 - t0,
    }


def benchmark_tsz_pk(hm, tracer, quick=False):
    """Benchmark tSZ 1-halo and 2-halo power spectra."""
    if quick:
        k = jnp.logspace(-3, 1, 50)
        m = jnp.logspace(11, 15, 30)
        z = jnp.linspace(0.01, 3.0, 10)
    else:
        k = jnp.logspace(-3, 1, 200)
        m = jnp.logspace(11, 15, 80)
        z = jnp.linspace(0.01, 3.0, 30)

    pk1h = hm.pk_1h(tracer, tracer, k=k, m=m, z=z)
    pk1h.block_until_ready()

    pk2h = hm.pk_2h(tracer, tracer, k=k, m=m, z=z)
    pk2h.block_until_ready()

    t_pk1h = time_fn(hm.pk_1h, tracer, tracer, k, m, z, warmup=1, repeats=5)
    t_pk2h = time_fn(hm.pk_2h, tracer, tracer, k, m, z, warmup=1, repeats=5)

    shapes = {
        "pk_1h_shape": list(pk1h.shape),
        "pk_2h_shape": list(pk2h.shape),
        "Nk": len(k), "Nm": len(m), "Nz": len(z),
    }

    return {
        "pk_1h": t_pk1h,
        "pk_2h": t_pk2h,
        "shapes": shapes,
        "finite": {
            "pk_1h": bool(jnp.all(jnp.isfinite(pk1h))),
            "pk_2h": bool(jnp.all(jnp.isfinite(pk2h))),
        },
    }


def benchmark_tsz_cl(hm, tracer, quick=False):
    """Benchmark tSZ angular power spectra (Limber projection)."""
    if quick:
        ell = jnp.logspace(1, 4, 30)
        m = jnp.logspace(11, 15, 30)
        z = jnp.linspace(0.01, 3.0, 20)
    else:
        ell = jnp.logspace(1, 4, 100)
        m = jnp.logspace(11, 15, 80)
        z = jnp.linspace(0.01, 3.0, 50)

    cl1h = hm.cl_1h(tracer, tracer, l=ell, m=m, z=z)
    cl1h.block_until_ready()

    cl2h = hm.cl_2h(tracer, tracer, l=ell, m=m, z=z)
    cl2h.block_until_ready()

    t_cl1h = time_fn(hm.cl_1h, tracer, tracer, ell, m, z, warmup=1, repeats=3)
    t_cl2h = time_fn(hm.cl_2h, tracer, tracer, ell, m, z, warmup=1, repeats=3)

    return {
        "cl_1h": t_cl1h,
        "cl_2h": t_cl2h,
        "shapes": {
            "cl_1h_shape": list(cl1h.shape),
            "cl_2h_shape": list(cl2h.shape),
            "Nell": len(ell), "Nm": len(m), "Nz": len(z),
        },
        "finite": {
            "cl_1h": bool(jnp.all(jnp.isfinite(cl1h))),
            "cl_2h": bool(jnp.all(jnp.isfinite(cl2h))),
        },
    }


def benchmark_cmb_lensing_cl(hm, quick=False):
    """Benchmark CMB lensing angular power spectrum."""
    matter_profile = NFWMatterProfile()
    cmb_tracer = CMBLensingTracer(profile=matter_profile)

    if quick:
        ell = jnp.logspace(1, 4, 30)
        m = jnp.logspace(11, 15, 30)
        z = jnp.linspace(0.01, 5.0, 20)
    else:
        ell = jnp.logspace(1, 4, 100)
        m = jnp.logspace(11, 15, 80)
        z = jnp.linspace(0.01, 5.0, 50)

    cl1h = hm.cl_1h(cmb_tracer, cmb_tracer, l=ell, m=m, z=z)
    cl1h.block_until_ready()

    t_cl1h = time_fn(hm.cl_1h, cmb_tracer, cmb_tracer, ell, m, z, warmup=1, repeats=3)

    return {
        "cl_1h": t_cl1h,
        "shape": list(cl1h.shape),
        "finite": bool(jnp.all(jnp.isfinite(cl1h))),
    }


def benchmark_gradient(hm, tracer, quick=False):
    """Benchmark gradient evaluation via jax.grad."""
    if quick:
        k = jnp.logspace(-3, 1, 30)
        m = jnp.logspace(11, 15, 20)
        z = jnp.linspace(0.01, 3.0, 5)
    else:
        k = jnp.logspace(-3, 1, 100)
        m = jnp.logspace(11, 15, 50)
        z = jnp.linspace(0.01, 3.0, 15)

    def loss_fn(omega_cdm):
        cosmo_new = hm.cosmology.update(omega_cdm=omega_cdm)
        hm_new = hm.update(cosmology=cosmo_new)
        pk = hm_new.pk_1h(tracer, tracer, k=k, m=m, z=z)
        return jnp.sum(pk ** 2)

    grad_fn = jax.jit(jax.grad(loss_fn))

    omega_cdm_0 = hm.cosmology.omega_cdm
    grad_val = grad_fn(omega_cdm_0)
    grad_val.block_until_ready()

    t_grad = time_fn(grad_fn, omega_cdm_0, warmup=1, repeats=3)

    return {
        "gradient_omega_cdm": t_grad,
        "grad_finite": bool(jnp.isfinite(grad_val)),
    }


def benchmark_all_tracers_pk1h(hm, quick=False):
    """Benchmark pk_1h for all 6 tracers to compare profile complexity."""
    if quick:
        k = jnp.logspace(-3, 1, 30)
        m = jnp.logspace(11, 15, 15)
        z = jnp.array([0.5])
    else:
        k = jnp.logspace(-3, 1, 100)
        m = jnp.logspace(11, 15, 50)
        z = jnp.array([0.5])

    tracers = {
        "tSZ": tSZTracer(profile=GNFWPressureProfile()),
        "kSZ": kSZTracer(profile=B16DensityProfile()),
        "CMB_lensing": CMBLensingTracer(profile=NFWMatterProfile()),
        "galaxy_HOD": GalaxyHODTracer(profile=Z07GalaxyHODProfile()),
        "galaxy_lensing": GalaxyLensingTracer(profile=NFWMatterProfile()),
        "CIB": CIBTracer(profile=S12CIBProfile(nu=100)),
    }

    results = {}
    for name, tracer in tracers.items():
        # Warmup
        pk = hm.pk_1h(tracer, tracer, k=k, m=m, z=z)
        pk.block_until_ready()
        t = time_fn(hm.pk_1h, tracer, tracer, k, m, z, warmup=1, repeats=3)
        results[name] = {
            "pk_1h": t,
            "shape": list(pk.shape),
            "finite": bool(jnp.all(jnp.isfinite(pk))),
        }

    return results


def benchmark_cosmology(hm):
    """Benchmark individual cosmology operations."""
    z = jnp.linspace(0.0, 5.0, 100)

    results = {}
    results["hubble_parameter"] = time_fn(
        lambda: hm.cosmology.hubble_parameter(z).block_until_ready(),
        warmup=1, repeats=5
    )
    results["angular_diameter_distance"] = time_fn(
        lambda: hm.cosmology.angular_diameter_distance(z).block_until_ready(),
        warmup=1, repeats=5
    )
    results["sigma8"] = time_fn(
        lambda: hm.cosmology.sigma8(z).block_until_ready(),
        warmup=1, repeats=5
    )
    results["pk_linear"] = time_fn(
        lambda: hm.cosmology.pk(0.5, linear=True)[1].block_until_ready(),
        warmup=1, repeats=5
    )
    results["derived_parameters"] = time_fn(
        lambda: hm.cosmology.derived_parameters()["sigma8"].block_until_ready(),
        warmup=1, repeats=5
    )
    results["cl_tt"] = time_fn(
        lambda: hm.cosmology.cl_tt()[1].block_until_ready(),
        warmup=1, repeats=5
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="hmfast benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Fast mode for CI")
    parser.add_argument("--output", default="benchmarks/results/baseline.json",
                        help="Output JSON path")
    args = parser.parse_args()

    git_hash, git_branch = get_git_info()
    device_info = get_device_info()

    print(f"hmfast benchmark suite")
    print(f"  git: {git_hash} ({git_branch})")
    print(f"  device: {device_info}")
    print(f"  quick: {args.quick}")
    print()

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "git_branch": git_branch,
        "devices": device_info,
        "jax_version": jax.__version__,
        "quick": args.quick,
    }

    # 1. Init
    print("=== Initialization ===")
    init_results = benchmark_init()
    results["initialization"] = init_results
    print(f"  Cosmology init: {init_results['cosmology_init_s']:.3f}s")
    print(f"  HaloModel init: {init_results['halomodel_init_s']:.3f}s")

    # Create objects for subsequent benchmarks
    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)
    tsz_tracer = tSZTracer(profile=GNFWPressureProfile())

    # 2. Cosmology operations
    print("\n=== Cosmology Operations ===")
    cosmo_results = benchmark_cosmology(hm)
    results["cosmology"] = cosmo_results
    for name, timing in cosmo_results.items():
        print(f"  {name}: {timing['mean_s']*1000:.1f} +/- {timing['std_s']*1000:.1f} ms")

    # 3. tSZ power spectra
    print("\n=== tSZ Power Spectra (3D) ===")
    tsz_pk_results = benchmark_tsz_pk(hm, tsz_tracer, quick=args.quick)
    results["tsz_pk"] = tsz_pk_results
    print(f"  pk_1h: {tsz_pk_results['pk_1h']['mean_s']*1000:.1f} +/- {tsz_pk_results['pk_1h']['std_s']*1000:.1f} ms")
    print(f"  pk_2h: {tsz_pk_results['pk_2h']['mean_s']*1000:.1f} +/- {tsz_pk_results['pk_2h']['std_s']*1000:.1f} ms")
    print(f"  Shapes: pk_1h={tsz_pk_results['shapes']['pk_1h_shape']}, pk_2h={tsz_pk_results['shapes']['pk_2h_shape']}")
    print(f"  Finite: pk_1h={tsz_pk_results['finite']['pk_1h']}, pk_2h={tsz_pk_results['finite']['pk_2h']}")

    # 4. tSZ angular power spectra
    print("\n=== tSZ Angular Power Spectra ===")
    tsz_cl_results = benchmark_tsz_cl(hm, tsz_tracer, quick=args.quick)
    results["tsz_cl"] = tsz_cl_results
    print(f"  cl_1h: {tsz_cl_results['cl_1h']['mean_s']*1000:.1f} +/- {tsz_cl_results['cl_1h']['std_s']*1000:.1f} ms")
    print(f"  cl_2h: {tsz_cl_results['cl_2h']['mean_s']*1000:.1f} +/- {tsz_cl_results['cl_2h']['std_s']*1000:.1f} ms")
    print(f"  Finite: cl_1h={tsz_cl_results['finite']['cl_1h']}, cl_2h={tsz_cl_results['finite']['cl_2h']}")

    # 5. CMB lensing
    print("\n=== CMB Lensing ===")
    cmb_results = benchmark_cmb_lensing_cl(hm, quick=args.quick)
    results["cmb_lensing"] = cmb_results
    print(f"  cl_1h: {cmb_results['cl_1h']['mean_s']*1000:.1f} +/- {cmb_results['cl_1h']['std_s']*1000:.1f} ms")
    print(f"  Finite: {cmb_results['finite']}")

    # 6. Gradient
    print("\n=== Gradient (d/d omega_cdm of sum pk_1h^2) ===")
    grad_results = benchmark_gradient(hm, tsz_tracer, quick=args.quick)
    results["gradient"] = grad_results
    print(f"  grad: {grad_results['gradient_omega_cdm']['mean_s']*1000:.1f} +/- {grad_results['gradient_omega_cdm']['std_s']*1000:.1f} ms")
    print(f"  grad finite: {grad_results['grad_finite']}")

    # 7. All-tracer pk_1h comparison
    print("\n=== All-Tracer pk_1h Comparison ===")
    all_tracer_results = benchmark_all_tracers_pk1h(hm, quick=args.quick)
    results["all_tracers_pk1h"] = all_tracer_results
    for name, r in all_tracer_results.items():
        print(f"  {name}: {r['pk_1h']['mean_s']*1000:.1f} +/- {r['pk_1h']['std_s']*1000:.1f} ms (finite={r['finite']})")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Operation':<30} {'Mean (ms)':<12} {'Std (ms)':<12}")
    print("-" * 54)
    for name, timing in cosmo_results.items():
        print(f"  cosmo/{name:<26} {timing['mean_s']*1000:>8.1f}     {timing['std_s']*1000:>8.1f}")
    print(f"  tsz/pk_1h{'':<24} {tsz_pk_results['pk_1h']['mean_s']*1000:>8.1f}     {tsz_pk_results['pk_1h']['std_s']*1000:>8.1f}")
    print(f"  tsz/pk_2h{'':<24} {tsz_pk_results['pk_2h']['mean_s']*1000:>8.1f}     {tsz_pk_results['pk_2h']['std_s']*1000:>8.1f}")
    print(f"  tsz/cl_1h{'':<24} {tsz_cl_results['cl_1h']['mean_s']*1000:>8.1f}     {tsz_cl_results['cl_1h']['std_s']*1000:>8.1f}")
    print(f"  tsz/cl_2h{'':<24} {tsz_cl_results['cl_2h']['mean_s']*1000:>8.1f}     {tsz_cl_results['cl_2h']['std_s']*1000:>8.1f}")
    print(f"  cmb/cl_1h{'':<24} {cmb_results['cl_1h']['mean_s']*1000:>8.1f}     {cmb_results['cl_1h']['std_s']*1000:>8.1f}")
    print(f"  grad/omega_cdm{'':<18} {grad_results['gradient_omega_cdm']['mean_s']*1000:>8.1f}     {grad_results['gradient_omega_cdm']['std_s']*1000:>8.1f}")


if __name__ == "__main__":
    main()
