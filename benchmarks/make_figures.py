"""
Generate benchmark and validation figures for the hmfast paper.

Outputs:
    paper/figures/benchmark_timing.pdf  - Bar chart of operation timings
    paper/figures/tsz_power_spectrum.pdf - tSZ 1h + 2h power spectrum validation
    paper/figures/tsz_cl.pdf            - tSZ angular power spectrum
    paper/figures/scaling.pdf           - Runtime vs grid size scaling
"""

import json
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer, CMBLensingTracer
from hmfast.halos.profiles.matter import NFWMatterProfile

os.makedirs("paper/figures", exist_ok=True)


def figure_timing_bar_chart():
    """Figure 1: Benchmark timing bar chart."""
    results_path = "benchmarks/results/baseline_full.json"
    if not os.path.exists(results_path):
        print(f"Skipping timing chart: {results_path} not found")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Collect timings
    labels = []
    means = []
    stds = []

    for name, timing in results["cosmology"].items():
        labels.append(f"cosmo/{name}")
        means.append(timing["mean_s"] * 1000)
        stds.append(timing["std_s"] * 1000)

    for key in ["pk_1h", "pk_2h"]:
        labels.append(f"tSZ/{key}")
        means.append(results["tsz_pk"][key]["mean_s"] * 1000)
        stds.append(results["tsz_pk"][key]["std_s"] * 1000)

    for key in ["cl_1h", "cl_2h"]:
        labels.append(f"tSZ/{key}")
        means.append(results["tsz_cl"][key]["mean_s"] * 1000)
        stds.append(results["tsz_cl"][key]["std_s"] * 1000)

    labels.append("CMB lens/cl_1h")
    means.append(results["cmb_lensing"]["cl_1h"]["mean_s"] * 1000)
    stds.append(results["cmb_lensing"]["cl_1h"]["std_s"] * 1000)

    labels.append(r"grad/$\Omega_{\rm cdm}$")
    means.append(results["gradient"]["gradient_omega_cdm"]["mean_s"] * 1000)
    stds.append(results["gradient"]["gradient_omega_cdm"]["std_s"] * 1000)

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(labels))
    colors = ["#4C72B0"] * 6 + ["#DD8452"] * 2 + ["#55A868"] * 2 + ["#C44E52"] + ["#8172B2"]

    bars = ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Wall-clock time (ms)", fontsize=11)
    ax.set_title("hmfast benchmark (NVIDIA RTX PRO 6000, after JIT warmup)", fontsize=11)
    ax.axvline(x=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, max(means) * 1.5)

    # Add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig("paper/figures/benchmark_timing.pdf", dpi=300)
    fig.savefig("paper/figures/benchmark_timing.png", dpi=300)
    plt.close()
    print("Saved paper/figures/benchmark_timing.pdf")


def figure_tsz_power_spectrum():
    """Figure 2: tSZ 3D power spectrum at z=0.5."""
    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)
    tracer = tSZTracer(profile=GNFWPressureProfile())

    k = jnp.logspace(-3, 1, 200)
    m = jnp.logspace(11, 15, 80)
    z = jnp.array([0.5])

    pk1h = hm.pk_1h(tracer, tracer, k=k, m=m, z=z)[:, 0]
    pk2h = hm.pk_2h(tracer, tracer, k=k, m=m, z=z)[:, 0]
    pk_tot = pk1h + pk2h

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(k, pk_tot, "k-", linewidth=2, label=r"$P_{\rm total}$")
    ax.loglog(k, pk1h, "C0--", linewidth=1.5, label=r"$P_{\rm 1h}$")
    ax.loglog(k, pk2h, "C1--", linewidth=1.5, label=r"$P_{\rm 2h}$")

    ax.set_xlabel(r"$k\;[\mathrm{Mpc}^{-1}]$", fontsize=12)
    ax.set_ylabel(r"$P(k, z{=}0.5)\;[\mathrm{Mpc}^3]$", fontsize=12)
    ax.set_title(r"tSZ power spectrum (GNFW pressure profile)", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(k[0], k[-1])

    fig.tight_layout()
    fig.savefig("paper/figures/tsz_power_spectrum.pdf", dpi=300)
    fig.savefig("paper/figures/tsz_power_spectrum.png", dpi=300)
    plt.close()
    print("Saved paper/figures/tsz_power_spectrum.pdf")


def figure_tsz_cl():
    """Figure 3: tSZ angular power spectrum."""
    # Use fresh objects to avoid JIT cache conflicts from previous figures
    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)
    tracer = tSZTracer(profile=GNFWPressureProfile())

    ell = jnp.logspace(1, 4, 30)
    m = jnp.logspace(11, 15, 20)
    z = jnp.linspace(0.01, 3.0, 15)

    # Warm up JIT first with a small call
    _ = hm.cl_1h(tracer, tracer, l=ell[:5], m=m[:5], z=z[:5])
    _.block_until_ready()
    _ = hm.cl_2h(tracer, tracer, l=ell[:5], m=m[:5], z=z[:5])
    _.block_until_ready()

    cl1h = hm.cl_1h(tracer, tracer, l=ell, m=m, z=z)
    cl1h.block_until_ready()
    cl2h = hm.cl_2h(tracer, tracer, l=ell, m=m, z=z)
    cl2h.block_until_ready()
    cl_tot = cl1h + cl2h

    fig, ax = plt.subplots(figsize=(6, 4))
    ell2cl = ell * (ell + 1) / (2 * jnp.pi)

    ax.loglog(ell, ell2cl * cl_tot, "k-", linewidth=2, label=r"$C_\ell^{\rm total}$")
    ax.loglog(ell, ell2cl * cl1h, "C0--", linewidth=1.5, label=r"$C_\ell^{\rm 1h}$")
    ax.loglog(ell, ell2cl * cl2h, "C1--", linewidth=1.5, label=r"$C_\ell^{\rm 2h}$")

    ax.set_xlabel(r"Multipole $\ell$", fontsize=12)
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell / 2\pi$", fontsize=12)
    ax.set_title(r"tSZ angular power spectrum", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(ell[0], ell[-1])

    fig.tight_layout()
    fig.savefig("paper/figures/tsz_cl.pdf", dpi=300)
    fig.savefig("paper/figures/tsz_cl.png", dpi=300)
    plt.close()
    print("Saved paper/figures/tsz_cl.pdf")


def figure_scaling():
    """Figure 4: Runtime scaling with grid size."""
    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)
    tracer = tSZTracer(profile=GNFWPressureProfile())

    grid_sizes = [10, 20, 40, 80]
    timings_pk1h = []
    timings_cl1h = []

    for n in grid_sizes:
        k = jnp.logspace(-3, 1, n * 2)
        m = jnp.logspace(11, 15, n)
        z = jnp.linspace(0.01, 3.0, n)

        # Warmup
        pk = hm.pk_1h(tracer, tracer, k=k, m=m, z=z)
        pk.block_until_ready()

        import time
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            pk = hm.pk_1h(tracer, tracer, k=k, m=m, z=z)
            pk.block_until_ready()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        timings_pk1h.append(np.mean(times) * 1000)

    for n in grid_sizes:
        ell = jnp.logspace(1, 4, n * 2)
        m = jnp.logspace(11, 15, n)
        z = jnp.linspace(0.01, 3.0, n)

        # Warmup
        cl = hm.cl_1h(tracer, tracer, l=ell, m=m, z=z)
        cl.block_until_ready()

        import time
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            cl = hm.cl_1h(tracer, tracer, l=ell, m=m, z=z)
            cl.block_until_ready()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        timings_cl1h.append(np.mean(times) * 1000)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid_sizes, timings_pk1h, "o-", linewidth=2, markersize=8, label=r"$P_{\rm 1h}(k,z)$")
    ax.plot(grid_sizes, timings_cl1h, "s-", linewidth=2, markersize=8, label=r"$C_\ell^{\rm 1h}$")
    ax.set_xlabel("Grid size $N_m = N_z$", fontsize=12)
    ax.set_ylabel("Wall-clock time (ms)", fontsize=12)
    ax.set_title("Runtime scaling with grid size", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("paper/figures/scaling.pdf", dpi=300)
    fig.savefig("paper/figures/scaling.png", dpi=300)
    plt.close()
    print("Saved paper/figures/scaling.pdf")


def figure_gpu_cpu_speedup():
    """Figure: GPU vs CPU speedup grouped bar chart."""
    gpu_path = "benchmarks/results/baseline_full.json"
    cpu_path = "benchmarks/results/baseline_full_cpu.json"
    if not os.path.exists(gpu_path) or not os.path.exists(cpu_path):
        print("Skipping GPU vs CPU chart: result files not found")
        return

    with open(gpu_path) as f:
        gpu = json.load(f)
    with open(cpu_path) as f:
        cpu = json.load(f)

    # Operations for the halo-model comparison (skip 1D cosmology ops)
    ops = [
        (r"tSZ $P_{\rm 1h}$", "tsz_pk", "pk_1h"),
        (r"tSZ $P_{\rm 2h}$", "tsz_pk", "pk_2h"),
        (r"tSZ $C_\ell^{\rm 1h}$", "tsz_cl", "cl_1h"),
        (r"tSZ $C_\ell^{\rm 2h}$", "tsz_cl", "cl_2h"),
        (r"CMB lens $C_\ell^{\rm 1h}$", "cmb_lensing", "cl_1h"),
        (r"$\nabla_{\Omega_{\rm cdm}}$", "gradient", "gradient_omega_cdm"),
    ]

    labels = [op[0] for op in ops]
    gpu_ms = [gpu[op[1]][op[2]]["mean_s"] * 1000 for op in ops]
    cpu_ms = [cpu[op[1]][op[2]]["mean_s"] * 1000 for op in ops]
    speedups = [c / g for c, g in zip(cpu_ms, gpu_ms)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: grouped bar chart
    x = np.arange(len(labels))
    w = 0.35
    bars_gpu = ax1.bar(x - w / 2, gpu_ms, w, label="GPU", color="#4C72B0", edgecolor="black", linewidth=0.5)
    bars_cpu = ax1.bar(x + w / 2, cpu_ms, w, label="CPU", color="#DD8452", edgecolor="black", linewidth=0.5)

    ax1.set_ylabel("Wall-clock time (ms)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.set_yscale("log")
    ax1.set_title("GPU vs CPU evaluation time", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    # Annotate GPU bars
    for bar, val in zip(bars_gpu, gpu_ms):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#4C72B0")

    # Right: speedup bar chart
    colors = ["#55A868" if s > 5 else "#C44E52" for s in speedups]
    ax2.barh(range(len(labels)), speedups, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("GPU speedup factor", fontsize=11)
    ax2.set_title("GPU speedup over CPU", fontsize=12)
    ax2.axvline(x=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.grid(axis="x", alpha=0.3)

    for i, s in enumerate(speedups):
        ax2.text(s + 0.3, i, f"{s:.1f}x", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig("paper/figures/gpu_cpu_speedup.pdf", dpi=300)
    fig.savefig("paper/figures/gpu_cpu_speedup.png", dpi=300)
    plt.close()
    print("Saved paper/figures/gpu_cpu_speedup.pdf")


def figure_batch_vmap():
    """Figure: Batch vmap throughput scaling."""
    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)
    tracer = tSZTracer(profile=GNFWPressureProfile())

    k = jnp.logspace(-3, 1, 100)
    m = jnp.logspace(11, 15, 40)
    z = jnp.array([0.5])

    def eval_pk(omega_cdm):
        c = cosmo.update(omega_cdm=omega_cdm)
        h = hm.update(cosmology=c)
        return h.pk_1h(tracer, tracer, k=k, m=m, z=z)

    batch_fn = jax.vmap(eval_pk)

    batch_sizes = [1, 2, 4, 8, 16, 32]
    times_per_cosmo = []

    for nb in batch_sizes:
        omega_vals = jnp.linspace(0.10, 0.14, nb)
        # Warmup
        _ = batch_fn(omega_vals)
        _.block_until_ready()
        # Measure
        import time
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = batch_fn(omega_vals)
            _.block_until_ready()
            ts.append(time.perf_counter() - t0)
        total = min(ts)
        times_per_cosmo.append(total / nb * 1000)

    # Single cosmology baseline
    _ = eval_pk(0.12)
    _.block_until_ready()
    import time
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        pk = eval_pk(0.12)
        pk.block_until_ready()
        ts.append(time.perf_counter() - t0)
    single_time = min(ts) * 1000

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(batch_sizes, times_per_cosmo, "o-", linewidth=2, markersize=8, color="#4C72B0",
            label=r"$P_{\rm 1h}$ via vmap (per-cosmo)")
    ax.axhline(y=single_time, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"Single cosmo: {single_time:.1f} ms")
    ax.set_xlabel("Batch size (number of cosmologies)", fontsize=12)
    ax.set_ylabel("Wall-clock time per cosmology (ms)", fontsize=12)
    ax.set_title("Batch evaluation throughput via jax.vmap", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("paper/figures/batch_vmap.pdf", dpi=300)
    fig.savefig("paper/figures/batch_vmap.png", dpi=300)
    plt.close()
    print("Saved paper/figures/batch_vmap.pdf")


def figure_tracer_comparison():
    """Figure: All 6 tracers pk_1h comparison + timing comparison."""
    gpu_path = "benchmarks/results/baseline_full.json"
    cpu_path = "benchmarks/results/baseline_full_cpu.json"

    # -- Timing comparison from saved data --
    tracer_names = ["tSZ", "kSZ", "CMB_lensing", "galaxy_HOD", "galaxy_lensing", "CIB"]
    tracer_labels = ["tSZ", "kSZ", "CMB lensing", "Galaxy HOD", "Galaxy lensing", "CIB"]
    gpu_ms = []
    cpu_ms = []

    if os.path.exists(gpu_path) and os.path.exists(cpu_path):
        with open(gpu_path) as f:
            gpu = json.load(f)
        with open(cpu_path) as f:
            cpu = json.load(f)
        for name in tracer_names:
            gpu_ms.append(gpu["all_tracers_pk1h"][name]["pk_1h"]["mean_s"] * 1000)
            cpu_ms.append(cpu["all_tracers_pk1h"][name]["pk_1h"]["mean_s"] * 1000)

    # -- Power spectrum computation for all tracers --
    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)

    from hmfast.tracers import tSZTracer, kSZTracer, CMBLensingTracer, GalaxyHODTracer, GalaxyLensingTracer, CIBTracer
    from hmfast.halos.profiles.pressure import GNFWPressureProfile
    from hmfast.halos.profiles.density import B16DensityProfile
    from hmfast.halos.profiles.matter import NFWMatterProfile
    from hmfast.halos.profiles.hod import Z07GalaxyHODProfile
    from hmfast.halos.profiles.cib import S12CIBProfile

    tracers = [
        ("tSZ", tSZTracer(profile=GNFWPressureProfile())),
        ("kSZ", kSZTracer(profile=B16DensityProfile())),
        ("CMB lensing", CMBLensingTracer(profile=NFWMatterProfile())),
        ("Galaxy HOD", GalaxyHODTracer(profile=Z07GalaxyHODProfile())),
        ("Galaxy lensing", GalaxyLensingTracer(profile=NFWMatterProfile())),
        ("CIB", CIBTracer(profile=S12CIBProfile(nu=100))),
    ]

    k = jnp.logspace(-2, 0, 100)
    m = jnp.logspace(11, 15, 40)
    z = jnp.array([0.5])

    # Warmup all tracers
    for _, tr in tracers:
        _ = hm.pk_1h(tr, tr, k=k[:5], m=m[:5], z=z)
        _.block_until_ready()

    pk_data = {}
    for label, tr in tracers:
        pk = hm.pk_1h(tr, tr, k=k, m=m, z=z)[:, 0]
        pk.block_until_ready()
        pk_data[label] = pk

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Power spectra for all tracers
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    for (label, _), color in zip(tracers, colors):
        pk = pk_data[label]
        # Only plot finite values
        mask = jnp.isfinite(pk) & (pk > 0)
        if jnp.any(mask):
            ax1.loglog(jnp.where(mask, k, jnp.nan), jnp.where(mask, pk, jnp.nan),
                       color=color, linewidth=1.5, label=label)

    ax1.set_xlabel(r"$k\;[\mathrm{Mpc}^{-1}]$", fontsize=12)
    ax1.set_ylabel(r"$P_{\rm 1h}(k, z{=}0.5)$", fontsize=12)
    ax1.set_title("1-halo power spectra for all tracers", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_xlim(k[0], k[-1])

    # Right: Timing comparison
    if gpu_ms and cpu_ms:
        x = np.arange(len(tracer_labels))
        w = 0.35
        ax2.bar(x - w / 2, gpu_ms, w, label="GPU", color="#4C72B0", edgecolor="black", linewidth=0.5)
        ax2.bar(x + w / 2, cpu_ms, w, label="CPU", color="#DD8452", edgecolor="black", linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(tracer_labels, fontsize=8, rotation=20, ha="right")
        ax2.set_ylabel("Wall-clock time (ms)", fontsize=11)
        ax2.set_title(r"$P_{\rm 1h}$ evaluation time per tracer", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("paper/figures/tracer_comparison.pdf", dpi=300)
    fig.savefig("paper/figures/tracer_comparison.png", dpi=300)
    plt.close()
    print("Saved paper/figures/tracer_comparison.pdf")


def figure_gradient_accuracy():
    """Figure: Gradient accuracy — analytic vs finite-difference."""
    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)
    tracer = tSZTracer(profile=GNFWPressureProfile())

    k = jnp.logspace(-2, 0, 50)
    m = jnp.logspace(11, 15, 30)
    z = jnp.array([0.5])

    # Compute analytic gradient
    def loss_fn(omega_cdm):
        c = cosmo.update(omega_cdm=omega_cdm)
        h = hm.update(cosmology=c)
        pk = h.pk_1h(tracer, tracer, k=k, m=m, z=z)
        return jnp.sum(pk ** 2)

    grad_fn = jax.grad(loss_fn)
    omega_cdm_0 = cosmo.omega_cdm
    analytic_grad = grad_fn(omega_cdm_0)

    # Compute finite-difference gradient at multiple eps values
    epsilons = jnp.logspace(-8, -1, 30)
    fd_grads = []
    rel_errors = []
    for eps in epsilons:
        fd = (loss_fn(omega_cdm_0 + eps) - loss_fn(omega_cdm_0 - eps)) / (2 * eps)
        fd_grads.append(float(fd))
        rel_errors.append(abs(float(fd) - float(analytic_grad)) / abs(float(analytic_grad)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(epsilons, rel_errors, "o-", markersize=4, linewidth=1.5, color="#4C72B0")
    ax.set_xlabel(r"Finite-difference step $\epsilon$", fontsize=12)
    ax.set_ylabel(r"Relative error $|\nabla_{\rm AD} - \nabla_{\rm FD}| / |\nabla_{\rm AD}|$", fontsize=11)
    ax.set_title("Gradient accuracy: AD vs finite difference", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Mark the optimal eps region
    best_idx = jnp.argmin(jnp.array(rel_errors))
    ax.annotate(f"Best: $\\epsilon$={float(epsilons[best_idx]):.0e}\nRel err={rel_errors[best_idx]:.1e}",
                xy=(float(epsilons[best_idx]), rel_errors[best_idx]),
                xytext=(float(epsilons[best_idx]) * 5, rel_errors[best_idx] * 3),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9)

    fig.tight_layout()
    fig.savefig("paper/figures/gradient_accuracy.pdf", dpi=300)
    fig.savefig("paper/figures/gradient_accuracy.png", dpi=300)
    plt.close()
    print("Saved paper/figures/gradient_accuracy.pdf")


if __name__ == "__main__":
    print("Generating hmfast paper figures...")
    figure_timing_bar_chart()
    figure_tsz_power_spectrum()
    figure_tsz_cl()
    figure_scaling()
    figure_batch_vmap()
    figure_gpu_cpu_speedup()
    figure_tracer_comparison()
    figure_gradient_accuracy()
    print("Done.")
