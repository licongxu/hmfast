"""
Generate histogram figure comparing hmfast masked tSZ D_ell against
1000 scatter realizations, with theory Gaussian PDF from trispectrum covariance.

Matches the style of check_binned_ps_scatter_masked_signal_only.png from tszpower.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 25,
    "axes.labelsize": 25,
    "axes.titlesize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 25,
    "text.latex.preamble": (
        r"\usepackage[T1]{fontenc}\usepackage{type1cm}"
        r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}"
    ),
})

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import ParametricGNFWPressureProfile
from hmfast.tracers import tSZTracer
from hmfast.tracers.tsz_completeness import (
    build_snr_grid,
    conditional_An_undetected,
    load_sigma_y0_curve,
)

ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835],
    dtype=int,
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085],
    dtype=int,
)
ELL_EFF = np.array(
    [10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5, 117.0, 152.5,
     198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5],
)

_L_MAX = int(np.max(ELL_MAX - ELL_MIN + 1))
_ELL_INT = ELL_MIN[:, None] + np.arange(_L_MAX)[None, :]
_ELL_MASK = (_ELL_INT <= ELL_MAX[:, None]).astype(np.float64)


def bin_to_18(ell_in, Cl_in):
    """Bin unbinned Cl to 18-band D_ell using Planck-like top-hat bins."""
    log_ell = np.log(np.asarray(ell_in))
    Cl_q = np.interp(np.log(_ELL_INT.astype(float)), log_ell, np.asarray(Cl_in))
    Dl_q = _ELL_INT * (_ELL_INT + 1.0) * Cl_q / (2.0 * np.pi) * 1e12
    num = np.sum(Dl_q * _ELL_MASK, axis=1)
    den = np.sum(_ELL_MASK, axis=1)
    return num / den


def main():
    # Fiducial parameters (matching the tszpower reference)
    H0 = 67.66
    omega_cdm = 0.1193
    omega_b = 0.02242
    ln10_10A_s = 2.9718
    n_s = 0.9665
    tau_reio = 0.0544
    B_val = 1.41
    A_SZ = -4.2373
    alpha_SZ = 1.12
    sigma_lnY = 0.173
    q_cat = 5.0
    f_sky_eff = 0.9558  # from reference notebook

    # Load 1000 scatter realizations
    print("Loading 1000 scatter realizations...")
    data_dir = "/rds/rds-lxu/tsz_project/tsz_ps_benchmark_scatter_masked_signal"
    Dl_all = []
    for i in range(1000):
        path = os.path.join(data_dir, f"Dl_binned_scatter_{i}.npz")
        if os.path.exists(path):
            Dl_all.append(np.load(path)["Dl_masked"])
    Dl_all = np.array(Dl_all)
    print(f"Loaded {Dl_all.shape[0]} realizations, shape {Dl_all.shape}")

    # Compute hmfast theory
    print("Computing hmfast masked tSZ theory...")
    cosmo = Cosmology(emulator_set="lcdm:v1")
    cosmo = cosmo.update(
        H0=H0, omega_cdm=omega_cdm, omega_b=omega_b,
        ln1e10A_s=ln10_10A_s, n_s=n_s, tau_reio=tau_reio,
    )
    hm = HaloModel(
        cosmology=cosmo,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
    )
    prof = ParametricGNFWPressureProfile(A_SZ=A_SZ, alpha_SZ=alpha_SZ, B=B_val)
    tsz = tSZTracer(profile=prof)

    ell_int = jnp.geomspace(float(ELL_MIN[0]), float(ELL_MAX[-1]), 50)
    m_grid = jnp.geomspace(6.766e13, 6.766e15, 100)
    z_grid = jnp.geomspace(0.005, 3.0, 100)

    coeff, _ = load_sigma_y0_curve()
    coeff_j = jnp.asarray(coeff)
    block = getattr(jax, "block_until_ready", lambda x: x)

    # Warmup: compute SNR grid, then masks for PS (n=2) and trispectrum (n=4)
    snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=A_SZ, alpha_SZ=alpha_SZ,
                         B=B_val, coeff=coeff_j)
    mask_ps = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat,
                                        n_power=2, n_grid=512, nsig=8.0)
    mask_tri = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat,
                                         n_power=4, n_grid=512, nsig=8.0)
    cl_int = hm.cl_1h_masked(tsz, None, ell_int, m_grid, z_grid, mask_ps, k_damp=0.0)
    block(cl_int)
    Dl_theory_18 = bin_to_18(np.asarray(ell_int), np.asarray(cl_int))

    print("hmfast theory D_ell:")
    for i in range(18):
        print(f"  ell={ELL_EFF[i]:7.1f}  D_ell={Dl_theory_18[i]:.6e}")

    # Trispectrum and covariance (with conditional 4th-moment mask)
    print("Computing hmfast masked trispectrum...")
    ell_tri = jnp.geomspace(float(ELL_MIN[0]), float(ELL_MAX[-1]), 30)
    T = hm.trispectrum_1h_masked(tsz, None, ell_tri, ell_tri, m_grid, z_grid,
                                  mask_tri, k_damp=0.0)
    block(T)
    T_np = np.asarray(T)
    ell_tri_np = np.asarray(ell_tri)

    n_ell = len(ell_tri_np)
    Bop = np.zeros((18, n_ell), dtype=float)
    for j in range(n_ell):
        e = np.zeros(n_ell, dtype=float)
        e[j] = 1.0
        Bop[:, j] = bin_to_18(ell_tri_np, e)

    delta_ell = np.empty(n_ell)
    delta_ell[:-1] = np.diff(ell_tri_np)
    delta_ell[-1] = delta_ell[-2]

    C_tot = np.interp(ell_tri_np, np.asarray(ell_int), np.asarray(cl_int))
    diag_term = (4.0 * np.pi) * (C_tot ** 2) / (ell_tri_np + 0.5)
    M_G = np.diag(diag_term / (4.0 * np.pi * f_sky_eff * delta_ell))
    M_C = M_G + T_np / (4.0 * np.pi * f_sky_eff)
    M_D_18 = Bop @ M_C @ Bop.T
    sigma_D_18 = np.sqrt(np.clip(np.diag(M_D_18), 0.0, None))

    print(f"D_ell errors from hmfast covariance:")
    for i in range(18):
        print(f"  ell={ELL_EFF[i]:7.1f}  D_ell={Dl_theory_18[i]:.6e}  sigma={sigma_D_18[i]:.6e}")

    # Ratio and bias against empirical mean
    Dl_mean = np.mean(Dl_all, axis=0)
    Dl_std = np.std(Dl_all, axis=0, ddof=1)
    ratio_mean = Dl_mean / Dl_theory_18
    ratio_std = Dl_std / sigma_D_18
    bias = (Dl_mean - Dl_theory_18) / Dl_std
    print(f"\nPer-bin diagnostics:")
    for i in range(18):
        print(f"  ell={ELL_EFF[i]:7.1f}: mean/th={ratio_mean[i]:.4f}  std/th={ratio_std[i]:.4f}  bias={bias[i]:+.4f}")

    # ---- Figure: histogram of realizations with theory Gaussian overlay ----
    fig, axes = plt.subplots(3, 6, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()

    for i in range(18):
        ax = axes[i]
        col = Dl_all[:, i]
        col = col[np.isfinite(col)]

        mu_th = Dl_theory_18[i]
        sig_th = sigma_D_18[i]

        x_lo = min(np.min(col), mu_th - 4.0 * sig_th)
        x_hi = max(np.max(col), mu_th + 4.0 * sig_th)
        if x_hi <= x_lo:
            x_hi = x_lo + 1e-12

        edges = np.histogram_bin_edges(col, bins="fd", range=(x_lo, x_hi))
        if len(edges) < 2:
            edges = np.linspace(x_lo, x_hi, max(2, min(50, col.size // 2)))

        ax.hist(
            col, bins=edges, density=True, alpha=0.5, color="C0",
            histtype="stepfilled", label="Realizations" if i == 0 else None,
        )

        xg = np.linspace(x_lo, x_hi, 400)
        if np.isfinite(sig_th) and sig_th > 0:
            yg = norm.pdf(xg, loc=mu_th, scale=sig_th)
            ax.plot(xg, yg, "r-", lw=1.8,
                    label="Theory Gaussian" if i == 0 else None)
            ax.axvline(mu_th, color="r", ls="--", lw=1.2,
                       label="Theory mean" if i == 0 else None)

        ax.axvline(np.mean(col), color="k", ls="--", lw=1.2,
                   label="Empirical mean" if i == 0 else None)
        ax.set_title(rf"$\ell_{{\rm eff}}={ELL_EFF[i]:.1f}$")

        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(fmt)

        if i >= 12:
            ax.set_xlabel(r"$D_\ell$")
        if i % 6 == 0:
            ax.set_ylabel("Probability Density")

    fig.canvas.draw()
    for ax in axes:
        offset_text = ax.xaxis.get_offset_text()
        offset_text.set_fontsize(18)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", frameon=False, ncol=4,
            bbox_to_anchor=(0.5, 1.08),
        )

    os.makedirs("benchmarks/results", exist_ok=True)
    fig.savefig("benchmarks/results/hmfast_histogram_scatter_masked_signal_only.png",
                dpi=300, bbox_inches="tight")
    fig.savefig("benchmarks/results/hmfast_histogram_scatter_masked_signal_only.pdf",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure saved to benchmarks/results/hmfast_histogram_scatter_masked_signal_only.pdf")


if __name__ == "__main__":
    main()
