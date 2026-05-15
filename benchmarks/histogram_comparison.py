"""
Generate a figure comparing hmfast masked tSZ D_ell against:
1. The saved synthetic realizations (scatter)
2. The tszpower theory curve

Matches the style of check_binned_ps_scatter_masked_signal_only.png
"""
from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)

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

ELL_MIN = np.array([9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835], dtype=int)
ELL_MAX = np.array([12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085], dtype=int)
ELL_EFF = np.array([10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5, 117.0, 152.5, 198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5])

# Binning
_L_MAX = int(np.max(ELL_MAX - ELL_MIN + 1))
_ELL_INT = ELL_MIN[:, None] + np.arange(_L_MAX)[None, :]
_ELL_MASK = (_ELL_INT <= ELL_MAX[:, None]).astype(np.float64)


def bin_to_18(ell_in, Cl_in):
    log_ell = np.log(np.asarray(ell_in))
    Cl_q = np.interp(np.log(_ELL_INT.astype(float)), log_ell, np.asarray(Cl_in))
    Dl_q = _ELL_INT * (_ELL_INT + 1.0) * Cl_q / (2.0 * np.pi) * 1e12
    num = np.sum(Dl_q * _ELL_MASK, axis=1)
    den = np.sum(_ELL_MASK, axis=1)
    return num / den


def main():
    # Parameters from reference notebook
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

    # Warmup
    snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=A_SZ, alpha_SZ=alpha_SZ,
                         B=B_val, coeff=coeff_j)
    mask = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat,
                                     n_power=2, n_grid=512, nsig=8.0)
    cl_int = hm.cl_1h_masked(tsz, None, ell_int, m_grid, z_grid, mask, k_damp=0.0)
    block(cl_int)

    Dl_hmfast = bin_to_18(np.asarray(ell_int), np.asarray(cl_int))

    print("hmfast D_ell:")
    for i in range(18):
        print(f"  ell={ELL_EFF[i]:7.1f}  D_ell={Dl_hmfast[i]:.6e}")

    # Compare with reference data real0
    ref_path = "/scratch/scratch-lxu/tsz_cnc_scatter/synthetic_data/Dl_binned_masked_signal_only_real0.txt"
    ref_data = np.loadtxt(ref_path)
    ref_ell = ref_data[:, 0]
    ref_Dl = ref_data[:, 1]

    # Compute trispectrum and covariance
    print("\nComputing hmfast trispectrum...")
    ell_tri = jnp.geomspace(float(ELL_MIN[0]), float(ELL_MAX[-1]), 30)
    T = hm.trispectrum_1h(tsz, None, ell_tri, ell_tri, m_grid, z_grid, k_damp=0.0)
    block(T)
    T_np = np.asarray(T)
    ell_tri_np = np.asarray(ell_tri)

    # Build binning operator
    n_ell = len(ell_tri_np)
    Bop = np.zeros((18, n_ell), dtype=float)
    for j in range(n_ell):
        e = np.zeros(n_ell, dtype=float)
        e[j] = 1.0
        Bop[:, j] = bin_to_18(ell_tri_np, e)

    # Diag Gaussian covariance
    delta_ell = np.empty(n_ell)
    delta_ell[:-1] = np.diff(ell_tri_np)
    delta_ell[-1] = delta_ell[-2]

    C_tot = np.interp(ell_tri_np, np.asarray(ell_int), np.asarray(cl_int))
    diag_term = (4.0 * np.pi) * (C_tot ** 2) / (ell_tri_np + 0.5)
    M_G = np.diag(diag_term / (4.0 * np.pi * 0.9505 * delta_ell))
    M_C = M_G + T_np / (4.0 * np.pi * 0.9505)
    M_D_18 = Bop @ M_C @ Bop.T
    sigma_D_18 = np.sqrt(np.clip(np.diag(M_D_18), 0.0, None))

    print(f"\nD_ell errors from hmfast covariance:")
    for i in range(18):
        print(f"  ell={ELL_EFF[i]:7.1f}  D_ell={Dl_hmfast[i]:.6e}  sigma={sigma_D_18[i]:.6e}")

    # Figure: hmfast theory vs reference data
    fig, axes = plt.subplots(3, 6, figsize=(20, 10), constrained_layout=True)
    axes = axes.flatten()

    for i in range(18):
        ax = axes[i]
        ax.axvline(Dl_hmfast[i], color="C1", ls="--", lw=2,
                   label="hmfast theory" if i == 0 else None)
        ax.axvline(ref_Dl[i], color="C0", ls=":", lw=2,
                   label="ref data (real0)" if i == 0 else None)

        if np.isfinite(sigma_D_18[i]) and sigma_D_18[i] > 0:
            xg = np.linspace(Dl_hmfast[i] - 4 * sigma_D_18[i],
                             Dl_hmfast[i] + 4 * sigma_D_18[i], 200)
            yg = norm.pdf(xg, loc=Dl_hmfast[i], scale=sigma_D_18[i])
            ax.plot(xg, yg, "C1-", lw=1.5, alpha=0.7)

        ax.set_title(rf"$\ell_{{\rm eff}}={ELL_EFF[i]:.1f}$")
        if i >= 12:
            ax.set_xlabel(r"$D_\ell$")
        if i % 6 == 0:
            ax.set_ylabel("Theory PDF")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", frameon=False,
                   ncol=2, bbox_to_anchor=(0.5, 1.05))

    os.makedirs("benchmarks/results", exist_ok=True)
    fig.savefig("benchmarks/results/hmfast_theory_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig("benchmarks/results/hmfast_theory_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to benchmarks/results/hmfast_theory_comparison.pdf")

    # Ratio comparison
    ratio = Dl_hmfast / ref_Dl
    print(f"\nhmfast/ref_data ratio: min={ratio.min():.4f}, max={ratio.max():.4f}, "
          f"mean={ratio.mean():.4f}, std={ratio.std():.4f}")


if __name__ == "__main__":
    main()
