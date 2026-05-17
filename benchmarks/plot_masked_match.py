"""
Plot hmfast vs tszpower masked tSZ D_ell across the 18 Planck bins.

Generates a two-panel figure:
  (top)    D_ell vs ell with both curves overplotted
  (bottom) ratio hmfast/tszpower with the +/- 1% band
"""
from __future__ import annotations
import sys
import os

import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
})
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import ParametricGNFWPressureProfile
from hmfast.tracers import tSZTracer
from hmfast.tracers.tsz_completeness import (
    build_snr_grid, conditional_An_undetected, load_sigma_y0_curve,
)

sys.path.insert(0, "/scratch/scratch-lxu/tszsbi/tszpower")
import tszpower.parametric_profile as pp
from tszpower.utils import ELL_EFF, ELL_MIN, ELL_MAX
from tszpower.config import classy_sz
from tszpower.initialise import initialise as _tszp_initialise

FID = dict(H0=67.66, omega_cdm=0.1193, omega_b=0.02242,
           ln10_10A_s=2.9718, n_s=0.9665, tau_reio=0.0544)
FID_B = 1.41
FID_A_SZ = -4.2373
FID_ALPHA_SZ = 1.12
FID_SIGMA_LNY = 0.173
Q_CAT = 5.0

M_MIN = 1e14
M_MAX = 1e16
_h = FID["H0"] / 100.0
M_MIN_TSZP = M_MIN * _h
M_MAX_TSZP = M_MAX * _h
Z_MIN, Z_MAX = 0.005, 3.0
N_MZ = 100

_FIXED_ASTRO = {
    "M_min": M_MIN_TSZP, "M_max": M_MAX_TSZP, "z_min": Z_MIN, "z_max": Z_MAX,
    "P0GNFW": 8.130, "c500": 1.156,
    "gammaGNFW": 0.3292, "alphaGNFW": 1.0620, "betaGNFW": 5.4807,
    "jax": 1, "cosmo_model": 0,
}

_L_MAX = int(np.max(ELL_MAX - ELL_MIN + 1))
_ELL_INT = ELL_MIN[:, None] + np.arange(_L_MAX)[None, :]
_ELL_MASK = (_ELL_INT <= ELL_MAX[:, None]).astype(np.float64)


def bin_to_18(ell_in, Cl_in):
    log_ell = np.log(np.asarray(ell_in))
    Cl_q = np.interp(np.log(_ELL_INT.astype(float)), log_ell, np.asarray(Cl_in))
    Dl_q = _ELL_INT * (_ELL_INT + 1.0) * Cl_q / (2.0 * np.pi) * 1e12
    return (Dl_q * _ELL_MASK).sum(1) / _ELL_MASK.sum(1)


def main():
    cosmo = Cosmology(emulator_set="lcdm:v1").update(
        H0=FID["H0"], omega_cdm=FID["omega_cdm"], omega_b=FID["omega_b"],
        ln1e10A_s=FID["ln10_10A_s"], n_s=FID["n_s"], tau_reio=FID["tau_reio"],
    )
    hm = HaloModel(cosmology=cosmo,
                   mass_definition=MassDefinition(500, "critical"),
                   convert_masses=True)
    prof = ParametricGNFWPressureProfile(A_SZ=FID_A_SZ, alpha_SZ=FID_ALPHA_SZ, B=FID_B)
    tsz = tSZTracer(profile=prof)

    ell_int = jnp.geomspace(float(ELL_MIN[0]), float(ELL_MAX[-1]), 50)
    m_grid = jnp.geomspace(M_MIN, M_MAX, N_MZ)
    z_grid = jnp.geomspace(Z_MIN, Z_MAX, N_MZ)

    coeff, _ = load_sigma_y0_curve()
    coeff_j = jnp.asarray(coeff)
    snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=FID_A_SZ, alpha_SZ=FID_ALPHA_SZ,
                         B=FID_B, coeff=coeff_j)
    mask = conditional_An_undetected(snr, sigma_lnY=FID_SIGMA_LNY, q_cat=Q_CAT,
                                     n_power=2, n_grid=1024, nsig=8.0)
    cl_hm = hm.cl_1h_masked(tsz, None, ell_int, m_grid, z_grid, mask, k_damp=0.0)
    Dl_hm = bin_to_18(np.asarray(ell_int), np.asarray(cl_hm))

    classy_sz.set({**_FIXED_ASTRO,
                   "H0": FID["H0"], "omega_b": FID["omega_b"], "omega_cdm": FID["omega_cdm"],
                   "ln10^{10}A_s": FID["ln10_10A_s"], "n_s": FID["n_s"],
                   "tau_reio": FID["tau_reio"], "B": FID_B})
    _tszp_initialise()
    params = {"H0": FID["H0"], "omega_b": FID["omega_b"], "omega_cdm": FID["omega_cdm"],
              "ln10^{10}A_s": FID["ln10_10A_s"], "n_s": FID["n_s"],
              "tau_reio": FID["tau_reio"], "B": FID_B, **_FIXED_ASTRO}
    Dl_tszp = np.array(jax.device_get(pp.compute_masked_tsz_Dell_binned_parametric(
        params_values_dict=params,
        A_SZ=FID_A_SZ, alpha_SZ=FID_ALPHA_SZ,
        q_cat=Q_CAT, sigma_lnY=FID_SIGMA_LNY,
    )))

    ratio = Dl_hm / Dl_tszp

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    ax1.plot(ELL_EFF, Dl_tszp, "o-", color="tab:orange", label="tszpower (reference)",
             markersize=6, lw=1.5)
    ax1.plot(ELL_EFF, Dl_hm, "x--", color="tab:blue", label="hmfast (fixed)",
             markersize=8, lw=1.2)
    ax1.set_yscale("log"); ax1.set_xscale("log")
    ax1.set_ylabel(r"$D_\ell^{yy,\,\mathrm{masked}}\times 10^{12}$")
    ax1.legend(frameon=False, loc="lower right")
    ax1.set_title(
        r"Masked tSZ $D_\ell$: hmfast vs tszpower (fiducial cosmology, $q_{\mathrm{cat}}=5$, $\sigma_{\ln Y}=0.173$)"
    )

    ax2.axhspan(0.99, 1.01, color="lightgray", alpha=0.6, label=r"$\pm 1\%$ band")
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.plot(ELL_EFF, ratio, "x-", color="tab:blue", markersize=8, lw=1.2)
    ax2.set_ylim(0.985, 1.020)
    ax2.set_xlabel(r"$\ell_{\rm eff}$")
    ax2.set_ylabel("hmfast / tszpower")
    ax2.legend(frameon=False, loc="upper right")

    out_dir = "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, "hmfast_vs_tszpower_masked.pdf")
    png = os.path.join(out_dir, "hmfast_vs_tszpower_masked.png")
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    print(f"Saved:\n  {pdf}\n  {png}")
    print(f"Max ratio deviation from 1.0: {float(np.max(np.abs(ratio - 1.0)))*100:.3f}%")
    print(f"Mean ratio: {float(np.mean(ratio)):.5f}")


if __name__ == "__main__":
    main()
