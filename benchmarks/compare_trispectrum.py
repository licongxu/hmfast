"""
hmfast vs tszpower trispectrum (1-halo Poisson) comparison.

Uses the same fiducial parameters and unit convention as the masked-Cl
comparison: hmfast in physical M_sun, tszpower in M_sun/h.
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

sys.path.insert(0, "/scratch/scratch-lxu/tszsbi/tszpower")
import tszpower.tsz as tszp_tsz
from tszpower.utils import ELL_EFF, ELL_MIN, ELL_MAX, get_ell_range
from tszpower.config import classy_sz
from tszpower.initialise import initialise as _tszp_initialise

FID = dict(H0=67.66, omega_cdm=0.1193, omega_b=0.02242,
           ln10_10A_s=2.9718, n_s=0.9665, tau_reio=0.0544)
FID_B = 1.41
FID_A_SZ = -4.2373
FID_ALPHA_SZ = 1.12
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


def main():
    # ---- hmfast ----
    cosmo = Cosmology(emulator_set="lcdm:v1").update(
        H0=FID["H0"], omega_cdm=FID["omega_cdm"], omega_b=FID["omega_b"],
        ln1e10A_s=FID["ln10_10A_s"], n_s=FID["n_s"], tau_reio=FID["tau_reio"],
    )
    hm = HaloModel(cosmology=cosmo,
                   mass_definition=MassDefinition(500, "critical"),
                   convert_masses=True)
    prof = ParametricGNFWPressureProfile(A_SZ=FID_A_SZ, alpha_SZ=FID_ALPHA_SZ, B=FID_B)
    tsz = tSZTracer(profile=prof)

    m_grid = jnp.geomspace(M_MIN, M_MAX, N_MZ)
    z_grid = jnp.geomspace(Z_MIN, Z_MAX, N_MZ)
    ell_internal = jnp.asarray(get_ell_range())
    print("ell_internal shape:", ell_internal.shape,
          "range:", float(ell_internal[0]), float(ell_internal[-1]))

    T_hm = hm.trispectrum_1h(tsz, None, ell_internal, ell_internal,
                             m_grid, z_grid, k_damp=0.0)
    T_hm = np.array(jax.device_get(T_hm))

    # ---- tszpower ----
    classy_sz.set({**_FIXED_ASTRO,
                   "H0": FID["H0"], "omega_b": FID["omega_b"], "omega_cdm": FID["omega_cdm"],
                   "ln10^{10}A_s": FID["ln10_10A_s"], "n_s": FID["n_s"],
                   "tau_reio": FID["tau_reio"], "B": FID_B})
    _tszp_initialise()
    params = {"H0": FID["H0"], "omega_b": FID["omega_b"], "omega_cdm": FID["omega_cdm"],
              "ln10^{10}A_s": FID["ln10_10A_s"], "n_s": FID["n_s"],
              "tau_reio": FID["tau_reio"], "B": FID_B, **_FIXED_ASTRO}
    # tszpower's compute_trispectrum returns (ell, T_{ll'}) on the internal grid.
    ell_tszp, T_tszp = tszp_tsz.compute_trispectrum(params_values_dict=params)
    ell_tszp = np.asarray(jax.device_get(ell_tszp))
    T_tszp = np.asarray(jax.device_get(T_tszp))
    # Interpolate hmfast T onto tszpower's ell grid for direct comparison
    # (tszpower returns on its native ~100-point smooth grid; hmfast was
    # evaluated on the same get_ell_range() grid, so they should match).
    if T_tszp.shape != T_hm.shape:
        print(f"Reshaping: hmfast {T_hm.shape}, tszpower {T_tszp.shape}")
        # 2D log-log interpolation on each axis
        def _interp2d(T, x_src, x_dst):
            log_T = np.log(np.clip(T, 1e-300, None))
            # interpolate along axis 1 first
            tmp = np.array([np.interp(np.log(x_dst), np.log(x_src), row) for row in log_T])
            # then along axis 0
            out = np.array([np.interp(np.log(x_dst), np.log(x_src), tmp[:, j])
                            for j in range(tmp.shape[1])]).T
            return np.exp(out)
        T_hm = _interp2d(T_hm, np.asarray(ell), ell_tszp)
        ell = ell_tszp

    print("T_hm shape:", T_hm.shape, " T_tszp shape:", T_tszp.shape)

    ell = np.asarray(ell_internal)

    # Diagonal comparison
    T_hm_diag = np.diag(T_hm)
    T_tszp_diag = np.diag(T_tszp)
    ratio_diag = T_hm_diag / T_tszp_diag
    print("\nDiagonal T(l,l) comparison:")
    print(f"{'ell':>8s} {'hmfast':>15s} {'tszpower':>15s} {'ratio':>10s}")
    for i in range(0, len(ell), max(1, len(ell) // 10)):
        print(f"{ell[i]:8.1f} {T_hm_diag[i]:15.5e} {T_tszp_diag[i]:15.5e} "
              f"{ratio_diag[i]:10.5f}")
    print(f"\nMax |ratio - 1| on diagonal: {np.max(np.abs(ratio_diag - 1.0))*100:.3f}%")
    print(f"Mean diag ratio: {np.mean(ratio_diag):.5f}")

    ratio_full = T_hm / T_tszp
    print(f"Max |ratio - 1| over full (l,l') grid: "
          f"{np.max(np.abs(ratio_full - 1.0))*100:.3f}%")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(ell, T_tszp_diag, "o-", color="tab:orange",
            label="tszpower (reference)", markersize=4, lw=1.5)
    ax.plot(ell, T_hm_diag, "x--", color="tab:blue",
            label="hmfast", markersize=6, lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$T_{\ell\ell}$")
    ax.set_title("Trispectrum diagonal (1h Poisson)")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    ax.axhspan(0.99, 1.01, color="lightgray", alpha=0.6, label=r"$\pm 1\%$ band")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.plot(ell, ratio_diag, "x-", color="tab:blue", markersize=6, lw=1.2)
    ax.set_xscale("log")
    ax.set_ylim(0.97, 1.03)
    ax.set_xlabel(r"$\ell$"); ax.set_ylabel("hmfast / tszpower")
    ax.set_title("Diagonal ratio")
    ax.legend(frameon=False, loc="upper right")

    out_dir = "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/benchmarks/results"
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, "hmfast_vs_tszpower_trispectrum.pdf")
    png = os.path.join(out_dir, "hmfast_vs_tszpower_trispectrum.png")
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    print(f"\nSaved:\n  {pdf}\n  {png}")


if __name__ == "__main__":
    main()
