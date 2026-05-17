"""
Test whether the residual ~0.7% Cl offset is from binning interpolation:
compute hmfast Cl at *every integer ell* in [9, 1085] (no interpolation needed
for binning), then bin uniformly to 18 Planck bands.
"""
from __future__ import annotations
import sys, os, time
import tensorflow as tf
try: tf.config.set_visible_devices([], "GPU")
except RuntimeError: pass

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.concentration import ConstantConcentration
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
B = 1.41; A_SZ = -4.2373; ALPHA_SZ = 1.12; SIGMA_LNY = 0.173; Q_CAT = 5.0
M_MIN = 1e14; M_MAX = 1e16
_h = FID["H0"] / 100.0
M_MIN_TSZP = M_MIN * _h; M_MAX_TSZP = M_MAX * _h
_FIXED_ASTRO = {
    "M_min": M_MIN_TSZP, "M_max": M_MAX_TSZP, "z_min": 5e-3, "z_max": 3.0,
    "P0GNFW": 8.130, "c500": 1.156,
    "gammaGNFW": 0.3292, "alphaGNFW": 1.0620, "betaGNFW": 5.4807,
    "jax": 1, "cosmo_model": 0,
}

# Exact integer-ell grid covering every band, plus the per-band masks
ell_min_arr = np.asarray(ELL_MIN, dtype=int)
ell_max_arr = np.asarray(ELL_MAX, dtype=int)
LMAX = int(ell_max_arr[-1])  # 1085
LMIN = int(ell_min_arr[0])   # 9
ell_int = np.arange(LMIN, LMAX + 1, dtype=np.float64)  # 1077 points


def bin_exact(Cl_at_int):
    """Bin Cl(ell) sampled at every integer ell to 18 Planck bands by uniform
    average of D_ell over integer ell in each band (matches tszpower convention,
    no interpolation needed since Cl is already on the integer grid)."""
    ell = ell_int
    Dl_q = ell * (ell + 1.0) * np.asarray(Cl_at_int) / (2.0 * np.pi) * 1e12
    out = np.empty(18)
    for i in range(18):
        sel = (ell >= ell_min_arr[i]) & (ell <= ell_max_arr[i])
        out[i] = Dl_q[sel].mean()
    return out


def bin_log_interp(ell_in, Cl_in):
    """tszpower-style binning: linearly interpolate Cl_in (sampled at sparse
    ell_in) onto every integer ell in each band, then average D_ell uniformly."""
    log_ell = np.log(np.asarray(ell_in))
    _L_MAX = int(np.max(ell_max_arr - ell_min_arr + 1))
    _ELL_INT = ell_min_arr[:, None] + np.arange(_L_MAX)[None, :]
    _ELL_MASK = (_ELL_INT <= ell_max_arr[:, None]).astype(np.float64)
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
                   concentration=ConstantConcentration(c=4.0),
                   convert_masses=True,
                   hm_consistency=False)
    prof = ParametricGNFWPressureProfile(A_SZ=A_SZ, alpha_SZ=ALPHA_SZ, B=B)
    tsz = tSZTracer(profile=prof)

    m_grid = jnp.geomspace(M_MIN, M_MAX, 100)
    z_grid = jnp.geomspace(5e-3, 3.0, 100)
    coeff, _ = load_sigma_y0_curve()
    coeff_j = jnp.asarray(coeff)
    snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=A_SZ, alpha_SZ=ALPHA_SZ,
                         B=B, coeff=coeff_j)
    mask = conditional_An_undetected(snr, sigma_lnY=SIGMA_LNY, q_cat=Q_CAT,
                                     n_power=2, n_grid=1024, nsig=8.0)

    # 1) hmfast on sparse geomspace(9, 1085, 50) then log-interp bin
    ell_sparse = jnp.geomspace(LMIN, LMAX, 50)
    cl_sparse = np.asarray(jax.device_get(
        hm.cl_1h_masked(tsz, None, ell_sparse, m_grid, z_grid, mask, k_damp=0.0)))
    Dl_hm_sparse = bin_log_interp(np.asarray(ell_sparse), cl_sparse)

    # 2) hmfast on EVERY INTEGER ELL (1077 points), then bin exactly
    print(f"Computing hmfast Cl at every integer ell ({len(ell_int)} points)...")
    t0 = time.perf_counter()
    cl_dense = np.asarray(jax.device_get(
        hm.cl_1h_masked(tsz, None, jnp.asarray(ell_int), m_grid, z_grid, mask, k_damp=0.0)))
    print(f"  done in {time.perf_counter()-t0:.2f}s")
    Dl_hm_dense = bin_exact(cl_dense)

    # 3) tszpower reference
    classy_sz.set({**_FIXED_ASTRO,
                   "H0": FID["H0"], "omega_b": FID["omega_b"], "omega_cdm": FID["omega_cdm"],
                   "ln10^{10}A_s": FID["ln10_10A_s"], "n_s": FID["n_s"],
                   "tau_reio": FID["tau_reio"], "B": B})
    _tszp_initialise()
    params = {"H0": FID["H0"], "omega_b": FID["omega_b"], "omega_cdm": FID["omega_cdm"],
              "ln10^{10}A_s": FID["ln10_10A_s"], "n_s": FID["n_s"],
              "tau_reio": FID["tau_reio"], "B": B, **_FIXED_ASTRO}
    Dl_tszp = np.asarray(jax.device_get(pp.compute_masked_tsz_Dell_binned_parametric(
        params_values_dict=params, A_SZ=A_SZ, alpha_SZ=ALPHA_SZ,
        q_cat=Q_CAT, sigma_lnY=SIGMA_LNY)))

    print(f"\n{'band':>4s} {'ell_eff':>8s} {'hm/tszp (sparse)':>18s} {'hm/tszp (dense)':>18s}")
    print("-" * 60)
    for i in range(18):
        r1 = Dl_hm_sparse[i] / Dl_tszp[i]
        r2 = Dl_hm_dense[i]  / Dl_tszp[i]
        print(f"{i:4d} {ELL_EFF[i]:8.1f} {r1:18.6f} {r2:18.6f}")
    print()
    print(f"Max |sparse/tszp - 1|: {np.max(np.abs(Dl_hm_sparse/Dl_tszp - 1.0))*100:.3f}%")
    print(f"Max |dense /tszp - 1|: {np.max(np.abs(Dl_hm_dense /Dl_tszp - 1.0))*100:.3f}%")


if __name__ == "__main__":
    main()
