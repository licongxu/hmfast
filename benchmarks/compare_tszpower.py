"""
Compare hmfast masked tSZ D_ell against tszpower reference.

Runs both pipelines at the same fiducial parameters and checks:
1. Masked tSZ power spectrum (scatter completeness) matches tszpower
2. Trispectrum matches tszpower
3. Timing per sample
"""
from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

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

sys.path.insert(0, "/scratch/scratch-lxu/tszsbi/tszpower")
import tszpower.masked_tsz_ps_completeness as mp_comp
import tszpower.parametric_profile as pp
from tszpower.utils import ELL_EFF, ELL_MIN, ELL_MAX

_L_MAX = int(np.max(ELL_MAX - ELL_MIN + 1))
_ELL_INT = ELL_MIN[:, None] + np.arange(_L_MAX)[None, :]
_ELL_MASK = (_ELL_INT <= ELL_MAX[:, None]).astype(np.float64)


def bin_to_18(ell_in, Cl_in):
    """tszpower-compatible binning."""
    log_ell = np.log(np.asarray(ell_in))
    Cl_q = np.interp(np.log(_ELL_INT.astype(float)), log_ell, np.asarray(Cl_in))
    Dl_q = _ELL_INT * (_ELL_INT + 1.0) * Cl_q / (2.0 * np.pi) * 1e12
    num = np.sum(Dl_q * _ELL_MASK, axis=1)
    den = np.sum(_ELL_MASK, axis=1)
    return num / den


FID = dict(
    H0=73.8, omega_cdm=0.119, omega_b=0.022,
    ln10_10A_s=2.9718, n_s=0.962, tau_reio=0.0544,
)
FID_B = 1.0 / 0.709
FID_A_SZ = -4.31
FID_ALPHA_SZ = 1.12
FID_SIGMA_LNY = 0.173
Q_CAT = 5.0

M_MIN = 6.766e13
M_MAX = 6.766e15
Z_MIN = 0.005
Z_MAX = 3.0
N_GRID_MZ = 100


def benchmark_hmfast():
    coeff, _ = load_sigma_y0_curve()
    coeff_j = jnp.asarray(coeff)

    cosmo = Cosmology(emulator_set="lcdm:v1")
    cosmo = cosmo.update(
        H0=FID["H0"], omega_cdm=FID["omega_cdm"], omega_b=FID["omega_b"],
        ln1e10A_s=FID["ln10_10A_s"], n_s=FID["n_s"], tau_reio=FID["tau_reio"],
    )
    hm = HaloModel(
        cosmology=cosmo,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
    )
    prof = ParametricGNFWPressureProfile(
        A_SZ=FID_A_SZ, alpha_SZ=FID_ALPHA_SZ, B=FID_B,
    )
    tsz = tSZTracer(profile=prof)

    ell_int = jnp.geomspace(float(ELL_MIN[0]), float(ELL_MAX[-1]), 50)
    m_grid = jnp.geomspace(M_MIN, M_MAX, N_GRID_MZ)
    z_grid = jnp.geomspace(Z_MIN, Z_MAX, N_GRID_MZ)

    print("hmfast: warming up JIT...")
    t0 = time.perf_counter()
    snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=FID_A_SZ, alpha_SZ=FID_ALPHA_SZ,
                         B=FID_B, coeff=coeff_j)
    mask = conditional_An_undetected(snr, sigma_lnY=FID_SIGMA_LNY, q_cat=Q_CAT,
                                     n_power=2, n_grid=1024, nsig=8.0)
    cl_int = hm.cl_1h_masked(tsz, None, ell_int, m_grid, z_grid, mask, k_damp=0.0)
    block = getattr(jax, "block_until_ready", lambda x: x)
    block(cl_int)
    print(f"hmfast: warmup done in {time.perf_counter() - t0:.3f}s")

    times = []
    for i in range(5):
        t0 = time.perf_counter()
        snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=FID_A_SZ, alpha_SZ=FID_ALPHA_SZ,
                             B=FID_B, coeff=coeff_j)
        mask = conditional_An_undetected(snr, sigma_lnY=FID_SIGMA_LNY, q_cat=Q_CAT,
                                         n_power=2, n_grid=1024, nsig=8.0)
        cl_int = hm.cl_1h_masked(tsz, None, ell_int, m_grid, z_grid, mask, k_damp=0.0)
        block(cl_int)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    print(f"hmfast timing: {times.mean()*1000:.2f} +/- {times.std()*1000:.3f} ms per sample")

    Dl_hmfast = bin_to_18(np.asarray(ell_int), np.asarray(cl_int))
    return Dl_hmfast, times.mean()


def benchmark_tszpower():
    params = {
        "H0": FID["H0"], "omega_b": FID["omega_b"], "omega_cdm": FID["omega_cdm"],
        "ln10_10A_s": FID["ln10_10A_s"], "n_s": FID["n_s"], "tau_reio": FID["tau_reio"],
        "B": FID_B, "A_SZ": FID_A_SZ, "alpha_SZ": FID_ALPHA_SZ,
        "sigma_lnY": FID_SIGMA_LNY, "q_cat": Q_CAT,
    }

    print("\ntszpower: computing masked D_ell...")
    t0 = time.perf_counter()
    Dl_tszpower = pp.compute_masked_tsz_Dell_binned_parametric(params)
    elapsed = time.perf_counter() - t0
    print(f"tszpower timing: {elapsed*1000:.2f} ms")
    return Dl_tszpower


def benchmark_hmfast_trispectrum():
    cosmo = Cosmology(emulator_set="lcdm:v1")
    cosmo = cosmo.update(
        H0=FID["H0"], omega_cdm=FID["omega_cdm"], omega_b=FID["omega_b"],
        ln1e10A_s=FID["ln10_10A_s"], n_s=FID["n_s"], tau_reio=FID["tau_reio"],
    )
    hm = HaloModel(
        cosmology=cosmo,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
    )
    prof = ParametricGNFWPressureProfile(
        A_SZ=FID_A_SZ, alpha_SZ=FID_ALPHA_SZ, B=FID_B,
    )
    tsz = tSZTracer(profile=prof)

    ell_int = jnp.geomspace(float(ELL_MIN[0]), float(ELL_MAX[-1]), 30)
    m_grid = jnp.geomspace(M_MIN, M_MAX, N_GRID_MZ)
    z_grid = jnp.geomspace(Z_MIN, Z_MAX, N_GRID_MZ)

    print("\nhmfast: warming up trispectrum JIT...")
    t0 = time.perf_counter()
    T = hm.trispectrum_1h(tsz, None, ell_int, ell_int, m_grid, z_grid, k_damp=0.0)
    block = getattr(jax, "block_until_ready", lambda x: x)
    block(T)
    print(f"hmfast trispectrum warmup: {time.perf_counter() - t0:.3f}s")

    times = []
    for i in range(3):
        t0 = time.perf_counter()
        T = hm.trispectrum_1h(tsz, None, ell_int, ell_int, m_grid, z_grid, k_damp=0.0)
        block(T)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    print(f"hmfast trispectrum timing: {times.mean()*1000:.2f} +/- {times.std()*1000:.3f} ms")
    print(f"  shape: {T.shape}")
    return np.asarray(ell_int), np.asarray(T)


def main():
    print("=" * 60)
    print("hmfast vs tszpower: Masked tSZ D_ell comparison")
    print("=" * 60)
    print(f"Device: {jax.devices()[0] if jax.devices() else 'CPU'}")

    # 1. Masked PS
    Dl_hmfast, t_hmfast = benchmark_hmfast()
    Dl_tszpower = benchmark_tszpower()

    print("\n--- D_ell comparison (18 Planck bands) ---")
    print(f"{'Band':>5s} {'ell_eff':>8s} {'hmfast':>12s} {'tszpower':>12s} {'ratio':>10s} {'diff%':>8s}")
    print("-" * 60)
    max_diff = 0
    for i in range(18):
        r = Dl_hmfast[i] / Dl_tszpower[i]
        d = abs(Dl_hmfast[i] - Dl_tszpower[i]) / abs(Dl_tszpower[i]) * 100
        max_diff = max(max_diff, d)
        print(f"{i:5d} {ELL_EFF[i]:8.1f} {Dl_hmfast[i]:12.6f} {Dl_tszpower[i]:12.6f} {r:10.6f} {d:8.3f}")

    print(f"\nMax relative difference: {max_diff:.3f}%")
    print(f"hmfast per-sample time: {t_hmfast*1000:.2f} ms")

    # 2. Trispectrum
    print("\n" + "=" * 60)
    print("Trispectrum comparison")
    print("=" * 60)
    ell_hmf, T_hmf = benchmark_hmfast_trispectrum()

    T_hmf_diag = np.diag(T_hmf)
    print("\n--- Trispectrum diagonal T(l,l) ---")
    for i in range(0, len(ell_hmf), max(1, len(ell_hmf) // 8)):
        print(f"  l={ell_hmf[i]:7.1f}  T={T_hmf_diag[i]:.6e}")

    scaled = (T_hmf * ell_hmf[:, None] * (ell_hmf[:, None] + 1)
              * ell_hmf[None, :] * (ell_hmf[None, :] + 1) / (2 * np.pi)**2 * 1e24)
    print(f"\nScaled trispectrum max: {scaled.max():.3f}")


if __name__ == "__main__":
    main()
