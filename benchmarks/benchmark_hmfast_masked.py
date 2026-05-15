"""
Benchmark hmfast masked tSZ D_ell standalone, then compare to reference data.

The reference data + covmat is saved at:
  /scratch/scratch-lxu/tsz_cnc_scatter/synthetic_data/
  - Dl_binned_masked_signal_only_real0.txt (ell_eff, D_ell)
  - covmat_Dl_binned_masked_signal_only.txt (18x18 covmat)

Reference theory values from the check notebook:
  Uses tszpower's compute_masked_tsz_Dell_binned_parametric().
"""
from __future__ import annotations

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

# Planck bands (same as tszpower)
ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78,
     102, 133, 173, 224, 292, 380, 494, 642, 835], dtype=int
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102,
     133, 173, 224, 292, 380, 494, 642, 835, 1085], dtype=int
)
ELL_EFF = np.array(
    [10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5,
     117.0, 152.5, 198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5]
)

_L_MAX = int(np.max(ELL_MAX - ELL_MIN + 1))
_ELL_INT = ELL_MIN[:, None] + np.arange(_L_MAX)[None, :]
_ELL_MASK = (_ELL_INT <= ELL_MAX[:, None]).astype(np.float64)


def bin_to_18(ell_in, Cl_in):
    """tszpower-compatible binning: interpolate C_ell -> D_ell, average."""
    log_ell = np.log(np.asarray(ell_in))
    Cl_q = np.interp(np.log(_ELL_INT.astype(float)), log_ell, np.asarray(Cl_in))
    Dl_q = _ELL_INT * (_ELL_INT + 1.0) * Cl_q / (2.0 * np.pi) * 1e12
    num = np.sum(Dl_q * _ELL_MASK, axis=1)
    den = np.sum(_ELL_MASK, axis=1)
    return num / den


def main():
    # Parameters from the reference notebook (cell-2 and cell-5)
    # Cosmo: H0=67.66, omega_b=0.02242, omega_cdm=0.1193, lnAs=2.9718, n_s=0.9665, tau_reio=0.0544
    # Astro: B=1.41, A_SZ=-4.2373, alpha_SZ=1.12, sigma_lnY=0.173, q_cat=5.0
    params_notebook = dict(
        H0=67.66,
        omega_b=0.02242,
        omega_cdm=0.1193,
        ln10_10A_s=2.9718,
        n_s=0.9665,
        tau_reio=0.0544,
        B=1.41,
        A_SZ=-4.2373,
        alpha_SZ=1.12,
        sigma_lnY=0.173,
        q_cat=5.0,
    )
    # Also test the chain YAML parameters
    params_chain = dict(
        H0=73.8,
        omega_b=0.022,
        omega_cdm=0.119,
        ln10_10A_s=2.9718,
        n_s=0.962,
        tau_reio=0.0544,
        B=1.0 / 0.709,
        A_SZ=-4.31,
        alpha_SZ=1.12,
        sigma_lnY=0.173,
        q_cat=5.0,
    )

    # Load noise curve once
    coeff, _ = load_sigma_y0_curve()
    coeff_j = jnp.asarray(coeff)

    # Grids matching tszpower N_GRID_MZ=100
    M_MIN = 6.766e13
    M_MAX = 6.766e15

    for label, p in [("notebook H0=67.66", params_notebook), ("chain H0=73.8", params_chain)]:
        print(f"\n{'='*60}")
        print(f"hmfast masked tSZ: {label}")
        print(f"{'='*60}")
        print(f"  A_SZ={p['A_SZ']}, alpha_SZ={p['alpha_SZ']}, sigma_lnY={p['sigma_lnY']}, "
              f"q_cat={p['q_cat']}, B={p['B']:.4f}, H0={p['H0']}")

        cosmo = Cosmology(emulator_set="lcdm:v1")
        cosmo = cosmo.update(
            H0=p["H0"], omega_cdm=p["omega_cdm"], omega_b=p["omega_b"],
            ln1e10A_s=p["ln10_10A_s"], n_s=p["n_s"], tau_reio=p["tau_reio"],
        )
        hm = HaloModel(
            cosmology=cosmo,
            mass_definition=MassDefinition(500, "critical"),
            convert_masses=True,
        )
        prof = ParametricGNFWPressureProfile(
            A_SZ=p["A_SZ"], alpha_SZ=p["alpha_SZ"], B=p["B"],
        )
        tsz = tSZTracer(profile=prof)

        ell_int = jnp.geomspace(float(ELL_MIN[0]), float(ELL_MAX[-1]), 50)
        m_grid = jnp.geomspace(M_MIN, M_MAX, 100)
        z_grid = jnp.geomspace(0.005, 3.0, 100)

        # Warmup
        print("  Warming up...")
        t0 = time.perf_counter()
        snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=p["A_SZ"], alpha_SZ=p["alpha_SZ"],
                             B=p["B"], coeff=coeff_j)
        mask = conditional_An_undetected(snr, sigma_lnY=p["sigma_lnY"], q_cat=p["q_cat"],
                                         n_power=2, n_grid=1024, nsig=8.0)
        cl_int = hm.cl_1h_masked(tsz, None, ell_int, m_grid, z_grid, mask, k_damp=0.0)
        block = getattr(jax, "block_until_ready", lambda x: x)
        block(cl_int)
        print(f"  Warmup: {time.perf_counter() - t0:.3f}s")

        # Timed runs
        times = []
        for i in range(10):
            t0 = time.perf_counter()
            snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=p["A_SZ"], alpha_SZ=p["alpha_SZ"],
                                 B=p["B"], coeff=coeff_j)
            mask = conditional_An_undetected(snr, sigma_lnY=p["sigma_lnY"], q_cat=p["q_cat"],
                                             n_power=2, n_grid=1024, nsig=8.0)
            cl_int = hm.cl_1h_masked(tsz, None, ell_int, m_grid, z_grid, mask, k_damp=0.0)
            block(cl_int)
            times.append(time.perf_counter() - t0)

        times = np.array(times)
        print(f"  Timing: {times.mean()*1000:.2f} +/- {times.std()*1000:.3f} ms per sample "
              f"(min={times.min()*1000:.2f}, max={times.max()*1000:.2f})")

        Dl_hmfast = bin_to_18(np.asarray(ell_int), np.asarray(cl_int))

        print(f"\n  {'Band':>5s} {'ell_eff':>8s} {'hmfast D_l':>14s}")
        print(f"  {'-'*32}")
        for i in range(18):
            print(f"  {i:5d} {ELL_EFF[i]:8.1f} {Dl_hmfast[i]:14.8f}")

        # Compare with reference data if available
        ref_path = "/scratch/scratch-lxu/tsz_cnc_scatter/synthetic_data/Dl_binned_masked_signal_only_real0.txt"
        try:
            ref_data = np.loadtxt(ref_path)
            ref_Dl = ref_data[:, 1]
            print(f"\n  --- Comparison with reference data ({ref_path}) ---")
            print(f"  {'ell_eff':>8s} {'hmfast':>14s} {'ref_data':>14s} {'ratio':>10s} {'diff%':>8s}")
            print(f"  {'-'*60}")
            max_diff = 0
            for i in range(18):
                r = Dl_hmfast[i] / ref_Dl[i]
                d = abs(Dl_hmfast[i] - ref_Dl[i]) / abs(ref_Dl[i]) * 100
                max_diff = max(max_diff, d)
                print(f"  {ELL_EFF[i]:8.1f} {Dl_hmfast[i]:14.8f} {ref_Dl[i]:14.8f} {r:10.4f} {d:8.3f}")
            print(f"  Max difference: {max_diff:.3f}%")
        except Exception as e:
            print(f"  Cannot compare with reference: {e}")

        # Also print raw C_ell for debugging
        print(f"\n  Raw C_ell (before binning):")
        for i in range(0, len(ell_int), max(1, len(ell_int) // 10)):
            print(f"    ell={ell_int[i]:8.2f}  C_ell={cl_int[i]:.6e}")


if __name__ == "__main__":
    main()
