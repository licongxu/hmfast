"""
Diagnose discrepancy between hmfast and tszpower masked tSZ C_ell.

Checks: unmasked C_ell, SNR grid, mask values step by step.
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
from hmfast.halos.profiles import ParametricGNFWPressureProfile, GNFWPressureProfile
from hmfast.tracers import tSZTracer
from hmfast.tracers.tsz_completeness import (
    build_snr_grid,
    conditional_An_undetected,
    load_sigma_y0_curve,
    compute_y0_parametric,
    compute_theta500_arcmin,
)

ELL_MIN = np.array([9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835], dtype=int)
ELL_MAX = np.array([12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085], dtype=int)
ELL_EFF = np.array([10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5, 117.0, 152.5, 198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5])


def main():
    # Use notebook parameters (H0=67.66, B=1.41, A_SZ=-4.2373)
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

    M_MIN = 6.766e13
    M_MAX = 6.766e15
    Z_MIN = 0.005
    Z_MAX = 3.0

    print("=== Diagnostic: hmfast masked tSZ ===")
    print(f"H0={H0}, omega_cdm={omega_cdm}, omega_b={omega_b}")
    print(f"B={B_val}, A_SZ={A_SZ}, alpha_SZ={alpha_SZ}, sigma_lnY={sigma_lnY}, q_cat={q_cat}")

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

    # Compute with BOTH GNFW and ParametricGNFW to compare
    prof_orig = GNFWPressureProfile(B=B_val, P0=8.130, c500=1.156,
                                    alpha=1.0620, beta=5.4807, gamma=0.3292)
    prof_param = ParametricGNFWPressureProfile(
        A_SZ=A_SZ, alpha_SZ=alpha_SZ, B=B_val,
        P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292,
    )
    tsz_orig = tSZTracer(profile=prof_orig)
    tsz_param = tSZTracer(profile=prof_param)

    ell_int = jnp.geomspace(float(ELL_MIN[0]), float(ELL_MAX[-1]), 50)
    m_grid = jnp.geomspace(M_MIN, M_MAX, 100)
    z_grid = jnp.geomspace(Z_MIN, Z_MAX, 100)

    # --- 1. Compare unmasked C_ell (GNFW vs ParametricGNFW) ---
    print("\n--- 1. Unmasked C_ell comparison ---")
    block = getattr(jax, "block_until_ready", lambda x: x)

    print("Computing GNFW C_ell...")
    cl_orig = hm.cl_1h(tsz_orig, None, ell_int, m_grid, z_grid, k_damp=0.0)
    block(cl_orig)

    print("Computing ParametricGNFW C_ell...")
    cl_param = hm.cl_1h(tsz_param, None, ell_int, m_grid, z_grid, k_damp=0.0)
    block(cl_param)

    ratio_cl = cl_param / cl_orig
    print(f"  C_ell ratio (Parametric/Original): min={ratio_cl.min():.4f}, "
          f"max={ratio_cl.max():.4f}, mean={ratio_cl.mean():.4f}")
    print(f"  First few ratios: {ratio_cl[:5]}")

    # --- 2. Check profile u_k for a single mass bin ---
    print("\n--- 2. Profile u_k check ---")
    ell_test = jnp.logspace(1, 3, 20)
    m_test = jnp.logspace(np.log10(1e14), np.log10(1e15), 5)
    z_test = jnp.array([0.5])

    uk_orig = prof_orig.u_k(hm, ell_test, m_test, z_test)
    uk_param = prof_param.u_k(hm, ell_test, m_test, z_test)
    ratio_uk = uk_param / uk_orig
    print(f"  u_k ratio shape: {ratio_uk.shape}")
    # The ratio should be constant across ell and r, varying only with (m,z)
    ratio_mz = ratio_uk.mean(axis=0)  # average over ell
    ratio_mz_std = ratio_uk.std(axis=0)  # std over ell
    print(f"  u_k ratio (avg over ell) for 5 masses at z=0.5: {ratio_mz[:, 0]}")
    print(f"  u_k ratio std over ell: {ratio_mz_std[:, 0]}")

    # The ratio should equal y0_param / y0_orig
    y0_param = compute_y0_parametric(hm, m_test, z_test, A_SZ, alpha_SZ, B_val)
    print(f"  y0_param at z=0.5: {y0_param[:, 0]}")

    # --- 3. Check SNR grid ---
    print("\n--- 3. SNR grid ---")
    coeff, _ = load_sigma_y0_curve()
    snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=A_SZ, alpha_SZ=alpha_SZ,
                         B=B_val, coeff=jnp.asarray(coeff))
    print(f"  SNR shape: {snr.shape}")
    print(f"  SNR range: [{snr.min():.3f}, {snr.max():.3f}]")
    print(f"  SNR at a few points: {snr[50, 50]:.3f}, {snr[80, 50]:.3f}, {snr[95, 50]:.3f}")

    # --- 4. Check mask ---
    print("\n--- 4. Mask (conditional_An_undetected) ---")
    mask = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat,
                                     n_power=2, n_grid=1024, nsig=8.0)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask range: [{mask.min():.6f}, {mask.max():.6f}]")
    print(f"  Fraction of (M,z) with mask > 0.01: {(mask > 0.01).mean():.3f}")
    print(f"  Fraction of (M,z) with mask > 0.1: {(mask > 0.1).mean():.3f}")

    # --- 5. Masked C_ell ---
    print("\n--- 5. Masked C_ell ---")
    cl_masked = hm.cl_1h_masked(tsz_param, None, ell_int, m_grid, z_grid, mask, k_damp=0.0)
    block(cl_masked)

    # Convert to D_ell
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

    dl_masked = bin_to_18(np.asarray(ell_int), np.asarray(cl_masked))
    dl_orig = bin_to_18(np.asarray(ell_int), np.asarray(cl_orig))
    dl_param = bin_to_18(np.asarray(ell_int), np.asarray(cl_param))

    print(f"  D_ell (masked) vs D_ell (unmasked GNFW):")
    for i in range(18):
        print(f"    ell={ELL_EFF[i]:7.1f}  "
              f"masked={dl_masked[i]:.6e}  "
              f"unmasked_gnfw={dl_orig[i]:.6e}  "
              f"unmasked_param={dl_param[i]:.6e}  "
              f"mask/unmask_ratio={dl_masked[i]/dl_orig[i]:.4f}")

    # --- 6. Timing breakdown ---
    print("\n--- 6. Timing breakdown ---")
    block = getattr(jax, "block_until_ready", lambda x: x)

    # Time each step separately
    t_snr = []
    for _ in range(10):
        t0 = time.perf_counter()
        snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=A_SZ, alpha_SZ=alpha_SZ,
                             B=B_val, coeff=jnp.asarray(coeff))
        block(snr)
        t_snr.append(time.perf_counter() - t0)

    t_mask = []
    for _ in range(10):
        t0 = time.perf_counter()
        mask = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat,
                                         n_power=2, n_grid=1024, nsig=8.0)
        block(mask)
        t_mask.append(time.perf_counter() - t0)

    t_cl = []
    for _ in range(10):
        t0 = time.perf_counter()
        cl = hm.cl_1h_masked(tsz_param, None, ell_int, m_grid, z_grid, mask, k_damp=0.0)
        block(cl)
        t_cl.append(time.perf_counter() - t0)

    t_total = []
    for _ in range(10):
        t0 = time.perf_counter()
        snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=A_SZ, alpha_SZ=alpha_SZ,
                             B=B_val, coeff=jnp.asarray(coeff))
        mask = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat,
                                         n_power=2, n_grid=1024, nsig=8.0)
        cl = hm.cl_1h_masked(tsz_param, None, ell_int, m_grid, z_grid, mask, k_damp=0.0)
        block(cl)
        t_total.append(time.perf_counter() - t0)

    print(f"  SNR grid:     {np.mean(t_snr)*1000:.3f} +/- {np.std(t_snr)*1000:.3f} ms")
    print(f"  Mask compute:  {np.mean(t_mask)*1000:.3f} +/- {np.std(t_mask)*1000:.3f} ms")
    print(f"  cl_1h_masked: {np.mean(t_cl)*1000:.3f} +/- {np.std(t_cl)*1000:.3f} ms")
    print(f"  Total:        {np.mean(t_total)*1000:.3f} +/- {np.std(t_total)*1000:.3f} ms")
    print(f"  Min total:    {np.min(t_total)*1000:.3f} ms")


if __name__ == "__main__":
    main()
