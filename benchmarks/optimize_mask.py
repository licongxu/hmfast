"""
Profile mask kernel accuracy vs speed trade-offs.

Target: 0.003s per sample total. Currently at 0.0038s, mask step is 0.9ms.
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

ELL_MIN = np.array([9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835], dtype=int)
ELL_MAX = np.array([12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085], dtype=int)


def main():
    H0 = 73.8
    omega_cdm = 0.119
    omega_b = 0.022
    ln10_10A_s = 2.9718
    n_s = 0.962
    tau_reio = 0.0544
    B_val = 1.0 / 0.709
    A_SZ = -4.31
    alpha_SZ = 1.12
    sigma_lnY = 0.173
    q_cat = 5.0

    coeff, _ = load_sigma_y0_curve()
    coeff_j = jnp.asarray(coeff)

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
    block = getattr(jax, "block_until_ready", lambda x: x)

    # Pre-compute SNR grid
    snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=A_SZ, alpha_SZ=alpha_SZ,
                         B=B_val, coeff=coeff_j)
    block(snr)

    # Reference mask (high accuracy)
    mask_ref = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat,
                                         n_power=2, n_grid=2048, nsig=8.0)
    cl_ref = hm.cl_1h_masked(tsz, None, ell_int, m_grid, z_grid, mask_ref, k_damp=0.0)
    block(mask_ref)
    block(cl_ref)

    # Test different grid sizes
    print("Testing mask grid accuracy vs speed:")
    print(f"{'n_grid':>8s} {'nsig':>6s} {'time_ms':>10s} {'max_rel_err':>14s} {'mean_rel_err':>14s}")
    print("-" * 60)

    for n_grid in [128, 256, 512, 1024]:
        for nsig in [4.0, 6.0, 8.0]:
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                mask = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat,
                                                 n_power=2, n_grid=n_grid, nsig=nsig)
                block(mask)
                times.append(time.perf_counter() - t0)

            err = jnp.abs(mask - mask_ref) / jnp.maximum(mask_ref, 1e-20)
            max_err = float(jnp.max(err))
            mean_err = float(jnp.mean(err))

            print(f"{n_grid:8d} {nsig:6.1f} {np.mean(times)*1000:10.3f} "
                  f"{max_err:14.6e} {mean_err:14.6e}")

    # Test n_grid_mz reduction (M,z grid)
    print(f"\nTesting M,z grid reduction (n_grid_mz):")
    print(f"{'n_mz':>6s} {'time_ms':>10s}")
    print("-" * 20)

    for n_mz in [50, 75, 100]:
        m_grid_s = jnp.geomspace(6.766e13, 6.766e15, n_mz)
        z_grid_s = jnp.geomspace(0.005, 3.0, n_mz)

        t0 = time.perf_counter()
        snr = build_snr_grid(hm, m_grid_s, z_grid_s, A_SZ=A_SZ, alpha_SZ=alpha_SZ,
                             B=B_val, coeff=coeff_j)
        mask = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat,
                                         n_power=2, n_grid=512, nsig=6.0)
        cl = hm.cl_1h_masked(tsz, None, ell_int, m_grid_s, z_grid_s, mask, k_damp=0.0)
        block(cl)
        elapsed = time.perf_counter() - t0
        print(f"{n_mz:6d} {elapsed*1000:10.3f}")

    # Best config: n_grid=512, nsig=6.0 for mask, keep n_mz=100
    print(f"\nBest config total timing:")
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        snr = build_snr_grid(hm, m_grid, z_grid, A_SZ=A_SZ, alpha_SZ=alpha_SZ,
                             B=B_val, coeff=coeff_j)
        mask = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat,
                                         n_power=2, n_grid=512, nsig=6.0)
        cl = hm.cl_1h_masked(tsz, None, ell_int, m_grid, z_grid, mask, k_damp=0.0)
        block(cl)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    print(f"  Total: {times.mean()*1000:.3f} +/- {times.std()*1000:.3f} ms (min={times.min()*1000:.3f})")

    # Compare cl with reference
    cl_err = jnp.abs(cl - cl_ref) / jnp.maximum(cl_ref, 1e-30)
    print(f"  C_ell max rel err vs ref: {float(jnp.max(cl_err)):.6e}")
    print(f"  C_ell mean rel err vs ref: {float(jnp.mean(cl_err)):.6e}")


if __name__ == "__main__":
    main()
