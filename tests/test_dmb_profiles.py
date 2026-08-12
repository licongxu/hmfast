"""
Tests for DMB (GODMAX BCM_18_wP) profiles in hmfast.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import (
    DMBPressureProfile,
    DMBMatterProfile,
    DMBNFWMatterProfile,
    DMBGasDensityProfile,
)
from hmfast.halos.profiles.dmb import dmb_halo_quantities, _DEFAULTS
from hmfast.tracers import tSZTracer


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GODMAX_SRC = os.path.join(REPO_ROOT, "ref_packages", "GODMAX", "src")


@pytest.fixture(scope="module")
def cosmology():
    return Cosmology(emulator_set="lcdm:v1")


@pytest.fixture(scope="module")
def halo_model(cosmology):
    return HaloModel(cosmology=cosmology)


def _cosmo_ob_om_h(cosmology):
    p = cosmology._cosmo_params()
    return float(p["Omega_b"]), float(p["Omega0_m"]), float(p["h"])


@pytest.mark.skipif(
    not os.path.isdir(GODMAX_SRC), reason="GODMAX reference package not present"
)
def test_dmb_pe_and_rho_vs_godmax(cosmology):
    """Pointwise Pe and rho_dmb agree with GODMAX BCM_18_wP under matched settings."""
    if GODMAX_SRC not in sys.path:
        sys.path.insert(0, GODMAX_SRC)
    from get_BCMP_profile_jit import BCM_18_wP

    Ob0, Om0, h = _cosmo_ob_om_h(cosmology)
    H0 = 100.0 * h
    M_h = 1e14
    z0 = 0.2
    c0 = 5.0

    sim = dict(
        cosmo=dict(
            H0=H0,
            Om0=Om0,
            Ob0=Ob0,
            sigma8=0.81,
            ns=0.96,
            w0=-1.0,
        ),
        theta_ej_0=4.0,
        theta_co_0=0.1,
        log10_Mc0=14.83,
        mu_beta=0.21,
        eta_star=0.3,
        eta_cga=0.6,
        A_starcga=0.09,
        log10_M1_starcga=11.4,
        alpha_nt=0.18,
        beta_nt=0.5,
        n_nt=0.3,
        gamma_rhogas=2.0,
        delta_rhogas=7.0,
        nfw_trunc=True,
        epsilon_rt=4.0,
    )
    halo = dict(
        rmin=5e-3,
        rmax=3.0,
        nr=32,
        z_array=[z0],
        lg10_Mmin=np.log10(M_h),
        lg10_Mmax=np.log10(M_h),
        nM=1,
        cmin=c0,
        cmax=c0,
        nc=1,
    )
    bcmp = BCM_18_wP(sim, halo, num_points_trapz_int=48)

    m_phys = M_h / h
    q = dmb_halo_quantities(
        m_phys,
        z0,
        c0,
        Ob0,
        Om0,
        h,
        {**_DEFAULTS, "num_points_trapz_int": 48},
    )

    r_god = np.asarray(bcmp.r_array) / h  # comoving Mpc
    r200 = float(q["r200c_comoving"])
    mask = (r_god > 0.05 * r200) & (r_god < 3.0 * r200)
    mask_core = (r_god > 0.05 * r200) & (r_god < 1.5 * r200)

    pe_h = np.exp(
        np.interp(
            np.log(r_god),
            np.log(np.asarray(q["r_comoving"])),
            np.log(np.asarray(q["Pe"]) + 1e-30),
        )
    )
    pe_g = np.asarray(bcmp.Pe_mat_physical[:, 0, 0, 0]) * 1000.0  # keV → eV
    rel_pe = np.abs(pe_h[mask] - pe_g[mask]) / np.maximum(pe_g[mask], 1e-30)
    rel_pe_core = np.abs(pe_h[mask_core] - pe_g[mask_core]) / np.maximum(
        pe_g[mask_core], 1e-30
    )
    med_pe, max_pe = float(np.median(rel_pe)), float(np.max(rel_pe_core))
    # Cumtrapz ζ–HSE path must stay within these GODMAX gates (correctness priority).
    # Median over 0.05–3 R200c; max over 0.05–1.5 R200c (outer HSE is quadrature-sensitive).
    assert med_pe < 0.02, f"Pe median rel err {med_pe}"
    assert max_pe < 0.05, f"Pe max rel err (core) {max_pe}"

    rho_h = np.exp(
        np.interp(
            np.log(r_god),
            np.log(np.asarray(q["r_comoving"])),
            np.log(np.asarray(q["rho_dmb"]) + 1e-30),
        )
    )
    rho_g = np.asarray(bcmp.rho_dmb_mat[:, 0, 0, 0]) * h**2
    rel_rho = np.abs(rho_h[mask] - rho_g[mask]) / np.maximum(rho_g[mask], 1e-30)
    rel_rho_core = np.abs(rho_h[mask_core] - rho_g[mask_core]) / np.maximum(
        rho_g[mask_core], 1e-30
    )
    med_rho, max_rho = float(np.median(rel_rho)), float(np.max(rel_rho_core))
    assert med_rho < 0.02, f"rho median rel err {med_rho}"
    assert max_rho < 0.05, f"rho max rel err (core) {max_rho}"


@pytest.mark.skipif(
    not os.path.isdir(GODMAX_SRC), reason="GODMAX reference package not present"
)
def test_dmb_pe_vs_godmax_second_halo(cosmology):
    """Second (M,z) point: vectorized ζ–HSE still tracks GODMAX."""
    if GODMAX_SRC not in sys.path:
        sys.path.insert(0, GODMAX_SRC)
    from get_BCMP_profile_jit import BCM_18_wP

    Ob0, Om0, h = _cosmo_ob_om_h(cosmology)
    H0 = 100.0 * h
    M_h, z0, c0 = 3e14, 0.5, 4.0
    sim = dict(
        cosmo=dict(H0=H0, Om0=Om0, Ob0=Ob0, sigma8=0.81, ns=0.96, w0=-1.0),
        theta_ej_0=4.0,
        theta_co_0=0.1,
        log10_Mc0=14.83,
        mu_beta=0.21,
        eta_star=0.3,
        eta_cga=0.6,
        A_starcga=0.09,
        log10_M1_starcga=11.4,
        alpha_nt=0.18,
        beta_nt=0.5,
        n_nt=0.3,
        gamma_rhogas=2.0,
        delta_rhogas=7.0,
        nfw_trunc=True,
        epsilon_rt=4.0,
    )
    halo = dict(
        rmin=5e-3,
        rmax=3.0,
        nr=32,
        z_array=[z0],
        lg10_Mmin=np.log10(M_h),
        lg10_Mmax=np.log10(M_h),
        nM=1,
        cmin=c0,
        cmax=c0,
        nc=1,
    )
    bcmp = BCM_18_wP(sim, halo, num_points_trapz_int=48)
    q = dmb_halo_quantities(
        M_h / h, z0, c0, Ob0, Om0, h, {**_DEFAULTS, "num_points_trapz_int": 48}
    )
    r_god = np.asarray(bcmp.r_array) / h
    r200 = float(q["r200c_comoving"])
    mask = (r_god > 0.05 * r200) & (r_god < 3.0 * r200)
    # Max-error window excludes the outer HSE fringe (quadrature mismatch vs GODMAX).
    mask_core = (r_god > 0.05 * r200) & (r_god < 1.5 * r200)
    pe_h = np.exp(
        np.interp(
            np.log(r_god),
            np.log(np.asarray(q["r_comoving"])),
            np.log(np.asarray(q["Pe"]) + 1e-30),
        )
    )
    pe_g = np.asarray(bcmp.Pe_mat_physical[:, 0, 0, 0]) * 1000.0
    rel = np.abs(pe_h[mask] - pe_g[mask]) / np.maximum(pe_g[mask], 1e-30)
    rel_core = np.abs(pe_h[mask_core] - pe_g[mask_core]) / np.maximum(
        pe_g[mask_core], 1e-30
    )
    assert float(np.median(rel)) < 0.02
    assert float(np.max(rel_core)) < 0.05


def test_dmb_pressure_profile_ur_finite(halo_model):
    Ob0, Om0, h = _cosmo_ob_om_h(halo_model.cosmology)
    m = jnp.array([1e14 / h])
    z = jnp.array([0.3])
    r = jnp.logspace(-2, 0.5, 16)
    prof = DMBPressureProfile(c200c=5.0, num_points_trapz_int=32)
    pe = prof.u_r(halo_model, r, m, z)
    assert pe.shape == (16, 1, 1)
    assert jnp.all(jnp.isfinite(pe))
    assert float(pe.max()) > 0.0


def test_dmb_and_godmax_nfw_same_mtot_uk0(halo_model):
    """GODMAX: both windows share Mtot, so u(k→0) must match."""
    h = float(halo_model.cosmology._cosmo_params()["h"])
    m = jnp.array([1e14 / h])
    z = jnp.array([0.0])
    k = jnp.array([1e-3, 3e-3, 1e-2])
    dmb = DMBMatterProfile(c200c=4.0, num_points_trapz_int=48)
    nfw = DMBNFWMatterProfile(c200c=4.0, num_points_trapz_int=48)
    ud = np.asarray(dmb.u_k(halo_model, k, m, z))[:, 0, 0]
    un = np.asarray(nfw.u_k(halo_model, k, m, z))[:, 0, 0]
    np.testing.assert_allclose(ud, un, rtol=0.05)


def test_dmb_matter_and_gas_profiles(halo_model):
    h = float(halo_model.cosmology._cosmo_params()["h"])
    m = jnp.array([1e14 / h])
    z = jnp.array([0.3])
    r = jnp.logspace(-2, 0.5, 12)
    mat = DMBMatterProfile(c200c=4.0, num_points_trapz_int=32)
    gas = DMBGasDensityProfile(c200c=4.0, num_points_trapz_int=32)
    ur_m = mat.u_r(halo_model, r, m, z)
    ur_g = gas.u_r(halo_model, r, m, z)
    assert jnp.all(jnp.isfinite(ur_m))
    assert jnp.all(jnp.isfinite(ur_g))
    assert float(ur_m.max()) > 0.0


def test_dmb_tsz_cl_smoke(halo_model):
    h = float(halo_model.cosmology._cosmo_params()["h"])
    m = jnp.logspace(13.0, 15.0, 8) / h
    z = jnp.linspace(0.05, 1.0, 6)
    ell = jnp.array([100.0, 500.0, 1000.0])
    tsz = tSZTracer(profile=DMBPressureProfile(c200c=5.0, num_points_trapz_int=32))
    cl = halo_model.cl_1h(tsz, None, ell, m, z)
    assert cl.shape[0] == 3
    assert jnp.all(jnp.isfinite(cl))
    assert float(jnp.max(jnp.abs(cl))) > 0.0


def test_dmb_update_and_pytree():
    p = DMBPressureProfile(theta_ej_0=4.0)
    p2 = p.update(theta_ej_0=6.0)
    assert p2.theta_ej_0 == 6.0
    assert p.theta_ej_0 == 4.0
    leaves, treedef = jax.tree_util.tree_flatten(p2)
    p3 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert p3.theta_ej_0 == 6.0


def test_cl_to_xi_helpers():
    from hmfast.halos.cl_to_xi import (
        cl_ky_to_gty,
        cl_kk_to_xip,
        cl_kk_to_xim,
        theta_to_arcmin,
    )

    ell = jnp.logspace(1.5, 3.5, 64)
    cl = 1e-10 / (ell * (ell + 1.0))
    th, gty = cl_ky_to_gty(ell, cl)
    th2, xip = cl_kk_to_xip(ell, cl)
    _, xim = cl_kk_to_xim(ell, cl)
    assert th.shape == gty.shape
    assert jnp.all(jnp.isfinite(gty))
    assert jnp.all(jnp.isfinite(xip))
    assert jnp.all(jnp.isfinite(xim))
    assert float(theta_to_arcmin(th2[0])) > 0.0


def test_dmb_act_n_nt_zcap_mode(halo_model):
    """ACT eq. 2.12: n_nt_zcap triggers redshift-cap formula with R_500c."""
    h = float(halo_model.cosmology._cosmo_params()["h"])
    m = jnp.array([1e14 / h])
    z = jnp.array([0.0])
    r = jnp.logspace(-2, 0.5, 12)

    # GODMAX mode (n_nt_zcap=None): default
    prof_godmax = DMBPressureProfile(
        c200c=4.0, num_points_trapz_int=32, alpha_nt=0.2, n_nt=0.3
    )
    # ACT mode (n_nt_zcap set): radial index fixed at 0.8, uses R_500c
    prof_act = DMBPressureProfile(
        c200c=4.0, num_points_trapz_int=32, alpha_nt=0.2, n_nt=0.8, n_nt_zcap=0.8
    )

    pe_godmax = np.asarray(prof_godmax.u_r(halo_model, r, m, z))[:, 0, 0]
    pe_act = np.asarray(prof_act.u_r(halo_model, r, m, z))[:, 0, 0]
    assert jnp.all(jnp.isfinite(pe_act))
    # ACT and GODMAX formulas differ → pressures should not be identical
    assert not np.allclose(pe_act, pe_godmax, rtol=1e-3)

    # Update should preserve n_nt_zcap
    prof_act2 = prof_act.update(alpha_nt=0.3)
    assert prof_act2.n_nt_zcap == 0.8
