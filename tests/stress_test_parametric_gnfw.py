"""
Stress test for ParametricGNFWPressureProfile.

Validates that the parametric profile is exactly the standard GNFW profile
rescaled by ``y0_param / y0_orig``. The ratio ``Pe_param / Pe_GNFW`` must
(1) be independent of r, and (2) equal the closed-form ratio of the parametric
Compton-y0 amplitude to the Arnaud profile's Compton-y0.
"""

import jax
import jax.numpy as jnp
import numpy as np

from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile, ParametricGNFWPressureProfile
from hmfast.halos.mass_definition import MassDefinition, convert_m_delta
from hmfast.utils import Const


def _make_profiles():
    A_SZ = 0.2175
    alpha_SZ = 0.7867
    P0, c500, alpha, beta, gamma, B = 8.130, 1.156, 1.0620, 5.4807, 0.3292, 1.4
    gnfw = GNFWPressureProfile(P0=P0, c500=c500, alpha=alpha, beta=beta, gamma=gamma, B=B)
    para = ParametricGNFWPressureProfile(
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
        P0=P0, c500=c500, alpha=alpha, beta=beta, gamma=gamma, B=B,
    )
    return gnfw, para


def test_shapes_and_finiteness():
    """Output shapes must be (Nr, Nm, Nz) and all entries finite."""
    hm = HaloModel()
    gnfw, para = _make_profiles()

    r = jnp.logspace(-2, 1, 20)
    m = jnp.logspace(13, 15.5, 8)
    z = jnp.array([0.0, 0.3, 0.7, 1.5])

    Pe_g = gnfw.u_r(hm, r, m, z)
    Pe_p = para.u_r(hm, r, m, z)

    expected_shape = (r.shape[0], m.shape[0], z.shape[0])
    assert Pe_g.shape == expected_shape, f"GNFW shape {Pe_g.shape} != {expected_shape}"
    assert Pe_p.shape == expected_shape, f"Parametric shape {Pe_p.shape} != {expected_shape}"
    assert jnp.all(jnp.isfinite(Pe_g)), "GNFW has non-finite entries"
    assert jnp.all(jnp.isfinite(Pe_p)), "Parametric has non-finite entries"


def test_ratio_independent_of_r():
    """Pe_param / Pe_GNFW must not depend on the radius."""
    hm = HaloModel()
    gnfw, para = _make_profiles()

    r = jnp.logspace(-2, 1, 20)
    m = jnp.logspace(13, 15.5, 8)
    z = jnp.array([0.0, 0.3, 0.7, 1.5])

    Pe_g = gnfw.u_r(hm, r, m, z)
    Pe_p = para.u_r(hm, r, m, z)

    ratio = Pe_p / Pe_g
    ratio_ref = ratio[0]
    rel_spread = jnp.max(jnp.abs(ratio - ratio_ref[None]) / jnp.abs(ratio_ref[None]))
    assert rel_spread < 1e-10, f"Pe_param/Pe_GNFW depends on r! max rel spread = {rel_spread}"


def test_amplitude_ratio_matches_closed_form():
    """The amplitude ratio must match y0_param / y0_orig computed in closed form."""
    hm = HaloModel()
    H0 = hm.cosmology.H0
    h = H0 / 100.0

    A_SZ = 0.2175
    alpha_SZ = 0.7867
    P0, c500, alpha, beta, gamma, B = 8.130, 1.156, 1.0620, 5.4807, 0.3292, 1.4

    para = ParametricGNFWPressureProfile(
        A_SZ=A_SZ, alpha_SZ=alpha_SZ,
        P0=P0, c500=c500, alpha=alpha, beta=beta, gamma=gamma, B=B,
    )
    gnfw = GNFWPressureProfile(P0=P0, c500=c500, alpha=alpha, beta=beta, gamma=gamma, B=B)

    r = jnp.logspace(-2, 1, 20)
    m = jnp.logspace(13, 15.5, 8)
    z = jnp.array([0.0, 0.3, 0.7, 1.5])

    Pe_g = gnfw.u_r(hm, r, m, z)
    Pe_p = para.u_r(hm, r, m, z)
    ratio_ref = (Pe_p / Pe_g)[0]

    # Compute y0_param and y0_orig in closed form (same formula as the profile)
    mass_def_500c = MassDefinition(500, "critical")
    c_old = hm.concentration.c_delta(hm, m, z)
    m500c = convert_m_delta(hm.cosmology, m, z, hm.mass_definition, mass_def_500c, c_old=c_old)

    H = jnp.atleast_1d(hm.cosmology.hubble_parameter(z))
    E = H / H0
    m500c_tilde = m500c * h / B
    pivot = 0.7 * 3e14

    P_500c_arnaud = (
        1.65 * (h / 0.7) ** 2 * E[None, :] ** (8.0 / 3.0)
        * (m500c_tilde / pivot) ** (2.0 / 3.0 + 0.12)
        * (0.7 / h) ** 1.5
    )

    r_500c = mass_def_500c.r_delta(hm.cosmology, m500c, z)

    sigma_T_cm2 = 6.6524587e-25
    m_e_c2_eV = 510998.95
    shape_integral = 0.470502095
    mpc_to_cm = Const._Mpc_over_m_ * 100.0

    r_500c_cm = r_500c * mpc_to_cm

    y0_orig = (
        2.0 * (sigma_T_cm2 / m_e_c2_eV)
        * P0 * P_500c_arnaud * r_500c_cm
        * shape_integral
    )

    y0_param = (
        (10.0 ** A_SZ)
        * (m500c_tilde / pivot) ** alpha_SZ
        * E[None, :] ** 2
        * (h / 0.7) ** (-0.5)
    )

    ratio_expected = y0_param / y0_orig

    rel_err = jnp.max(jnp.abs(ratio_ref - ratio_expected) / jnp.abs(ratio_expected))
    assert rel_err < 1e-10, f"Amplitude ratio mismatch: max rel err = {rel_err}"


def test_u_k_end_to_end():
    """The u_k pipeline must produce finite output with shape (Nk, Nm, Nz)."""
    hm = HaloModel()
    gnfw, para = _make_profiles()

    k = jnp.logspace(-3, 2, 30)
    m = jnp.logspace(13, 15.5, 8)
    z = jnp.array([0.1, 0.3, 0.7, 1.5])  # z > 0 to avoid d_A = 0 singularity

    u_k = para.u_k(hm, k, m, z)
    assert u_k.shape == (k.shape[0], m.shape[0], z.shape[0]), f"u_k shape wrong: {u_k.shape}"
    assert jnp.all(jnp.isfinite(u_k)), "u_k has non-finite entries"


def test_gradient_through_params():
    """Gradients through A_SZ and alpha_SZ must be finite."""
    hm = HaloModel()
    gnfw, para = _make_profiles()
    r = jnp.logspace(-2, 1, 20)
    m = jnp.logspace(13, 15.5, 8)
    z = jnp.array([0.0, 0.3, 0.7, 1.5])

    def loss(A, a):
        prof = para.update(A_SZ=A, alpha_SZ=a)
        return jnp.sum(prof.u_r(hm, r, m, z) ** 2)

    g_A, g_a = jax.grad(loss, argnums=(0, 1))(para.A_SZ, para.alpha_SZ)
    assert jnp.isfinite(g_A) and jnp.isfinite(g_a), "non-finite gradient"


def test_update_preserves_identity():
    """Calling update() with no args must return an identical profile."""
    hm = HaloModel()
    gnfw, para = _make_profiles()
    r = jnp.logspace(-2, 1, 20)
    m = jnp.logspace(13, 15.5, 8)
    z = jnp.array([0.0, 0.3, 0.7, 1.5])

    Pe_p = para.u_r(hm, r, m, z)
    para2 = para.update()
    Pe_p2 = para2.u_r(hm, r, m, z)
    assert jnp.allclose(Pe_p, Pe_p2), "update() with no args changed the output"


def test_a_sz_rescaling():
    """Doubling 10^A_SZ should double the profile everywhere."""
    hm = HaloModel()
    gnfw, para = _make_profiles()
    r = jnp.logspace(-2, 1, 20)
    m = jnp.logspace(13, 15.5, 8)
    z = jnp.array([0.0, 0.3, 0.7, 1.5])

    Pe_p = para.u_r(hm, r, m, z)
    para3 = para.update(A_SZ=para.A_SZ + jnp.log10(2.0))
    Pe_p3 = para3.u_r(hm, r, m, z)
    rel = jnp.max(jnp.abs(Pe_p3 / Pe_p - 2.0))
    assert rel < 1e-10, f"A_SZ rescaling broken: max |ratio - 2| = {rel}"
