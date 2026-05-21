"""
Stress test for ``HaloModel.cl_2h_masked``.

Validates the masked 2-halo tSZ angular power spectrum against analytic
limits and the existing unmasked 2-halo path:

1. Shapes and finiteness on several ``(l, M, z)`` grids.
2. Unmasked limit: with ``mask == 1`` the result reproduces ``cl_2h`` of a
   HaloModel whose consistency counterterm is disabled (``cl_2h_masked``
   intentionally drops that counterterm, see its docstring).
3. Quadratic mask scaling: each of the two bias-weighted brackets is
   linear in the mask, so a constant mask ``c`` rescales the spectrum by
   ``c ** 2``.
4. Heaviside selection (0/1 mask) lowers the power and keeps it positive.
5. Log-normal-scatter no-detection limit: with ``q_cat -> infinity`` the
   conditional first moment :math:`\\mathcal{W}_1 \\to \\langle A\\rangle
   = e^{\\sigma_{\\ln Y}^2 / 2}`, so the masked spectrum approaches
   :math:`e^{\\sigma_{\\ln Y}^2}` times the ``mask == 1`` result.
6. The 2-halo term must use the ``n_power=1`` conditional moment; the
   ``n_power=2`` weight (correct for the 1-halo term) is a different,
   physically wrong selection weight here.
7. ``cl_2h_masked`` is JIT-compiled and differentiable; the gradient with
   respect to a scalar mask amplitude matches a finite difference.

Run with::

    pytest tests/stress_test_cl_2h_masked.py -v
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer
from hmfast.tracers.tsz_completeness import conditional_An_undetected


# ---------------------------------------------------------------------------
# Fixtures (session-scoped to avoid repeated emulator loads / JIT recompiles)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def cosmology():
    return Cosmology(emulator_set="lcdm:v1")


@pytest.fixture(scope="session")
def hm(cosmology):
    return HaloModel(cosmology=cosmology)


@pytest.fixture(scope="session")
def hm_no_counterterm(cosmology):
    """HaloModel with the halo-model consistency counterterm disabled.

    ``cl_2h_masked`` drops the counterterm; this is the matching reference
    for the unmasked limit.
    """
    return HaloModel(cosmology=cosmology, hm_consistency=False)


@pytest.fixture(scope="session")
def tsz():
    return tSZTracer(profile=GNFWPressureProfile())


@pytest.fixture(scope="session")
def grids():
    """Reference ``(l, M, z)`` grids used by most tests."""
    l = jnp.logspace(jnp.log10(80.0), jnp.log10(8000.0), 24)
    m = jnp.logspace(11.0, 15.5, 48)
    z = jnp.linspace(0.02, 3.0, 32)
    return l, m, z


def _synthetic_snr(m, z):
    """Monotone synthetic SNR grid :math:`\\bar q(M, z)`, shape ``(Nm, Nz)``.

    Avoids a hard dependency on the Planck SZiFi noise files at test time.
    """
    return 5.0 * (m[:, None] / 1.0e14) ** 1.2 / (1.0 + z[None, :])


# ---------------------------------------------------------------------------
# 1. Shapes and finiteness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nl,nm,nz", [(1, 8, 6), (24, 48, 32), (60, 30, 20)])
def test_shape_and_finiteness(hm, tsz, nl, nm, nz):
    """Output is ``(Nl,)``, finite, and positive for an all-ones mask."""
    l = jnp.logspace(2.0, 3.9, nl)
    m = jnp.logspace(11.0, 15.5, nm)
    z = jnp.linspace(0.02, 3.0, nz)
    mask = jnp.ones((nm, nz))

    cl = hm.cl_2h_masked(tsz, None, l, m, z, mask)

    assert cl.shape == (nl,)
    assert jnp.all(jnp.isfinite(cl))
    assert jnp.all(cl > 0.0)


# ---------------------------------------------------------------------------
# 2. Unmasked limit
# ---------------------------------------------------------------------------
def test_unmasked_limit_matches_cl_2h(hm, hm_no_counterterm, tsz, grids):
    """``mask == 1`` reproduces ``cl_2h`` without the consistency counterterm."""
    l, m, z = grids
    mask = jnp.ones((m.shape[0], z.shape[0]))

    cl_masked = hm.cl_2h_masked(tsz, None, l, m, z, mask)
    cl_ref = hm_no_counterterm.cl_2h(tsz, None, l, m, z)

    assert cl_masked.shape == cl_ref.shape
    assert jnp.all(jnp.isfinite(cl_masked))
    assert jnp.allclose(cl_masked, cl_ref, rtol=1e-5, atol=0.0)


# ---------------------------------------------------------------------------
# 3. Quadratic mask scaling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c", [0.3, 2.5])
def test_constant_mask_scales_quadratically(hm, tsz, grids, c):
    """A constant mask ``c`` rescales the spectrum by exactly ``c ** 2``.

    This encodes the central physics: each of the two bias-weighted
    brackets is linear in the mask, so the spectrum is quadratic in it.
    """
    l, m, z = grids
    ones = jnp.ones((m.shape[0], z.shape[0]))

    cl1 = hm.cl_2h_masked(tsz, None, l, m, z, ones)
    cl_c = hm.cl_2h_masked(tsz, None, l, m, z, c * ones)

    assert jnp.all(jnp.isfinite(cl_c))
    assert jnp.allclose(cl_c, c**2 * cl1, rtol=1e-6, atol=0.0)


# ---------------------------------------------------------------------------
# 4. Heaviside selection lowers the power
# ---------------------------------------------------------------------------
def test_heaviside_mask_reduces_power(hm, tsz, grids):
    """A 0/1 selection keeping only low-mass halos lowers, never raises, power."""
    l, m, z = grids
    ones = jnp.ones((m.shape[0], z.shape[0]))
    # Keep undetected (low-mass) halos only: drop M > 5e14 M_sun.
    keep = (m[:, None] < 5.0e14).astype(jnp.float64) * jnp.ones((1, z.shape[0]))

    cl_full = hm.cl_2h_masked(tsz, None, l, m, z, ones)
    cl_cut = hm.cl_2h_masked(tsz, None, l, m, z, keep)

    assert jnp.all(jnp.isfinite(cl_cut))
    assert jnp.all(cl_cut > 0.0)
    assert jnp.all(cl_cut <= cl_full)
    assert jnp.sum(cl_cut) < jnp.sum(cl_full)


# ---------------------------------------------------------------------------
# 5. Log-normal-scatter no-detection limit
# ---------------------------------------------------------------------------
def test_no_detection_limit_scatter(hm, tsz, grids):
    """``q_cat -> infinity`` gives ``W1 -> <A>`` and ``Cl -> exp(sigma**2) Cl``."""
    l, m, z = grids
    sigma_lnY = 0.2
    ones = jnp.ones((m.shape[0], z.shape[0]))
    snr = _synthetic_snr(m, z)

    # Nothing is detected: the conditional first moment collapses to <A>.
    w1 = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=1.0e8, n_power=1)
    expected_mean_A = jnp.exp(0.5 * sigma_lnY**2)
    assert jnp.allclose(w1, expected_mean_A, rtol=1e-4, atol=0.0)

    cl_ones = hm.cl_2h_masked(tsz, None, l, m, z, ones)
    cl_w1 = hm.cl_2h_masked(tsz, None, l, m, z, w1)

    # Each bracket picks up <A>, so the spectrum picks up <A>**2 = exp(sigma**2).
    # atol=0.0: Cl ~ 1e-19, so the comparison must be purely relative.
    assert jnp.all(jnp.isfinite(cl_w1))
    assert jnp.allclose(cl_w1, jnp.exp(sigma_lnY**2) * cl_ones, rtol=1e-3, atol=0.0)


# ---------------------------------------------------------------------------
# 6. The 2-halo term needs the first moment, not the second
# ---------------------------------------------------------------------------
def test_two_halo_requires_first_moment(hm, tsz, grids):
    """``n_power=1`` and ``n_power=2`` weights give genuinely different spectra.

    The 2-halo masked term must be fed the ``n_power=1`` conditional moment;
    the ``n_power=2`` weight is the (correct) 1-halo weight and would be
    physically wrong here.
    """
    l, m, z = grids
    sigma_lnY = 0.25
    q_cat = 6.0
    snr = _synthetic_snr(m, z)

    w1 = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat, n_power=1)
    w2 = conditional_An_undetected(snr, sigma_lnY=sigma_lnY, q_cat=q_cat, n_power=2)

    cl_w1 = hm.cl_2h_masked(tsz, None, l, m, z, w1)
    cl_w2 = hm.cl_2h_masked(tsz, None, l, m, z, w2)

    assert jnp.all(jnp.isfinite(cl_w1))
    assert jnp.all(cl_w1 > 0.0)
    # The two selection weights, and hence the spectra, are genuinely distinct.
    # atol=0.0: Cl ~ 1e-19, so the default atol=1e-8 would swamp the comparison.
    assert not jnp.allclose(w1, w2, rtol=1e-2, atol=0.0)
    assert not jnp.allclose(cl_w1, cl_w2, rtol=1e-2, atol=0.0)


# ---------------------------------------------------------------------------
# 7. JIT and autodiff
# ---------------------------------------------------------------------------
def test_jit_and_grad(hm, tsz, grids):
    """``cl_2h_masked`` is differentiable; grad matches the analytic ``2 c``."""
    l, m, z = grids
    ones = jnp.ones((m.shape[0], z.shape[0]))

    def total_power(c):
        return jnp.sum(hm.cl_2h_masked(tsz, None, l, m, z, c * ones))

    base = total_power(1.0)
    grad = jax.grad(total_power)(1.0)

    # total_power(c) = c**2 * base  =>  d/dc = 2 c base, which is 2*base at c=1.
    # atol=0.0: the summed power is ~1e-18, far below the default atol.
    assert jnp.isfinite(grad)
    assert jnp.allclose(grad, 2.0 * base, rtol=1e-5, atol=0.0)

    eps = 1e-4
    fd = (total_power(1.0 + eps) - total_power(1.0 - eps)) / (2.0 * eps)
    assert jnp.allclose(grad, fd, rtol=1e-4, atol=0.0)
