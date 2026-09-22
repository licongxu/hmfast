"""Tests for the periodic-universe non-Limber 2-halo angular spectrum.

Drives the shipped lattice helpers and ``HaloModel.cl_2h_periodic`` /
``radial_transfer_R_ell`` (not a reimplementation). The 1-halo and usual
Limber 2-halo paths must remain callable and numerically unchanged.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import (
    HaloModel,
    cubic_lattice_shells,
    cubic_lattice_vectors,
    lattice_wavenumbers,
)
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer


@pytest.fixture(scope="module")
def cosmology():
    return Cosmology(emulator_set="lcdm:v1")


@pytest.fixture(scope="module")
def halo_model(cosmology):
    return HaloModel(cosmology=cosmology, hm_consistency=False)


@pytest.fixture(scope="module")
def tsz_tracer():
    return tSZTracer(profile=GNFWPressureProfile())


# Cheap tSZ grid used by every angular test in this module.
_L = 250.0
_NMAX = 2
_ELL = jnp.array([2, 4], dtype=jnp.int64)
_M = jnp.logspace(12.0, 14.5, 8)
_Z = jnp.linspace(0.05, 1.0, 6)


class TestCubicLattice:
    def test_dc_excluded_and_known_multiplicities(self):
        s, g = cubic_lattice_shells(n_max=2)
        assert 0 not in np.asarray(s)
        table = {int(si): int(gi) for si, gi in zip(s, g)}
        # PDF §6.1: first shells and their multiplicities.
        assert table[1] == 6
        assert table[2] == 12
        assert table[3] == 8
        assert table[4] == 6
        assert table[5] == 24

    def test_no_k_below_fundamental(self):
        s, _g = cubic_lattice_shells(n_max=2)
        k_s = lattice_wavenumbers(_L, s)
        k_f = 2.0 * np.pi / _L
        assert np.min(k_s) >= k_f - 1e-15
        assert np.isclose(np.min(k_s), k_f)
        assert np.all(k_s > 0.0)

    def test_vectors_match_shell_count(self):
        s, g = cubic_lattice_shells(n_max=2)
        p = cubic_lattice_vectors(n_max=2)
        assert p.shape[1] == 3
        assert not np.any(np.all(p == 0, axis=1))
        assert p.shape[0] == int(np.sum(g))
        s_from_p = np.sum(p * p, axis=1)
        assert np.min(s_from_p) >= 1
        unique, counts = np.unique(s_from_p, return_counts=True)
        np.testing.assert_array_equal(unique, s)
        np.testing.assert_array_equal(counts, g)

    def test_s_max_excludes_higher_shells(self):
        s, g = cubic_lattice_shells(s_max=3)
        assert set(s.tolist()) == {1, 2, 3}
        assert int(np.sum(g)) == 6 + 12 + 8


class TestPeriodicTwoHalo:
    def test_I1_matches_pk_2h(self, halo_model, tsz_tracer):
        """Shipped I_1 is the same bias-weighted integral used by pk_2h."""
        k = jnp.logspace(-2.0, 0.0, 6)
        z = jnp.array([0.3, 0.8])
        I = halo_model.bias_weighted_I(tsz_tracer, k, _M, z)
        pk2h = halo_model.pk_2h(tsz_tracer, tsz_tracer, k=k, m=_M, z=z, linear=True)
        P_m = halo_model._emulator_pk(k, z, linear=True)
        np.testing.assert_allclose(
            np.asarray(pk2h), np.asarray(P_m * I * I), rtol=1e-10, atol=0.0
        )

    def test_cl_finite_real_shape(self, halo_model, tsz_tracer):
        cl = np.asarray(
            halo_model.cl_2h_periodic(tsz_tracer, None, _ELL, _M, _Z, L=_L, n_max=_NMAX)
        )
        assert cl.shape == (_ELL.size,)
        assert np.all(np.isfinite(cl))
        assert np.isrealobj(cl)
        assert np.all(np.isreal(cl))
        # Auto-spectrum is a sum of squares.
        assert np.all(cl >= 0.0)

    def test_shell_sum_matches_explicit_p_sum(self, halo_model, tsz_tracer):
        s, g = cubic_lattice_shells(n_max=_NMAX)
        cl_shell = np.asarray(
            halo_model.cl_2h_periodic(tsz_tracer, None, _ELL, _M, _Z, L=_L, s=s, g=g)
        )
        p = cubic_lattice_vectors(n_max=_NMAX)
        k_p = lattice_wavenumbers(_L, np.sum(p * p, axis=1))
        R = np.asarray(halo_model.radial_transfer_R_ell(tsz_tracer, _ELL, k_p, _M, _Z))
        cl_p = (4.0 * np.pi / _L**3) * np.sum(R * R, axis=1)
        np.testing.assert_allclose(cl_shell, cl_p, rtol=1e-10, atol=0.0)

    def test_dc_not_in_wavenumbers_used_by_cl(self, halo_model, tsz_tracer):
        s, g = cubic_lattice_shells(n_max=_NMAX)
        k_s = lattice_wavenumbers(_L, s)
        assert 0 not in np.asarray(s)
        assert np.min(k_s) >= 2.0 * np.pi / _L - 1e-15
        cl = halo_model.cl_2h_periodic(tsz_tracer, None, _ELL, _M, _Z, L=_L, s=s, g=g)
        assert np.all(np.isfinite(np.asarray(cl)))

    def test_one_halo_unchanged_on_same_grid(self, halo_model, tsz_tracer):
        cl1_a = np.asarray(halo_model.cl_1h(tsz_tracer, tsz_tracer, l=_ELL, m=_M, z=_Z))
        _ = halo_model.cl_2h_periodic(tsz_tracer, None, _ELL, _M, _Z, L=_L, n_max=_NMAX)
        cl1_b = np.asarray(halo_model.cl_1h(tsz_tracer, tsz_tracer, l=_ELL, m=_M, z=_Z))
        np.testing.assert_allclose(cl1_a, cl1_b, rtol=0.0, atol=0.0)
        assert cl1_a.shape == (_ELL.size,)
        assert np.all(np.isfinite(cl1_a))

    def test_usual_limber_2h_still_callable(self, halo_model, tsz_tracer):
        cl_limber = np.asarray(
            halo_model.cl_2h(tsz_tracer, tsz_tracer, l=_ELL, m=_M, z=_Z)
        )
        cl_per = np.asarray(
            halo_model.cl_2h_periodic(tsz_tracer, None, _ELL, _M, _Z, L=_L, n_max=_NMAX)
        )
        assert cl_limber.shape == cl_per.shape == (_ELL.size,)
        assert np.all(np.isfinite(cl_limber))
        assert np.all(np.isfinite(cl_per))

    def test_nonlimber_default_is_exact_limber_fallback(self, halo_model, tsz_tracer):
        cl_limber = np.asarray(halo_model.cl_2h(tsz_tracer, None, _ELL, _M, _Z))
        cl_hybrid = np.asarray(
            halo_model.cl_2h_nonlimber(tsz_tracer, None, _ELL, _M, _Z)
        )
        np.testing.assert_array_equal(cl_hybrid, cl_limber)

    def test_nonlimber_ratio_matches_upstream_swiftcl(self, halo_model, tsz_tracer):
        ell = jnp.array([2.0, 8.0, 32.0])
        m = jnp.logspace(12.0, 14.5, 12)
        z = jnp.linspace(0.05, 1.0, 48)
        k = jnp.geomspace(1.0e-3, 3.0, 96)
        cl_limber = np.asarray(halo_model.cl_2h(tsz_tracer, None, ell, m, z))
        cl_hybrid = np.asarray(
            halo_model.cl_2h_nonlimber(
                tsz_tracer,
                None,
                ell,
                m,
                z,
                l_limber=16.0,
                k=k,
                n_chi=512,
            )
        )

        # hmfast/hmfast@243f744 on this grid gives these SwiftCl/Limber ratios.
        # Absolute amplitudes differ after its profile-unit refactor, so the
        # projection ratio is the portable parity quantity.
        upstream_ratio = np.array([0.97713996, 1.01324944, 1.0])
        np.testing.assert_allclose(
            cl_hybrid / cl_limber, upstream_ratio, rtol=0.05, atol=0.0
        )

    def test_high_ell_large_L_approaches_limber(self, halo_model, tsz_tracer):
        """At high ℓ and large L the lattice 2h meets Limber cl_2h (same physical P(k))."""
        L = 1000.0
        ell = jnp.array([256], dtype=jnp.int64)
        z = jnp.linspace(0.25, 1.5, 16)
        chi = np.asarray(halo_model.cosmology.angular_diameter_distance(z) * (1.0 + z))
        k_max = 1.5 * float((256.0 + 0.5) / np.min(chi))
        cl_limber = np.asarray(halo_model.cl_2h(tsz_tracer, None, ell, _M, z))
        cl_per = np.asarray(
            halo_model.cl_2h_periodic(
                tsz_tracer, None, ell, _M, z, L=L, k_max=k_max, n_k=48, n_chi=512
            )
        )
        assert cl_limber.shape == cl_per.shape == (1,)
        assert np.all(np.isfinite(cl_limber)) and np.all(np.isfinite(cl_per))
        ratio = float(cl_per[0] / cl_limber[0])
        assert 0.95 < ratio < 1.05, f"high-ℓ large-L ratio {ratio} not near 1"

    def test_one_halo_same_after_high_ell_call(self, halo_model, tsz_tracer):
        ell = jnp.array([64], dtype=jnp.int64)
        z = jnp.linspace(0.4, 1.2, 20)
        cl1_a = np.asarray(halo_model.cl_1h(tsz_tracer, None, ell, _M, z))
        _ = halo_model.cl_2h_periodic(
            tsz_tracer, None, ell, _M, z, L=1500.0, n_max=6, n_chi=128
        )
        cl1_b = np.asarray(halo_model.cl_1h(tsz_tracer, None, ell, _M, z))
        np.testing.assert_allclose(cl1_a, cl1_b, rtol=0.0, atol=0.0)
