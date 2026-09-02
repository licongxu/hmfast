"""Usual vs periodic tSZ C_ell variance (Gaussian + 1-halo trispectrum).

Drives shipped ``HaloModel.var_cl`` / ``var_cl_periodic``,
``connected_1h_cl_variance``, and ``lattice_Q_ell``.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import (
    HaloModel,
    gaussian_cl_variance,
    lattice_Q_ell,
    multipole_bin_weights,
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


_L = 250.0
_NMAX = 2
_ELL = jnp.array([2, 4], dtype=jnp.int64)
_M = jnp.logspace(12.0, 14.5, 8)
_Z = jnp.linspace(0.05, 1.0, 6)


class TestLatticeQ:
    def test_fundamental_shell_analytic(self):
        """PDF §5.7: six equal-weight axes give Q = 1/3 + (2/3) P_ℓ(0)^2."""
        p = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=float,
        )
        ell = np.array([2.0, 3.0])
        A = np.ones((2, 6))
        Q = lattice_Q_ell(ell, p, A)
        # P_2(0) = -1/2 → Q_2 = 1/3 + (2/3)(1/4) = 1/2
        # P_3(0) = 0     → Q_3 = 1/3
        np.testing.assert_allclose(Q, np.array([0.5, 1.0 / 3.0]), rtol=0.0, atol=1e-12)
        assert Q[0] >= 1.0 / 5.0
        assert Q[1] >= 1.0 / 7.0

    def test_A_iso_recovers_isotropic_floor(self):
        p = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=float,
        )
        ell = np.array([2.0, 3.0])
        A = np.ones((2, 6))
        floor = 1.0 / (2.0 * ell + 1.0)
        Q0 = lattice_Q_ell(ell, p, A, A_iso=0.0)
        Qinf = lattice_Q_ell(ell, p, A, A_iso=1.0e12)
        np.testing.assert_allclose(Qinf, floor, rtol=0.0, atol=1e-12)
        assert np.all(Q0 > Qinf + 1e-12)


class TestUsualVariance:
    def test_gaussian_is_2c2_over_2l1(self, halo_model, tsz_tracer):
        out = halo_model.var_cl(tsz_tracer, None, _ELL, _M, _Z)
        cl = out["cl"]
        expected = gaussian_cl_variance(cl, _ELL)
        np.testing.assert_allclose(out["var_gaussian"], expected, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            out["var_gaussian"],
            2.0 * cl * cl / (2.0 * np.asarray(_ELL) + 1.0),
            rtol=1e-12,
            atol=0.0,
        )

    def test_1h_connected_is_existing_trispectrum(self, halo_model, tsz_tracer):
        var_c = np.asarray(
            halo_model.connected_1h_cl_variance(tsz_tracer, None, _ELL, _M, _Z)
        )
        T = np.asarray(halo_model.trispectrum_1h(tsz_tracer, None, _ELL, _ELL, _M, _Z))
        expected = np.diag(T) / (4.0 * np.pi)
        np.testing.assert_allclose(var_c, expected, rtol=0.0, atol=0.0)
        # Second call to the existing trispectrum path is unchanged.
        T2 = np.asarray(halo_model.trispectrum_1h(tsz_tracer, None, _ELL, _ELL, _M, _Z))
        np.testing.assert_allclose(T, T2, rtol=0.0, atol=0.0)

    def test_total_finite_real_shape(self, halo_model, tsz_tracer):
        out = halo_model.var_cl(tsz_tracer, None, _ELL, _M, _Z)
        for key in ("cl", "var_gaussian", "var_1h", "var_total"):
            arr = np.asarray(out[key])
            assert arr.shape == (_ELL.size,)
            assert np.all(np.isfinite(arr))
            assert np.isrealobj(arr)
        np.testing.assert_allclose(
            out["var_total"], out["var_gaussian"] + out["var_1h"], rtol=0.0, atol=0.0
        )
        assert np.all(out["var_gaussian"] >= 0.0)
        assert np.all(out["var_1h"] >= 0.0)


class TestPeriodicVariance:
    def test_Q_and_gaussian_bound(self, halo_model, tsz_tracer):
        out = halo_model.var_cl_periodic(
            tsz_tracer, None, _ELL, _M, _Z, L=_L, n_max=_NMAX
        )
        ell = np.asarray(_ELL, dtype=float)
        floor = 1.0 / (2.0 * ell + 1.0)
        Q = np.asarray(out["Q"])
        assert np.all(np.isfinite(Q))
        assert np.all(Q >= floor - 1e-12)
        bound = 2.0 * out["cl"] ** 2 * floor
        assert np.all(out["var_gaussian"] >= bound - 1e-30)

    def test_1h_same_as_usual(self, halo_model, tsz_tracer):
        usual = halo_model.var_cl(tsz_tracer, None, _ELL, _M, _Z)
        per = halo_model.var_cl_periodic(
            tsz_tracer, None, _ELL, _M, _Z, L=_L, n_max=_NMAX
        )
        np.testing.assert_allclose(per["var_1h"], usual["var_1h"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(per["cl_1h"], usual["cl_1h"], rtol=0.0, atol=0.0)

    def test_cl_2h_matches_shipped_periodic_mean(self, halo_model, tsz_tracer):
        per = halo_model.var_cl_periodic(
            tsz_tracer, None, _ELL, _M, _Z, L=_L, n_max=_NMAX
        )
        cl_2h = np.asarray(
            halo_model.cl_2h_periodic(tsz_tracer, None, _ELL, _M, _Z, L=_L, n_max=_NMAX)
        )
        np.testing.assert_allclose(per["cl_2h"], cl_2h, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(per["A_iso"], 0.0, atol=1e-20, rtol=0.0)

    def test_n_max_aniso_preserves_mean_and_bound(self, halo_model, tsz_tracer):
        full = halo_model.var_cl_periodic(
            tsz_tracer, None, _ELL, _M, _Z, L=_L, n_max=_NMAX
        )
        split = halo_model.var_cl_periodic(
            tsz_tracer, None, _ELL, _M, _Z, L=_L, n_max=_NMAX, n_max_aniso=1
        )
        np.testing.assert_allclose(split["cl_2h"], full["cl_2h"], rtol=1e-10, atol=0.0)
        assert np.all(split["A_iso"] >= -1e-30)
        assert np.any(split["A_iso"] > 0.0)
        floor = 1.0 / (2.0 * np.asarray(_ELL, dtype=float) + 1.0)
        assert np.all(split["Q"] >= floor - 1e-12)

    def test_total_finite_real_shape(self, halo_model, tsz_tracer):
        out = halo_model.var_cl_periodic(
            tsz_tracer, None, _ELL, _M, _Z, L=_L, n_max=_NMAX
        )
        for key in ("cl", "Q", "var_gaussian", "var_1h", "var_total"):
            arr = np.asarray(out[key])
            assert arr.shape == (_ELL.size,)
            assert np.all(np.isfinite(arr))
            assert np.isrealobj(arr)
        np.testing.assert_allclose(
            out["var_total"], out["var_gaussian"] + out["var_1h"], rtol=0.0, atol=0.0
        )


class TestBinnedVariance:
    def test_single_ell_bins_recover_unbinned(self, halo_model, tsz_tracer):
        edges = [2, 4, 4]
        usual_u = halo_model.var_cl(tsz_tracer, None, _ELL, _M, _Z)
        usual_b = halo_model.var_cl_binned(
            tsz_tracer, None, _ELL, _M, _Z, ell_edges=edges
        )
        per_u = halo_model.var_cl_periodic(
            tsz_tracer, None, _ELL, _M, _Z, L=_L, n_max=_NMAX
        )
        per_b = halo_model.var_cl_periodic_binned(
            tsz_tracer, None, _ELL, _M, _Z, L=_L, ell_edges=edges, n_max=_NMAX
        )
        np.testing.assert_allclose(usual_b["cl"], usual_u["cl"], rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            usual_b["var_total"], usual_u["var_total"], rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(per_b["cl"], per_u["cl"], rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(
            per_b["var_gaussian"], per_u["var_gaussian"], rtol=1e-10, atol=0.0
        )
        np.testing.assert_allclose(
            per_b["var_1h"], usual_b["var_1h"], rtol=0.0, atol=0.0
        )

    def test_usual_gaussian_bin_is_weighted_diagonal(self, halo_model, tsz_tracer):
        ell = jnp.array([2, 3, 4], dtype=jnp.int64)
        unb = halo_model.var_cl(tsz_tracer, None, ell, _M, _Z)
        binned = halo_model.var_cl_binned(
            tsz_tracer, None, ell, _M, _Z, ell_edges=[2, 5]
        )
        W, _ell_eff, n_ell = multipole_bin_weights(ell, [2, 5])
        expected_g = float(np.sum(W[0] ** 2 * unb["var_gaussian"]))
        np.testing.assert_allclose(binned["var_gaussian"][0], expected_g, rtol=1e-12)
        assert int(n_ell[0]) == 3
        assert binned["var_total"].shape == (1,)
        assert np.all(np.isfinite(binned["var_total"]))
        f = np.asarray(ell, dtype=float)
        f = f * (f + 1.0) / (2.0 * np.pi)
        expected_d = float(np.sum(W[0] * f * unb["cl"]))
        np.testing.assert_allclose(binned["dell"][0], expected_d, rtol=1e-12)
        expected_vd = float(np.sum((W[0] * f) ** 2 * unb["var_gaussian"]))
        np.testing.assert_allclose(
            binned["var_dell_gaussian"][0], expected_vd, rtol=1e-12
        )

    def test_binned_shapes_finite(self, halo_model, tsz_tracer):
        edges = [2, 3, 5]
        out = halo_model.var_cl_periodic_binned(
            tsz_tracer,
            None,
            jnp.array([2, 3, 4]),
            _M,
            _Z,
            L=_L,
            ell_edges=edges,
            n_max=_NMAX,
        )
        assert out["cl"].shape == (2,)
        assert out["cov_total"].shape == (2, 2)
        assert np.all(np.isfinite(out["var_total"]))
        assert np.all(np.isfinite(out["cov_total"]))
        np.testing.assert_allclose(
            out["var_total"], np.diag(out["cov_total"]), rtol=1e-12, atol=0.0
        )
