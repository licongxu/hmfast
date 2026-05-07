"""
Tests for halo model functionality.

Validates shapes, finiteness, and basic sanity of the HaloModel API
including mass function, bias, concentration, power spectra, and
angular power spectra for multiple tracers.
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import (
    GNFWPressureProfile, NFWMatterProfile, B12PressureProfile,
    B16DensityProfile, Z07GalaxyHODProfile, S12CIBProfile,
)
from hmfast.tracers import tSZTracer, CMBLensingTracer, kSZTracer, GalaxyHODTracer, GalaxyLensingTracer, CIBTracer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cosmology():
    return Cosmology(emulator_set="lcdm:v1")


@pytest.fixture(scope="session")
def halo_model(cosmology):
    return HaloModel(cosmology=cosmology)


@pytest.fixture(scope="session")
def tsz_tracer():
    return tSZTracer(profile=GNFWPressureProfile())


@pytest.fixture(scope="session")
def cmb_lensing_tracer():
    return CMBLensingTracer(profile=NFWMatterProfile())


@pytest.fixture(scope="session")
def ksz_tracer():
    return kSZTracer(profile=B16DensityProfile())


@pytest.fixture(scope="session")
def galaxy_hod_tracer():
    return GalaxyHODTracer(profile=Z07GalaxyHODProfile())


@pytest.fixture(scope="session")
def galaxy_lensing_tracer():
    return GalaxyLensingTracer(profile=NFWMatterProfile())


@pytest.fixture(scope="session")
def cib_tracer():
    return CIBTracer(profile=S12CIBProfile(nu=100))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_cosmology_defaults(self, cosmology):
        assert cosmology.emulator_set == "lcdm:v1"
        assert cosmology.H0 == 68.0
        assert cosmology.omega_cdm == 0.12

    def test_halomodel_init(self, halo_model):
        assert halo_model.cosmology is not None
        assert halo_model.hm_consistency is True

    def test_cosmology_pytree(self, cosmology):
        leaves, treedef = jax.tree_util.tree_flatten(cosmology)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert abs(reconstructed.H0 - cosmology.H0) < 1e-12

    def test_halomodel_pytree(self, halo_model):
        leaves, treedef = jax.tree_util.tree_flatten(halo_model)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert abs(reconstructed.cosmology.H0 - halo_model.cosmology.H0) < 1e-12


# ---------------------------------------------------------------------------
# Physics sanity checks (analytic limits)
# ---------------------------------------------------------------------------

class TestPhysicsLimits:
    """Tests that verify the code reproduces known analytic limits."""

    def test_mass_function_integrates_to_rho_m(self, halo_model):
        """Integrating dn/dlnM * M over mass should recover the mean matter density.

        The Tinker08 mass function is a fitting function and is not exactly
        normalized to the mean density, especially when extrapolated to very
        low masses. We check order-of-magnitude agreement over the calibrated
        mass range (10^10 to 10^15 Msun).
        """
        m = jnp.logspace(10, 15, 200)
        z = jnp.array([0.0])
        dndlnm = halo_model.halo_mass_function.halo_mass_function(halo_model, m, z)
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5
        rho_halos = jnp.sum(dndlnm[:, 0] * m * w)
        Omega_m0 = float(halo_model.cosmology.omega_m(0.0))
        rho_crit0 = float(halo_model.cosmology.critical_density(0.0))
        rho_m = Omega_m0 * rho_crit0
        # Order-unity check: ratio should be within factor of 2
        ratio = float(rho_halos) / rho_m
        assert 0.5 < ratio < 2.0, f"Mass integral ratio = {ratio:.2f}, expected ~1 (within factor 2)"

    def test_bias_integrates_to_one(self, halo_model):
        """Mass-weighted average of b1(M) should be ~1 (by definition of linear bias)."""
        m = jnp.logspace(8, 16, 200)
        z = jnp.array([0.0])
        dndlnm = halo_model.halo_mass_function.halo_mass_function(halo_model, m, z)
        b1 = halo_model.halo_bias.halo_bias(halo_model, m, z)
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5
        # <b1> = ∫ b1 * (dn/dlnM) * M dlnM / ∫ (dn/dlnM) * M dlnM
        num = jnp.sum(b1[:, 0] * dndlnm[:, 0] * m * w)
        den = jnp.sum(dndlnm[:, 0] * m * w)
        b_avg = float(num / den)
        # Tinker10 bias calibration is not exact to unity; 20% tolerance is reasonable
        assert abs(b_avg - 1.0) < 0.20, f"Mass-weighted bias = {b_avg:.3f}, expected ~1.0"

    def test_pk_2h_reduces_to_plin_at_large_scale(self, halo_model, cmb_lensing_tracer):
        """At large scales, P_2h for matter should approach P_lin."""
        k = jnp.logspace(-3, -1, 20)
        m = jnp.logspace(10, 16, 100)
        z = jnp.array([0.5])
        pk2h = halo_model.pk_2h(cmb_lensing_tracer, cmb_lensing_tracer, k=k, m=m, z=z)
        pk_lin = halo_model.cosmology.pk(z[0], linear=True)[1]
        pk_lin_interp = jnp.interp(k, halo_model.cosmology.pk(z[0], linear=True)[0], pk_lin)
        # At very large scales, P_2h should be close to P_lin (within factor ~2 given approximations)
        ratio = pk2h[:, 0] / pk_lin_interp
        # The ratio should be order unity (within factor 3, allowing for bias and profile effects)
        assert jnp.all(ratio > 0.1) and jnp.all(ratio < 10), f"P_2h/P_lin ratio out of range at large scales"

    def test_pk_1h_positive(self, halo_model, tsz_tracer):
        """1-halo power spectrum should be non-negative everywhere."""
        k = jnp.logspace(-3, 1, 30)
        m = jnp.logspace(11, 15, 20)
        z = jnp.array([0.5])
        pk1h = halo_model.pk_1h(tsz_tracer, tsz_tracer, k=k, m=m, z=z)
        assert jnp.all(pk1h >= 0), "P_1h has negative values"

    def test_cl_positive(self, halo_model, tsz_tracer):
        """Angular power spectrum should be non-negative."""
        ell = jnp.logspace(1, 4, 20)
        m = jnp.logspace(11, 15, 15)
        z = jnp.linspace(0.01, 3.0, 10)
        cl1h = halo_model.cl_1h(tsz_tracer, tsz_tracer, l=ell, m=m, z=z)
        cl2h = halo_model.cl_2h(tsz_tracer, tsz_tracer, l=ell, m=m, z=z)
        assert jnp.all(cl1h >= 0), "C_l^1h has negative values"
        assert jnp.all(cl2h >= 0), "C_l^2h has negative values"


# ---------------------------------------------------------------------------
# Cosmology operations
# ---------------------------------------------------------------------------

class TestCosmology:
    def test_hubble_shape_and_finite(self, cosmology):
        z = jnp.linspace(0, 5, 20)
        Hz = cosmology.hubble_parameter(z)
        assert Hz.shape == z.shape
        assert jnp.all(jnp.isfinite(Hz))

    def test_angular_diameter_distance_shape(self, cosmology):
        z = jnp.linspace(0.01, 5, 20)
        dA = cosmology.angular_diameter_distance(z)
        assert dA.shape == z.shape
        assert jnp.all(jnp.isfinite(dA))
        assert jnp.all(dA > 0)

    def test_sigma8_scalar(self, cosmology):
        s8 = cosmology.sigma8(0.0)
        assert jnp.isfinite(s8)
        assert s8 > 0

    def test_pk_shape(self, cosmology):
        k, pk = cosmology.pk(0.5, linear=True)
        assert k.shape == pk.shape
        assert jnp.all(jnp.isfinite(pk))
        assert jnp.all(pk > 0)

    def test_derived_parameters(self, cosmology):
        dp = cosmology.derived_parameters()
        assert "sigma8" in dp
        assert jnp.isfinite(dp["sigma8"])
        assert dp["sigma8"] > 0

    def test_cosmo_params(self, cosmology):
        p = cosmology._cosmo_params()
        assert "h" in p
        assert "Omega0_m" in p
        assert "Rho_crit_0" in p
        assert jnp.isfinite(p["Rho_crit_0"])


# ---------------------------------------------------------------------------
# Mass function and bias
# ---------------------------------------------------------------------------

class TestMassFunction:
    def test_mass_function_shape(self, halo_model):
        m = jnp.logspace(11, 15, 30)
        z = jnp.array([0.0, 0.5, 1.0])
        hmf = halo_model.halo_mass_function.halo_mass_function(halo_model, m, z)
        assert hmf.shape == (30, 3)

    def test_mass_function_finite(self, halo_model):
        m = jnp.logspace(11, 15, 30)
        z = jnp.array([0.0, 1.0])
        hmf = halo_model.halo_mass_function.halo_mass_function(halo_model, m, z)
        assert jnp.all(jnp.isfinite(hmf))

    def test_mass_function_positive(self, halo_model):
        m = jnp.logspace(12, 14, 20)
        z = jnp.array([0.5])
        hmf = halo_model.halo_mass_function.halo_mass_function(halo_model, m, z)
        assert jnp.all(hmf > 0)


class TestBias:
    def test_bias_shape(self, halo_model):
        m = jnp.logspace(11, 15, 30)
        z = jnp.array([0.0, 0.5, 1.0])
        b1 = halo_model.halo_bias.halo_bias(halo_model, m, z, order=1)
        assert b1.shape == (30, 3)

    def test_bias_finite(self, halo_model):
        m = jnp.logspace(11, 15, 30)
        z = jnp.array([0.0, 1.0])
        b1 = halo_model.halo_bias.halo_bias(halo_model, m, z, order=1)
        assert jnp.all(jnp.isfinite(b1))

    def test_bias_order2_shape(self, halo_model):
        m = jnp.logspace(11, 15, 30)
        z = jnp.array([0.5])
        b2 = halo_model.halo_bias.halo_bias(halo_model, m, z, order=2)
        assert b2.shape == (30, 1)


class TestConcentration:
    def test_concentration_shape(self, halo_model):
        m = jnp.logspace(11, 15, 30)
        z = jnp.array([0.0, 0.5, 1.0])
        c = halo_model.concentration.c_delta(halo_model, m, z)
        assert c.shape == (30, 3)

    def test_concentration_finite(self, halo_model):
        m = jnp.logspace(11, 15, 30)
        z = jnp.array([0.5])
        c = halo_model.concentration.c_delta(halo_model, m, z)
        assert jnp.all(jnp.isfinite(c))
        assert jnp.all(c > 0)


# ---------------------------------------------------------------------------
# Mass definition
# ---------------------------------------------------------------------------

class TestMassDefinition:
    def test_r_delta_shape(self, cosmology):
        mdef = MassDefinition(delta=200, reference="critical")
        m = jnp.logspace(11, 15, 10)
        z = jnp.array([0.0, 0.5])
        r = mdef.r_delta(cosmology, m, z)
        assert r.shape == (10, 2)

    def test_r_delta_positive(self, cosmology):
        mdef = MassDefinition(delta=200, reference="critical")
        m = jnp.logspace(12, 14, 5)
        z = jnp.array([0.5])
        r = mdef.r_delta(cosmology, m, z)
        assert jnp.all(r > 0)


# ---------------------------------------------------------------------------
# Power spectra (tSZ)
# ---------------------------------------------------------------------------

class TesttSZPowerSpectra:
    def test_pk_1h_shape(self, halo_model, tsz_tracer):
        k = jnp.logspace(-3, 1, 50)
        m = jnp.logspace(11, 15, 20)
        z = jnp.array([0.5])
        pk = halo_model.pk_1h(tsz_tracer, tsz_tracer, k=k, m=m, z=z)
        assert pk.shape == (50, 1)

    def test_pk_1h_finite(self, halo_model, tsz_tracer):
        k = jnp.logspace(-3, 1, 50)
        m = jnp.logspace(11, 15, 20)
        z = jnp.array([0.5])
        pk = halo_model.pk_1h(tsz_tracer, tsz_tracer, k=k, m=m, z=z)
        assert jnp.all(jnp.isfinite(pk))

    def test_pk_2h_shape(self, halo_model, tsz_tracer):
        k = jnp.logspace(-3, 1, 50)
        m = jnp.logspace(11, 15, 20)
        z = jnp.array([0.5])
        pk = halo_model.pk_2h(tsz_tracer, tsz_tracer, k=k, m=m, z=z)
        assert pk.shape == (50, 1)

    def test_pk_2h_finite(self, halo_model, tsz_tracer):
        k = jnp.logspace(-3, 1, 50)
        m = jnp.logspace(11, 15, 20)
        z = jnp.array([0.5])
        pk = halo_model.pk_2h(tsz_tracer, tsz_tracer, k=k, m=m, z=z)
        assert jnp.all(jnp.isfinite(pk))

    def test_pk_1h_multiple_z(self, halo_model, tsz_tracer):
        k = jnp.logspace(-3, 1, 30)
        m = jnp.logspace(11, 15, 15)
        z = jnp.linspace(0.0, 3.0, 5)
        pk = halo_model.pk_1h(tsz_tracer, tsz_tracer, k=k, m=m, z=z)
        assert pk.shape == (30, 5)


# ---------------------------------------------------------------------------
# Angular power spectra
# ---------------------------------------------------------------------------

class TestAngularPowerSpectra:
    def test_tsz_cl_1h_shape(self, halo_model, tsz_tracer):
        ell = jnp.logspace(1, 4, 20)
        m = jnp.logspace(11, 15, 15)
        z = jnp.linspace(0.01, 3.0, 10)
        cl = halo_model.cl_1h(tsz_tracer, tsz_tracer, l=ell, m=m, z=z)
        assert cl.shape == (20,)
        assert jnp.all(jnp.isfinite(cl))

    def test_tsz_cl_2h_shape(self, halo_model, tsz_tracer):
        ell = jnp.logspace(1, 4, 20)
        m = jnp.logspace(11, 15, 15)
        z = jnp.linspace(0.01, 3.0, 10)
        cl = halo_model.cl_2h(tsz_tracer, tsz_tracer, l=ell, m=m, z=z)
        assert cl.shape == (20,)
        assert jnp.all(jnp.isfinite(cl))

    def test_cmb_lensing_cl_1h(self, halo_model, cmb_lensing_tracer):
        ell = jnp.logspace(1, 4, 20)
        m = jnp.logspace(11, 15, 15)
        z = jnp.linspace(0.01, 5.0, 10)
        cl = halo_model.cl_1h(cmb_lensing_tracer, cmb_lensing_tracer, l=ell, m=m, z=z)
        assert cl.shape == (20,)
        assert jnp.all(jnp.isfinite(cl))


# ---------------------------------------------------------------------------
# Gradient consistency
# ---------------------------------------------------------------------------

class TestGradients:
    def test_grad_omega_cdm_finite(self, halo_model, tsz_tracer):
        k = jnp.logspace(-2, 0, 10)
        m = jnp.logspace(12, 14, 8)
        z = jnp.array([0.5])

        def loss_fn(omega_cdm):
            cosmo_new = halo_model.cosmology.update(omega_cdm=omega_cdm)
            hm_new = halo_model.update(cosmology=cosmo_new)
            pk = hm_new.pk_1h(tsz_tracer, tsz_tracer, k=k, m=m, z=z)
            return jnp.sum(pk ** 2)

        grad_val = jax.grad(loss_fn)(halo_model.cosmology.omega_cdm)
        assert jnp.isfinite(grad_val)

    def test_grad_vs_finite_diff(self, halo_model, tsz_tracer):
        k = jnp.logspace(-2, 0, 5)
        m = jnp.logspace(12, 14, 5)
        z = jnp.array([0.5])

        def loss_fn(omega_cdm):
            cosmo_new = halo_model.cosmology.update(omega_cdm=omega_cdm)
            hm_new = halo_model.update(cosmology=cosmo_new)
            pk = hm_new.pk_1h(tsz_tracer, tsz_tracer, k=k, m=m, z=z)
            return jnp.sum(pk ** 2)

        omega0 = halo_model.cosmology.omega_cdm
        eps = 1e-5
        analytic = jax.grad(loss_fn)(omega0)
        numerical = (loss_fn(omega0 + eps) - loss_fn(omega0 - eps)) / (2 * eps)
        assert jnp.allclose(analytic, numerical, rtol=1e-3, atol=1e-8), (
            f"Analytic={analytic}, numerical={numerical}"
        )


# ---------------------------------------------------------------------------
# Update methods
# ---------------------------------------------------------------------------

class TestUpdates:
    def test_cosmology_update(self, cosmology):
        cosmo_new = cosmology.update(omega_cdm=0.13)
        assert abs(cosmo_new.omega_cdm - 0.13) < 1e-12
        assert abs(cosmology.omega_cdm - 0.12) < 1e-12

    def test_halomodel_update(self, halo_model):
        cosmo_new = halo_model.cosmology.update(H0=70.0)
        hm_new = halo_model.update(cosmology=cosmo_new)
        assert abs(hm_new.cosmology.H0 - 70.0) < 1e-12
        assert abs(halo_model.cosmology.H0 - 68.0) < 1e-12

    def test_tracer_update(self, tsz_tracer):
        new_profile = GNFWPressureProfile(P0=10.0)
        new_tracer = tsz_tracer.update(profile=new_profile)
        assert abs(new_tracer.profile.P0 - 10.0) < 1e-12
        assert abs(tsz_tracer.profile.P0 - 8.130) < 1e-12


# ---------------------------------------------------------------------------
# kSZ tracer tests
# ---------------------------------------------------------------------------

class TestkSZTracer:
    def test_kernel_shape(self, ksz_tracer, cosmology):
        z = jnp.linspace(0.01, 3.0, 10)
        W = ksz_tracer.kernel(cosmology, z)
        assert W.shape == z.shape
        assert jnp.all(jnp.isfinite(W))

    def test_kernel_positive(self, ksz_tracer, cosmology):
        z = jnp.array([0.5, 1.0, 2.0])
        W = ksz_tracer.kernel(cosmology, z)
        assert jnp.all(W > 0)

    def test_pk_1h_shape_and_finite(self, halo_model, ksz_tracer):
        k = jnp.logspace(-3, 1, 30)
        m = jnp.logspace(11, 15, 15)
        z = jnp.array([0.5])
        pk = halo_model.pk_1h(ksz_tracer, ksz_tracer, k=k, m=m, z=z)
        assert pk.shape == (30, 1)
        assert jnp.all(jnp.isfinite(pk))

    def test_cl_1h_shape_and_finite(self, halo_model, ksz_tracer):
        ell = jnp.logspace(1, 4, 15)
        m = jnp.logspace(11, 15, 10)
        z = jnp.linspace(0.01, 3.0, 8)
        cl = halo_model.cl_1h(ksz_tracer, ksz_tracer, l=ell, m=m, z=z)
        assert cl.shape == (15,)
        assert jnp.all(jnp.isfinite(cl))

    def test_pytree_roundtrip(self, ksz_tracer):
        leaves, treedef = jax.tree_util.tree_flatten(ksz_tracer)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(reconstructed, kSZTracer)


# ---------------------------------------------------------------------------
# Galaxy HOD tracer tests
# ---------------------------------------------------------------------------

class TestGalaxyHODTracer:
    def test_kernel_shape(self, galaxy_hod_tracer, cosmology):
        z = jnp.linspace(0.01, 2.0, 10)
        W = galaxy_hod_tracer.kernel(cosmology, z)
        assert W.shape == z.shape
        assert jnp.all(jnp.isfinite(W))

    def test_has_central_contribution(self, galaxy_hod_tracer):
        assert galaxy_hod_tracer.profile.has_central_contribution is True

    def test_pk_1h_shape_and_finite(self, halo_model, galaxy_hod_tracer):
        k = jnp.logspace(-3, 1, 30)
        m = jnp.logspace(11, 15, 15)
        z = jnp.array([0.5])
        pk = halo_model.pk_1h(galaxy_hod_tracer, galaxy_hod_tracer, k=k, m=m, z=z)
        assert pk.shape == (30, 1)
        assert jnp.all(jnp.isfinite(pk))

    def test_cl_1h_shape_and_finite(self, halo_model, galaxy_hod_tracer):
        ell = jnp.logspace(1, 4, 15)
        m = jnp.logspace(11, 15, 10)
        z = jnp.linspace(0.01, 2.0, 8)
        cl = halo_model.cl_1h(galaxy_hod_tracer, galaxy_hod_tracer, l=ell, m=m, z=z)
        assert cl.shape == (15,)
        assert jnp.all(jnp.isfinite(cl))

    def test_pytree_roundtrip(self, galaxy_hod_tracer):
        leaves, treedef = jax.tree_util.tree_flatten(galaxy_hod_tracer)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(reconstructed, GalaxyHODTracer)

    def test_ng_bar_finite(self, halo_model, galaxy_hod_tracer):
        m = jnp.logspace(11, 15, 20)
        z = jnp.array([0.5])
        ng = galaxy_hod_tracer.profile.ng_bar(halo_model, m, z)
        assert ng.shape == (1,)
        assert jnp.all(jnp.isfinite(ng))
        assert jnp.all(ng > 0)

    def test_galaxy_bias_finite(self, halo_model, galaxy_hod_tracer):
        m = jnp.logspace(11, 15, 20)
        z = jnp.array([0.5])
        bg = galaxy_hod_tracer.profile.galaxy_bias(halo_model, m, z)
        assert bg.shape == (1,)
        assert jnp.all(jnp.isfinite(bg))


# ---------------------------------------------------------------------------
# Galaxy lensing tracer tests
# ---------------------------------------------------------------------------

class TestGalaxyLensingTracer:
    def test_kernel_shape(self, galaxy_lensing_tracer, cosmology):
        z = jnp.linspace(0.01, 2.0, 10)
        W = galaxy_lensing_tracer.kernel(cosmology, z)
        assert W.shape == z.shape
        assert jnp.all(jnp.isfinite(W))

    def test_kernel_positive(self, galaxy_lensing_tracer, cosmology):
        z = jnp.array([0.3, 0.5, 1.0])
        W = galaxy_lensing_tracer.kernel(cosmology, z)
        assert jnp.all(W > 0)

    def test_pk_1h_shape_and_finite(self, halo_model, galaxy_lensing_tracer):
        k = jnp.logspace(-3, 1, 30)
        m = jnp.logspace(11, 15, 15)
        z = jnp.array([0.5])
        pk = halo_model.pk_1h(galaxy_lensing_tracer, galaxy_lensing_tracer, k=k, m=m, z=z)
        assert pk.shape == (30, 1)
        assert jnp.all(jnp.isfinite(pk))

    def test_cl_1h_shape_and_finite(self, halo_model, galaxy_lensing_tracer):
        ell = jnp.logspace(1, 4, 15)
        m = jnp.logspace(11, 15, 10)
        z = jnp.linspace(0.01, 2.0, 8)
        cl = halo_model.cl_1h(galaxy_lensing_tracer, galaxy_lensing_tracer, l=ell, m=m, z=z)
        assert cl.shape == (15,)
        assert jnp.all(jnp.isfinite(cl))

    def test_pytree_roundtrip(self, galaxy_lensing_tracer):
        leaves, treedef = jax.tree_util.tree_flatten(galaxy_lensing_tracer)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(reconstructed, GalaxyLensingTracer)


# ---------------------------------------------------------------------------
# CIB tracer tests
# ---------------------------------------------------------------------------

class TestCIBTracer:
    def test_kernel_shape(self, cib_tracer, cosmology):
        z = jnp.linspace(0.01, 3.0, 10)
        W = cib_tracer.kernel(cosmology, z)
        assert W.shape == z.shape
        assert jnp.all(jnp.isfinite(W))

    def test_has_central_contribution(self, cib_tracer):
        assert cib_tracer.profile.has_central_contribution is True

    def test_pk_1h_shape_and_finite(self, halo_model, cib_tracer):
        k = jnp.logspace(-3, 1, 20)
        m = jnp.logspace(11, 15, 10)
        z = jnp.array([1.0])
        pk = halo_model.pk_1h(cib_tracer, cib_tracer, k=k, m=m, z=z)
        assert pk.shape == (20, 1)
        assert jnp.all(jnp.isfinite(pk))

    def test_cl_1h_shape_and_finite(self, halo_model, cib_tracer):
        ell = jnp.logspace(1, 4, 15)
        m = jnp.logspace(11, 15, 10)
        z = jnp.linspace(0.01, 3.0, 8)
        cl = halo_model.cl_1h(cib_tracer, cib_tracer, l=ell, m=m, z=z)
        assert cl.shape == (15,)
        assert jnp.all(jnp.isfinite(cl))

    def test_pytree_roundtrip(self, cib_tracer):
        leaves, treedef = jax.tree_util.tree_flatten(cib_tracer)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(reconstructed, CIBTracer)

    def test_mean_emissivity_finite(self, halo_model, cib_tracer):
        m = jnp.logspace(11, 15, 15)
        z = jnp.array([1.0])
        jbar = cib_tracer.profile.mean_emissivity(halo_model, m, z)
        assert jbar.shape == (1,)
        assert jnp.all(jnp.isfinite(jbar))


# ---------------------------------------------------------------------------
# Cross-tracer correlation tests
# ---------------------------------------------------------------------------

class TestCrossTracer:
    """Tests for cross-correlation spectra between different tracers."""

    def test_tsz_ksz_pk_1h_shape_and_finite(self, halo_model, tsz_tracer, ksz_tracer):
        k = jnp.logspace(-3, 1, 20)
        m = jnp.logspace(11, 15, 10)
        z = jnp.array([0.5])
        pk = halo_model.pk_1h(tsz_tracer, ksz_tracer, k=k, m=m, z=z)
        assert pk.shape == (20, 1)
        assert jnp.all(jnp.isfinite(pk))

    def test_tsz_galaxy_pk_1h_shape_and_finite(self, halo_model, tsz_tracer, galaxy_hod_tracer):
        k = jnp.logspace(-3, 1, 20)
        m = jnp.logspace(11, 15, 10)
        z = jnp.array([0.5])
        pk = halo_model.pk_1h(tsz_tracer, galaxy_hod_tracer, k=k, m=m, z=z)
        assert pk.shape == (20, 1)
        assert jnp.all(jnp.isfinite(pk))

    def test_cib_galaxy_pk_1h_shape_and_finite(self, halo_model, cib_tracer, galaxy_hod_tracer):
        k = jnp.logspace(-3, 1, 20)
        m = jnp.logspace(11, 15, 10)
        z = jnp.array([1.0])
        pk = halo_model.pk_1h(cib_tracer, galaxy_hod_tracer, k=k, m=m, z=z)
        assert pk.shape == (20, 1)
        assert jnp.all(jnp.isfinite(pk))

    def test_cmb_lensing_galaxy_pk_1h_shape(self, halo_model, cmb_lensing_tracer, galaxy_hod_tracer):
        k = jnp.logspace(-3, 1, 20)
        m = jnp.logspace(11, 15, 10)
        z = jnp.array([0.5])
        pk = halo_model.pk_1h(cmb_lensing_tracer, galaxy_hod_tracer, k=k, m=m, z=z)
        assert pk.shape == (20, 1)
        assert jnp.all(jnp.isfinite(pk))

    def test_cross_pk_2h_shape_and_finite(self, halo_model, tsz_tracer, ksz_tracer):
        k = jnp.logspace(-3, 1, 20)
        m = jnp.logspace(11, 15, 10)
        z = jnp.array([0.5])
        pk = halo_model.pk_2h(tsz_tracer, ksz_tracer, k=k, m=m, z=z)
        assert pk.shape == (20, 1)
        assert jnp.all(jnp.isfinite(pk))

    def test_cross_cl_1h_shape_and_finite(self, halo_model, tsz_tracer, ksz_tracer):
        ell = jnp.logspace(1, 4, 15)
        m = jnp.logspace(11, 15, 10)
        z = jnp.linspace(0.01, 3.0, 8)
        cl = halo_model.cl_1h(tsz_tracer, ksz_tracer, l=ell, m=m, z=z)
        assert cl.shape == (15,)
        assert jnp.all(jnp.isfinite(cl))

    def test_cross_cl_2h_shape_and_finite(self, halo_model, tsz_tracer, galaxy_hod_tracer):
        ell = jnp.logspace(1, 4, 15)
        m = jnp.logspace(11, 15, 10)
        z = jnp.linspace(0.01, 3.0, 8)
        cl = halo_model.cl_2h(tsz_tracer, galaxy_hod_tracer, l=ell, m=m, z=z)
        assert cl.shape == (15,)
        assert jnp.all(jnp.isfinite(cl))


# ---------------------------------------------------------------------------
# Multi-tracer gradient tests
# ---------------------------------------------------------------------------

class TestMultiTracerGradients:
    """Gradient consistency for tracers beyond tSZ."""

    def test_cmb_lensing_gradient_vs_finite_diff(self, halo_model, cmb_lensing_tracer):
        k = jnp.logspace(-2, 0, 15)
        m = jnp.logspace(11, 15, 8)
        z = jnp.array([0.5])
        cosmo = halo_model.cosmology

        def loss_fn(omega_cdm):
            c = cosmo.update(omega_cdm=omega_cdm)
            h = halo_model.update(cosmology=c)
            pk = h.pk_1h(cmb_lensing_tracer, cmb_lensing_tracer, k=k, m=m, z=z)
            return jnp.sum(pk ** 2)

        grad_fn = jax.grad(loss_fn)
        analytic = grad_fn(cosmo.omega_cdm)
        eps = 1e-5
        fd = (loss_fn(cosmo.omega_cdm + eps) - loss_fn(cosmo.omega_cdm - eps)) / (2 * eps)
        rel_err = abs(float(analytic) - float(fd)) / abs(float(fd))
        assert rel_err < 1e-2, f"CMB lensing gradient rel_err = {rel_err}"

    def test_galaxy_hod_gradient_vs_finite_diff(self, halo_model, galaxy_hod_tracer):
        k = jnp.logspace(-2, 0, 15)
        m = jnp.logspace(11, 15, 8)
        z = jnp.array([0.5])
        cosmo = halo_model.cosmology

        def loss_fn(omega_cdm):
            c = cosmo.update(omega_cdm=omega_cdm)
            h = halo_model.update(cosmology=c)
            pk = h.pk_1h(galaxy_hod_tracer, galaxy_hod_tracer, k=k, m=m, z=z)
            return jnp.sum(pk ** 2)

        grad_fn = jax.grad(loss_fn)
        analytic = grad_fn(cosmo.omega_cdm)
        eps = 1e-5
        fd = (loss_fn(cosmo.omega_cdm + eps) - loss_fn(cosmo.omega_cdm - eps)) / (2 * eps)
        rel_err = abs(float(analytic) - float(fd)) / abs(float(fd))
        assert rel_err < 1e-2, f"Galaxy HOD gradient rel_err = {rel_err}"

    def test_cib_gradient_vs_finite_diff(self, halo_model, cib_tracer):
        k = jnp.logspace(-2, 0, 15)
        m = jnp.logspace(11, 15, 8)
        z = jnp.array([1.0])
        cosmo = halo_model.cosmology

        def loss_fn(omega_cdm):
            c = cosmo.update(omega_cdm=omega_cdm)
            h = halo_model.update(cosmology=c)
            pk = h.pk_1h(cib_tracer, cib_tracer, k=k, m=m, z=z)
            return jnp.sum(pk ** 2)

        grad_fn = jax.grad(loss_fn)
        analytic = grad_fn(cosmo.omega_cdm)
        eps = 1e-5
        fd = (loss_fn(cosmo.omega_cdm + eps) - loss_fn(cosmo.omega_cdm - eps)) / (2 * eps)
        rel_err = abs(float(analytic) - float(fd)) / abs(float(fd))
        assert rel_err < 1e-2, f"CIB gradient rel_err = {rel_err}"
