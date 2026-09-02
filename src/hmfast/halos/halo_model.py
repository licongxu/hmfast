"""
Core halo model implementation using JAX for differentiability.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from scipy.integrate import trapezoid
from typing import Dict, Any, Optional, Callable
from functools import partial
from mcfit import TophatVar
from hmfast.utils import log_interp1d_extrap


def _simpson_nonuniform(y, x, axis=-1):
    """Composite Simpson's rule on a possibly non-uniform 1D grid.

    Uses the 3-point parabolic rule on each consecutive pair of intervals.
    For an odd number of intervals (even ``N``), the last (orphan) interval
    is handled with a 3-point parabolic correction using the last three
    samples (matches ``scipy.integrate.simpson``).

    Parameters
    ----------
    y : array
        Integrand evaluated at ``x``.
    x : 1D array
        Grid points (need not be uniform).
    axis : int
        Integration axis of ``y``.

    Returns
    -------
    integral : array
        Integral with ``axis`` reduced.
    """
    y = jnp.moveaxis(jnp.asarray(y), axis, -1)
    x = jnp.asarray(x)
    N = y.shape[-1]
    h = jnp.diff(x)  # (N-1,)

    if N < 3:
        return jnp.trapezoid(y, x=x, axis=-1)

    # Composite Simpson 1/3 over consecutive pairs of intervals.
    # n_pairs = number of (2-interval) Simpson pairs; uses 2*n_pairs intervals.
    n_pairs = (N - 1) // 2
    n_int = 2 * n_pairs

    h0 = h[0:n_int:2]                              # first  interval of each pair
    h1 = h[1:n_int:2]                              # second interval of each pair
    hsum = h0 + h1
    hprod = h0 * h1
    # 3-point parabolic formula (non-uniform Simpson):
    #   ∫_{x_{2i}}^{x_{2i+2}} y dx ≈ (h0+h1)/6 * [ y0*(2 - h1/h0) + y1*(h0+h1)^2/(h0 h1) + y2*(2 - h0/h1) ]
    y0 = y[..., 0:n_int:2]                         # samples at left of each pair
    y1 = y[..., 1:n_int:2]                         # samples at middle of each pair
    y2 = y[..., 2:n_int + 1:2]                     # samples at right of each pair
    seg = (hsum / 6.0) * (
        y0 * (2.0 - h1 / h0)
        + y1 * (hsum * hsum / hprod)
        + y2 * (2.0 - h0 / h1)
    )
    integral = jnp.sum(seg, axis=-1)

    if N % 2 == 0:
        # Orphan last interval [x_{N-2}, x_{N-1}] needs a 3-point correction
        # using the last three samples (matches scipy.integrate.simpson "last"
        # rule). Coefficients derive from the same parabolic interpolation.
        h_m1 = h[-1]
        h_m2 = h[-2]
        hs = h_m1 + h_m2
        a = (2.0 * h_m1 ** 2 + 3.0 * h_m1 * h_m2) / (6.0 * hs)
        b = (h_m1 ** 2 + 3.0 * h_m1 * h_m2) / (6.0 * h_m2)
        c = h_m1 ** 3 / (6.0 * h_m2 * hs)
        last = a * y[..., -1] + b * y[..., -2] - c * y[..., -3]
        integral = integral + last

    return integral


from hmfast.halos.massfunc import T08HaloMass, TW10SubHaloMass
from hmfast.halos.bias import T10HaloBias
from hmfast.halos.concentration import D08Concentration, B13Concentration
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.periodic import (
    cubic_lattice_shells,
    cubic_lattice_vectors,
    gaussian_cl_variance,
    interpolate_kz,
    integrate_radial_bessel,
    lattice_Q_ell,
    lattice_gaussian_cl_cov,
    lattice_wavenumbers,
    multipole_bin_weights,
    n_max_for_kmax,
)
from hmfast.cosmology import Cosmology

jax.config.update("jax_enable_x64", True)


class HaloModel:
    """
    Differentiable halo model.

    Provides halo-model predictions for arbitrary tracers using a configurable
    cosmology, halo mass function, halo bias model, concentration relation,
    and subhalo mass function.

    Attributes
    ----------
    cosmology : Cosmology
        Cosmology object supplying background, growth, and matter power spectra quantities.
    mass_definition : MassDefinition
        Native spherical-overdensity mass definition used throughout the halo model.
    halo_mass_function : HaloMass
        Halo mass function model used to compute :math:`dn / d\\ln M`.
    halo_bias : HaloBias
        Halo bias model used for large-scale halo bias predictions.
    subhalo_mass_function : SubHaloMass
        Subhalo mass function model used in observables with satellite or subhalo contributions.
    concentration : Concentration
        Halo concentration relation used to map halo mass and redshift to concentration.
    hm_consistency : bool
        Flag controlling whether halo-model consistency counterterms are applied.
    convert_masses : bool
        Flag controlling whether profile-specific native mass definitions are converted automatically.
    """

    def __init__(self, 
                 cosmology=Cosmology(emulator_set="lcdm:v1"), 
                 mass_definition=MassDefinition(delta=200, reference="critical"), 
                 halo_mass_function=T08HaloMass(), 
                 halo_bias=T10HaloBias(), 
                 subhalo_mass_function=TW10SubHaloMass(),
                 concentration=D08Concentration(), 
                 hm_consistency=True, 
                 convert_masses=False):
        """Initialize the halo model."""
        
        # Load cosmology and make sure the required files are loaded outside of jitted functions (note that DER is needed for CMB lensing tracers)
        self.cosmology = cosmology 
        self.cosmology._load_emulator("DAZ")
        self.cosmology._load_emulator("HZ")
        self.cosmology._load_emulator("PKL")
        self.cosmology._load_emulator("DER")
        
        self.halo_mass_function = halo_mass_function
        self.halo_bias = halo_bias
        self.subhalo_mass_function = subhalo_mass_function
        self.concentration = concentration

        self.mass_definition = mass_definition
        self.hm_consistency = hm_consistency
        self.convert_masses = convert_masses


        # Create TophatVar instance once to instantiate it
        dummy_k, _ = self.cosmology.pk(1., linear=True)
        self._tophat_instance = partial(TophatVar(dummy_k, lowring=True, backend='jax'), extrap=True)


    def _tree_flatten(self):
        # The cosmology is a Pytree, so it is a child.
        # Everything else is configuration/metadata.
        children = (self.cosmology,)
        aux_data = (self.halo_mass_function, self.halo_bias, self.subhalo_mass_function, self.concentration,
            self.mass_definition, self.hm_consistency, self.convert_masses, self._tophat_instance
        )
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        cosmology, = children
        obj = cls.__new__(cls)
        obj.cosmology = cosmology
        (obj.halo_mass_function, obj.halo_bias, obj.subhalo_mass_function, 
         obj.concentration, obj.mass_definition, obj.hm_consistency, 
         obj.convert_masses, obj._tophat_instance) = aux_data
        return obj

    def update(self, cosmology=None, halo_mass_function=None, halo_bias=None, subhalo_mass_function=None, concentration=None, mass_definition=None, 
               hm_consistency=None, convert_masses=None):
        """
        Return a new HaloModel instance with updated components.

        Parameters
        ----------
        cosmology, halo_mass_function, halo_bias, subhalo_mass_function, concentration, mass_definition, hm_consistency, convert_masses : optional
            Replacement values for the corresponding class attributes. Any argument left as ``None`` keeps its current value.

        Returns
        -------
        HaloModel
            New halo-model instance with updated attributes.
        """
        # Flatten current state
        children, aux_data = self._tree_flatten()
        # Unpack
        (cosmo_child,) = children
        (
            halo_mass_function0, halo_bias0, subhalo_mass_function0, concentration0,
            mass_definition0, hm_consistency0, convert_masses0, tophat_instance0
        ) = aux_data
    
        # Update only provided components
        new_cosmo = cosmology if cosmology is not None else cosmo_child
        new_halo_mass_function = halo_mass_function if halo_mass_function is not None else halo_mass_function0
        new_halo_bias = halo_bias if halo_bias is not None else halo_bias0
        new_subhalo_mass_function = subhalo_mass_function if subhalo_mass_function is not None else subhalo_mass_function0
        new_concentration = concentration if concentration is not None else concentration0
        new_mass_definition = mass_definition if mass_definition is not None else mass_definition0
        new_hm_consistency = hm_consistency if hm_consistency is not None else hm_consistency0
        new_convert_masses = convert_masses if convert_masses is not None else convert_masses0
    
        # Reuse the existing tophat instance (or update if needed)
        new_aux_data = (
            new_halo_mass_function, new_halo_bias, new_subhalo_mass_function, new_concentration,
            new_mass_definition, new_hm_consistency, new_convert_masses, tophat_instance0
        )
        # Use _tree_unflatten to create the new instance efficiently
        return self._tree_unflatten(new_aux_data, (new_cosmo,))
       
    @jax.jit
    def _counter_terms(self, m, z):
        """
        Compute :math:`n_{\\min}`, :math:`b_{1,\\min}`, and :math:`b_{2,\\min}` counter terms for halo model consistency.

        Parameters
        ----------
        m : array-like
            Halo mass grid in physical :math:`M_\\odot`.
        z : array-like
            Redshift(s).

        Returns
        -------
        n_min : array
            Minimum number density.
        b1_min : array
            Minimum linear bias.
        b2_min : array
            Minimum quadratic bias.
        """
       
        m = jnp.atleast_1d(m)
        cparams = self.cosmology._cosmo_params()
        h = self.cosmology.H0 / 100.0
        m_internal = m * h
        logm = jnp.log(m_internal)
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_cb"] / h**2   # internal halo-model normalization
        m_over_rho_mean = (m_internal / rho_mean_0)[:, None]  # (Nm, 1)


        # Public HMF and bias interfaces use physical masses.
        dn_dlnm = self.halo_mass_function.halo_mass_function(self, m=m, z=z)  # (Nm, Nz)
        b1 = self.halo_bias.halo_bias(self, m=m, z=z, order=1)      # (Nm, Nz)
        b2 = self.halo_bias.halo_bias(self, m=m, z=z, order=2)      # (Nm, Nz)
    
        # Compute integrals I0, I1, I2
        I0 = jnp.trapezoid(dn_dlnm * m_over_rho_mean, x=logm, axis=0)  # (Nz,)
        I1 = jnp.trapezoid(b1 * dn_dlnm * m_over_rho_mean, x=logm, axis=0)
        I2 = jnp.trapezoid(b2 * dn_dlnm * m_over_rho_mean, x=logm, axis=0)
    
        # Apply formulas
        m_min =  m_internal[0]
        n_min =  (1.0 - I0) * rho_mean_0 / m_min
        b1_min = (1.0 - I1) * rho_mean_0 / m_min / n_min
        b2_min = -I2 * rho_mean_0 / m_min / n_min
    
        return n_min, b1_min, b2_min


    @jax.jit
    def pk_1h(self, tracer1, tracer2, k, m, z,  k_damp=0.01):
        """
        Compute the 1-halo contribution to the 3D power spectrum in
        physical units.

        .. math::

            P_{1h}(k, z) = \\int d\\ln M \\, \\frac{dn}{d\\ln M} \\, u_1(k, M, z) u_2(k, M, z)

        where :math:`dn/d\\ln M` is the halo mass function 
        and :math:`u_i(k \\mid M, z)` is the Fourier-space tracer profile.

        Parameters
        ----------
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        k : array-like
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        m : array
            Mass array in physical :math:`M_\\odot`. This must be an array because it
            defines the integration grid over halo mass.
        z : array-like
            Redshift grid.
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` for the low-k suppression factor.

        Returns
        -------
        pk_1h : array
            1-halo power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`.
        """
    
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        
        # Weights and Setup
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5
        
        dndlnm = self.halo_mass_function.halo_mass_function(self, m, z)
        total_weights = dndlnm * w[:, None] # (Nm, Nz)

        # Use object identity (not ==) so JAX traces tracers as PyTrees; needed when B varies.
        tracer2 = tracer1 if tracer2 is None else tracer2
        is_same_tracer = tracer1 is tracer2

        # Process a single mass bin at a time and extract the uk^2 at the lowest mass for the halo model consistency term
        def process_bin(i):
            # We need the profiles for index 'i' while squaring uk if the user is doing an autocorrelation
            if is_same_tracer:
                if tracer1.profile.has_central_contribution:
                    s1, c1 = tracer1.profile._sat_and_cen_contribution(self, k, m, z)
                    uk_sq_row = s1[:, i, :] * s1[:, i, :] + 2.0 * s1[:, i, :] * c1[:, i, :]
                else:
                    u1 = tracer1.profile.u_k(self, k, m, z)
                    uk_sq_row = u1[:, i, :] ** 2
            elif tracer1.profile.has_central_contribution and tracer2.profile.has_central_contribution:
                s1, c1 = tracer1.profile._sat_and_cen_contribution(self, k, m, z)
                s2, c2 = tracer2.profile._sat_and_cen_contribution(self, k, m, z)
                uk_sq_row = s1[:, i, :] * s2[:, i, :] + s1[:, i, :] * c2[:, i, :] + s2[:, i, :] * c1[:, i, :]
            else:
                u1 = tracer1.profile.u_k(self, k, m, z)
                u2 = tracer2.profile.u_k(self, k, m, z)
                uk_sq_row = u1[:, i, :] * u2[:, i, :]
    
            return uk_sq_row * total_weights[i], uk_sq_row
    
        # vmap through the mass bins
        integrand_rows, all_sq_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))
    
        pk1h = jnp.sum(integrand_rows, axis=0)
    
        # Apply halo model consistency correction: n_min * uk_sq_min 
        uk_sq_min = all_sq_profiles[0] 
        n_min, _, _ = self._counter_terms(m, z)
        correction = n_min[None, :] * uk_sq_min
        pk1h = pk1h + self.hm_consistency * correction
    
        # Apply damping
        mask = k_damp > 0
        damping = jnp.where(mask, 1.0 - jnp.exp(-(k / jnp.where(mask, k_damp, 1.0))**2), 1.0)
    
        return pk1h * damping[:, None]
            
       
    @jax.jit
    def cl_1h(self, tracer1, tracer2, l, m, z, k_damp=0.01):
        """
        Compute the 1-halo contribution to the angular power spectrum
        :math:`C_\\ell^{1h}`.

        The Limber-projected spectrum is obtained by integrating the 1-halo
        3D power spectrum against the tracer kernels and the comoving volume
        element written in the legacy :math:`(\\mathrm{Mpc}/h)^3` convention used by the
        current tracer kernels.

        Parameters
        ----------
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        l : array-like
            Multipole grid.
        m : array
            Mass array in physical :math:`M_\\odot`. This must be an array because it
            defines the integration grid over halo mass.
        z : array
            Redshift array. This must be an array because it defines the
            integration grid over redshift.
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` passed through to :meth:`pk_1h`.

        Returns
        -------
        cl_1h : array
            Dimensionless 1-halo angular power spectrum with shape
            :math:`(N_\\ell,)`.
        """

        tracer2 = tracer1 if tracer2 is None else tracer2

        # Define the slice function to map l -> k for a specific z
        def get_pk_slice(zi):
            chi_i = self.cosmology.angular_diameter_distance(zi) * (1 + zi) 
            ki = (l + 0.5) / chi_i
            pk = self.pk_1h(tracer1, tracer2, k=ki, m=m, z=jnp.atleast_1d(zi), k_damp=k_damp)
            return pk.flatten()

        # Get the halo model pk_1h, the kernel, and the comoving volume
        P_1h_grid = jax.vmap(get_pk_slice)(z)
        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        # Comoving volume in physical Mpc³ (paired with HMF in 1/Mpc³).
        comov_vol = self.cosmology.comoving_volume_element(z)

        # Integrate over redshift
        integrand = P_1h_grid * (comov_vol[:, None] * kernel1[:, None] * kernel2[:, None])
        
        return jnp.trapezoid(integrand, x=z, axis=0)
    


    @partial(jax.jit, static_argnames=("linear",))
    def pk_2h(self, tracer1, tracer2, k, m, z, linear=True):
        """
        Compute the 2-halo contribution to the 3D power spectrum in
        physical units.

        .. math::

            P_{2h}(k, z) = P_{\\mathrm{m}}(k, z) \\, I_1(k, z) \\, I_2(k, z)
        
        with
        
        .. math::
        
            I_i(k, z) = \\int d\\ln M \\, \\frac{dn}{d\\ln M}(M, z) \\, b(M, z) \\, u_i(k \\mid M, z),
        
        where :math:`u_i(k \\mid M, z)` is the Fourier-space tracer profile,
        :math:`dn/d\\ln M` is the halo mass function, and :math:`b(M, z)` is the
        linear halo bias. :math:`P_{\\mathrm{m}}` is the linear matter power
        spectrum by default (``linear=True``); set ``linear=False`` to use the
        nonlinear matter power spectrum instead.

        Parameters
        ----------
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        k : array-like
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        m : array
            Mass array in physical :math:`M_\\odot`. This must be an array because it
            defines the integration grid over halo mass.
        z : array-like
            Redshift grid.
        linear : bool, optional
            If True (default) use the linear matter power spectrum
            :math:`P_{\\mathrm{lin}}`; if False use the nonlinear matter power
            spectrum :math:`P_{\\mathrm{nl}}`. Passed through to
            :meth:`Cosmology.pk`.

        Returns
        -------
        pk_2h : array
            2-halo power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`.
        """
        
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        tracer2 = tracer1 if tracer2 is None else tracer2
    
        # Weights and Ingredients
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        # Combine hmf, bias, and weights into a single (Nm, Nz) weight grid
        dndlnm = self.halo_mass_function.halo_mass_function(self, m, z)
        bias = self.halo_bias.halo_bias(self, m, z)
        total_weights = dndlnm * bias * w[:, None]
    
        def get_I(tracer):
            # This function processes a single index 'i' of the mass axis
            def process_bin(i):
                uk_full = tracer.profile.u_k(self, k, m, z)
                uk_slice = uk_full[:, i, :] 
                return uk_slice * total_weights[i], uk_slice
    
            # Vmap over the indices 0...Nm-1, then integrate and pluck index 0 for hm consistency
            integrand_rows, all_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))
            integral = jnp.sum(integrand_rows, axis=0)
            u_k_min = all_profiles[0] # vmap output is (Nm, Nk, Nz)
    
            n_min, b1_min, _ = self._counter_terms(m, z)
            correction = b1_min[None, :] * n_min[None, :] * u_k_min
            
            return integral + self.hm_consistency * correction
    
        # Final Power Spectrum
        I1 = get_I(tracer1)
        I2 = I1 if tracer1 is tracer2 else get_I(tracer2)
        
        # Cosmology.pk already returns physical k [Mpc^{-1}] and P [Mpc^3].
        # Log-log interpolation matches official hmfast.Cosmology.pk(k, z).
        P_m = jax.vmap(lambda zi: log_interp1d_extrap(k, *self.cosmology.pk(zi, linear=linear)))(z).T

        return P_m * I1 * I2


    @jax.jit
    def trispectrum_1h(self, tracer1, tracer2, l1, l2, m, z, k_damp=0.0):
        """
        Compute the 1-halo connected angular trispectrum
        :math:`T^{1h}_{\\ell\\ell'}` for two tracers via the Limber approximation.

        .. math::

            T^{1h}_{\\ell\\ell'} = \\int dz \\, \\frac{dV}{dz\\,d\\Omega} \\,
                W_1(z)^2 W_2(z)^2
                \\int d\\ln M \\, \\frac{dn}{d\\ln M}\\,
                |u_1(k_\\ell, M, z)|^2 |u_2(k_{\\ell'}, M, z)|^2

        with :math:`k_\\ell = (\\ell + 1/2) / \\chi(z)`. The user supplies the
        :math:`\\ell` grids; both axes can be different lengths and need not
        coincide.

        Parameters
        ----------
        tracer1, tracer2 : Tracer or None
            Tracers for the two multipole axes. ``tracer2=None`` means
            :math:`\\ell` and :math:`\\ell'` use the same tracer (auto trispectrum).
        l1, l2 : array-like
            Multipole grids for the two axes.
        m : array
            Halo-mass grid in physical :math:`M_\\odot`.
        z : array
            Redshift grid.
        k_damp : float, default 0.0
            Optional low-:math:`k` damping wavenumber. ``0`` disables damping
            (matches the tszpower trispectrum convention).

        Returns
        -------
        T : array
            Trispectrum with shape :math:`(N_{\\ell_1}, N_{\\ell_2})`.
        """

        tracer2 = tracer1 if tracer2 is None else tracer2

        l1 = jnp.atleast_1d(l1)
        l2 = jnp.atleast_1d(l2)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        logm = jnp.log(m)

        # Mass-integration weights (trapezoid in ln M)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        dndlnm = self.halo_mass_function.halo_mass_function(self, m, z)  # (Nm, Nz)

        # Damping factor in k (shared between the two ell axes per redshift slice)
        damp_mask = k_damp > 0

        def _damping(k_arr):
            return jnp.where(damp_mask,
                             1.0 - jnp.exp(-(k_arr / jnp.where(damp_mask, k_damp, 1.0))**2),
                             1.0)

        is_same_tracer = tracer1 is tracer2

        def trisp_slice(i):
            zi = z[i]
            chi_i = self.cosmology.angular_diameter_distance(zi) * (1.0 + zi)
            k1 = (l1 + 0.5) / chi_i
            k2 = (l2 + 0.5) / chi_i

            u1_a = tracer1.profile.u_k(self, k1, m, jnp.atleast_1d(zi))[:, :, 0]  # (Nl1, Nm)
            u1_b = tracer2.profile.u_k(self, k2, m, jnp.atleast_1d(zi))[:, :, 0]  # (Nl2, Nm)

            if is_same_tracer:
                u2_a = u1_a
                u2_b = u1_b
            else:
                u2_a = tracer2.profile.u_k(self, k1, m, jnp.atleast_1d(zi))[:, :, 0]
                u2_b = tracer1.profile.u_k(self, k2, m, jnp.atleast_1d(zi))[:, :, 0]

            usq_a = u1_a * u2_a  # (Nl1, Nm)
            usq_b = u1_b * u2_b  # (Nl2, Nm)

            damp_a = _damping(k1)[:, None]
            damp_b = _damping(k2)[:, None]
            usq_a = usq_a * damp_a
            usq_b = usq_b * damp_b

            # Mass integral: weighted by dn/dlnM at this redshift
            weight = dndlnm[:, i] * w  # (Nm,)
            integrand = usq_a[:, None, :] * usq_b[None, :, :] * weight[None, None, :]
            T_z = jnp.sum(integrand, axis=-1)  # (Nl1, Nl2)
            return T_z

        T_grid = jax.vmap(trisp_slice)(jnp.arange(z.shape[0]))  # (Nz, Nl1, Nl2)

        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        # Comoving volume in physical Mpc³ (paired with HMF in 1/Mpc³ after the
        # massfunc.py h³-removal). The previous ×h³ paired with the now-removed
        # /h³ in halo_mass_function.
        comov_vol = self.cosmology.comoving_volume_element(z)

        weight_z = comov_vol * (kernel1 ** 2) * (kernel2 ** 2)
        integrand_z = T_grid * weight_z[:, None, None]
        return jnp.trapezoid(integrand_z, x=z, axis=0)

    @partial(jax.jit, static_argnames=())
    def trispectrum_1h_masked(self, tracer1, tracer2, l1, l2, m, z, mask_mz, k_damp=0.0):
        """Compute the 1-halo connected angular trispectrum with a user-supplied
        :math:`(M, z)` selection mask applied to the mass integrand.

        .. math::

            T^{1h,\\mathrm{masked}}_{\\ell\\ell'} = \\int dz \\,
                \\frac{dV}{dz\\,d\\Omega} \\,
                W_1(z)^2 W_2(z)^2
                \\int d\\ln M \\, \\frac{dn}{d\\ln M}\\,
                |u_1(k_\\ell)|^2 |u_2(k_{\\ell'})|^2 \\, \\mathcal{M}(M, z)

        Parameters
        ----------
        tracer1, tracer2 : Tracer or None
        l1, l2 : array-like
        m, z : array
        mask_mz : array of shape (N_m, N_z)
            Mask values in :math:`[0, 1]`.
        k_damp : float, default 0.0

        Returns
        -------
        T : array, shape (N_l1, N_l2)
        """
        tracer2 = tracer1 if tracer2 is None else tracer2

        l1 = jnp.atleast_1d(l1)
        l2 = jnp.atleast_1d(l2)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        dndlnm = self.halo_mass_function.halo_mass_function(self, m, z)  # (Nm, Nz)

        damp_mask = k_damp > 0

        def _damping(k_arr):
            return jnp.where(damp_mask,
                             1.0 - jnp.exp(-(k_arr / jnp.where(damp_mask, k_damp, 1.0))**2),
                             1.0)

        is_same_tracer = tracer1 is tracer2

        def trisp_slice(i):
            zi = z[i]
            chi_i = self.cosmology.angular_diameter_distance(zi) * (1.0 + zi)
            k1 = (l1 + 0.5) / chi_i
            k2 = (l2 + 0.5) / chi_i

            u1_a = tracer1.profile.u_k(self, k1, m, jnp.atleast_1d(zi))[:, :, 0]
            u1_b = tracer2.profile.u_k(self, k2, m, jnp.atleast_1d(zi))[:, :, 0]

            if is_same_tracer:
                u2_a = u1_a
                u2_b = u1_b
            else:
                u2_a = tracer2.profile.u_k(self, k1, m, jnp.atleast_1d(zi))[:, :, 0]
                u2_b = tracer1.profile.u_k(self, k2, m, jnp.atleast_1d(zi))[:, :, 0]

            usq_a = u1_a * u2_a
            usq_b = u1_b * u2_b

            damp_a = _damping(k1)[:, None]
            damp_b = _damping(k2)[:, None]
            usq_a = usq_a * damp_a
            usq_b = usq_b * damp_b

            weight = dndlnm[:, i] * w * mask_mz[:, i]  # (Nm,)
            integrand = usq_a[:, None, :] * usq_b[None, :, :] * weight[None, None, :]
            T_z = jnp.sum(integrand, axis=-1)  # (Nl1, Nl2)
            return T_z

        T_grid = jax.vmap(trisp_slice)(jnp.arange(z.shape[0]))  # (Nz, Nl1, Nl2)

        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        # Comoving volume in physical Mpc³ (paired with HMF in 1/Mpc³ after the
        # massfunc.py h³-removal). The previous ×h³ paired with the now-removed
        # /h³ in halo_mass_function.
        comov_vol = self.cosmology.comoving_volume_element(z)

        weight_z = comov_vol * (kernel1 ** 2) * (kernel2 ** 2)
        integrand_z = T_grid * weight_z[:, None, None]
        return jnp.trapezoid(integrand_z, x=z, axis=0)


    @partial(jax.jit, static_argnames=())
    def cl_1h_masked(self, tracer1, tracer2, l, m, z, mask_mz, k_damp=0.01):
        """
        Compute the 1-halo angular power spectrum with a user-supplied
        :math:`(M, z)` selection mask applied to the integrand.

        .. math::

            C_\\ell^{1h, \\mathrm{masked}} = \\int dz \\,
                \\frac{dV}{dz\\,d\\Omega} W_1 W_2
                \\int d\\ln M \\, \\frac{dn}{d\\ln M}
                u_1(k_\\ell, M, z) u_2(k_\\ell, M, z) \\, \\mathcal{M}(M, z)

        where :math:`\\mathcal{M}(M, z)` is the supplied mask. This is the same
        Limber projection used by :meth:`cl_1h`, but with the inner mass
        integral weighted pointwise by ``mask_mz``. The halo-model consistency
        counterterm is dropped here since it is calibrated to the unmasked
        integral.

        Parameters
        ----------
        tracer1, tracer2 : Tracer or None
            Same conventions as :meth:`cl_1h`.
        l : array-like
            Multipole grid.
        m : array
            Halo-mass grid in physical :math:`M_\\odot`. Must match the first
            axis of ``mask_mz``.
        z : array
            Redshift grid. Must match the second axis of ``mask_mz``.
        mask_mz : array
            Mask of shape :math:`(N_m, N_z)`. Values in :math:`[0, 1]` are
            expected but any nonnegative weights are supported.
        k_damp : float, default 0.01
            Low-:math:`k` damping passed through to the underlying integrand.

        Returns
        -------
        cl_1h_masked : array
            1-halo angular power spectrum on the masked map with shape
            :math:`(N_\\ell,)`.
        """

        tracer2 = tracer1 if tracer2 is None else tracer2

        l = jnp.atleast_1d(l)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        logm = jnp.log(m)

        dndlnm = self.halo_mass_function.halo_mass_function(self, m, z)  # (Nm, Nz)

        is_same_tracer = tracer1 is tracer2

        damp_mask = k_damp > 0

        def _damping(k_arr):
            return jnp.where(damp_mask,
                             1.0 - jnp.exp(-(k_arr / jnp.where(damp_mask, k_damp, 1.0))**2),
                             1.0)

        def slice_z(i):
            zi = z[i]
            chi_i = self.cosmology.angular_diameter_distance(zi) * (1.0 + zi)
            ki = (l + 0.5) / chi_i
            u1 = tracer1.profile.u_k(self, ki, m, jnp.atleast_1d(zi))[:, :, 0]  # (Nl, Nm)
            if is_same_tracer:
                u_sq = u1 * u1
            else:
                u2 = tracer2.profile.u_k(self, ki, m, jnp.atleast_1d(zi))[:, :, 0]
                u_sq = u1 * u2
            damp = _damping(ki)[:, None]
            u_sq = u_sq * damp
            # Mass integral via trapezoid in ln M (log-log HMF interp is the
            # dominant accuracy fix; Simpson here gives only ~0.05% extra at
            # 7x runtime cost on GPU).
            mass_integrand = u_sq * (dndlnm[:, i] * mask_mz[:, i])[None, :]  # (Nl, Nm)
            return jnp.trapezoid(mass_integrand, x=logm, axis=-1)            # (Nl,)

        pk_grid = jax.vmap(slice_z)(jnp.arange(z.shape[0]))  # (Nz, Nl)

        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        # Use the comoving volume element in PHYSICAL Mpc³, matching the
        # 1/Mpc³ convention now used by halo_mass_function. The two h³ factors
        # (one on dV, the inverse on dn/dlnM) cancel mathematically but their
        # explicit cancellation was a source of sub-percent floating-point drift.
        comov_vol = self.cosmology.comoving_volume_element(z)
        weight_z = comov_vol * kernel1 * kernel2
        integrand = pk_grid * weight_z[:, None]
        # z-integral via trapezoid (log-log HMF interp is the dominant fix).
        return jnp.trapezoid(integrand, x=z, axis=0)


    @jax.jit
    def cl_1h_integrand(self, tracer1, tracer2, l, m, z, mask_mz=None, k_damp=0.01):
        """
        Compute the Limber 1-halo integrand :math:`d^2 C_\\ell / (dz\\, d\\ln M)`
        on the full :math:`(z, \\ell, M)` grid, without integrating.

        .. math::

            \\frac{d^2 C_\\ell^{1h}}{dz\\, d\\ln M} =
                \\frac{dV}{dz\\, d\\Omega} W_1(z) W_2(z)
                \\frac{dn}{d\\ln M}
                u_1(k_\\ell, M, z)\\, u_2(k_\\ell, M, z)\\, \\mathcal{M}(M, z)

        Integrating this array with ``jnp.trapezoid`` over :math:`\\ln M` (last
        axis) and then :math:`z` (first axis) reproduces :meth:`cl_1h_masked`
        to float round-off (same quadrature). As in :meth:`cl_1h_masked`, the
        halo-model consistency counterterm is dropped, so the integral differs
        from :meth:`cl_1h` by that counterterm when ``hm_consistency=True``.
        Useful for kernel-analysis plots showing which halos contribute to
        :math:`C_\\ell` in the :math:`(z, M)` plane. Memory scales as
        :math:`N_z N_\\ell N_m \\times 8` bytes times a small integer (u_k
        intermediates), so bound the grid accordingly.

        Parameters
        ----------
        tracer1, tracer2 : Tracer or None
            Same conventions as :meth:`cl_1h`.
        l : array-like
            Multipole grid.
        m : array
            Halo-mass grid in physical :math:`M_\\odot`.
        z : array
            Redshift grid.
        mask_mz : array or None
            Optional selection weights of shape :math:`(N_m, N_z)` applied
            pointwise to the integrand (e.g. a completeness-based mask).
            ``None`` means unit weights.
        k_damp : float, default 0.01
            Low-:math:`k` damping, as in :meth:`cl_1h`.

        Returns
        -------
        integrand : array
            :math:`d^2 C_\\ell / (dz\\, d\\ln M)` with shape
            :math:`(N_z, N_\\ell, N_m)`.
        """

        tracer2 = tracer1 if tracer2 is None else tracer2

        l = jnp.atleast_1d(l)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)

        dndlnm = self.halo_mass_function.halo_mass_function(self, m, z)  # (Nm, Nz)
        weights_mz = dndlnm if mask_mz is None else dndlnm * mask_mz

        is_same_tracer = tracer1 is tracer2

        damp_mask = k_damp > 0

        def _damping(k_arr):
            return jnp.where(damp_mask,
                             1.0 - jnp.exp(-(k_arr / jnp.where(damp_mask, k_damp, 1.0))**2),
                             1.0)

        def slice_z(i):
            zi = z[i]
            chi_i = self.cosmology.angular_diameter_distance(zi) * (1.0 + zi)
            ki = (l + 0.5) / chi_i
            u1 = tracer1.profile.u_k(self, ki, m, jnp.atleast_1d(zi))[:, :, 0]  # (Nl, Nm)
            if is_same_tracer:
                u_sq = u1 * u1
            else:
                u2 = tracer2.profile.u_k(self, ki, m, jnp.atleast_1d(zi))[:, :, 0]
                u_sq = u1 * u2
            u_sq = u_sq * _damping(ki)[:, None]
            return u_sq * weights_mz[:, i][None, :]  # (Nl, Nm)

        grid = jax.vmap(slice_z)(jnp.arange(z.shape[0]))  # (Nz, Nl, Nm)

        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        comov_vol = self.cosmology.comoving_volume_element(z)
        weight_z = comov_vol * kernel1 * kernel2  # (Nz,)
        return grid * weight_z[:, None, None]


    @partial(jax.jit, static_argnames=("linear",))
    def cl_2h(self, tracer1, tracer2, l, m, z, linear=True):
        """
        Compute the 2-halo contribution to the angular power spectrum
        :math:`C_\\ell^{2h}`.

        The Limber-projected spectrum is obtained by integrating the 2-halo
        3D power spectrum against the tracer kernels and the comoving volume
        element written in the legacy :math:`(\\mathrm{Mpc}/h)^3` convention used by the
        current tracer kernels.

        Parameters
        ----------
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        l : array-like
            Multipole grid.
        m : array
            Mass array in physical :math:`M_\\odot`. This must be an array because it
            defines the integration grid over halo mass.
        z : array
            Redshift array. This must be an array because it defines the
            integration grid over redshift.
        linear : bool, optional
            If True (default) build the 2-halo term from the linear matter
            power spectrum; if False use the nonlinear one. Forwarded to
            :meth:`pk_2h`.

        Returns
        -------
        cl_2h : array
            Dimensionless 2-halo angular power spectrum with shape
            :math:`(N_\\ell,)`.
        """
        tracer2 = tracer1 if tracer2 is None else tracer2

        # Define the slice function for Limber integration
        def get_pk_slice(zi):
            # Map l to k using the Limber approximation and then get the pk_2h
            chi_i = self.cosmology.angular_diameter_distance(zi) * (1 + zi)
            ki = (l + 0.5) / chi_i
            return self.pk_2h(tracer1, tracer2, k=ki, m=m, z=jnp.atleast_1d(zi), linear=linear).flatten()
    
        # Map over redshift to get P(k=l/chi, z)
        P_2h_grid = jax.vmap(get_pk_slice)(z) 
        
        # Get individual kernels
        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)

        # Comoving volume in physical Mpc³ (paired with HMF in 1/Mpc³).
        comov_vol = self.cosmology.comoving_volume_element(z)
    
        # Limber Integral: C_l = int dz P(k,z) * [W1 * W2 * dV/dz]
        integrand = P_2h_grid * (comov_vol[:, None] * kernel1[:, None] * kernel2[:, None])

        return jnp.trapezoid(integrand, x=z, axis=0)


    @partial(jax.jit, static_argnames=())
    def cl_2h_masked(self, tracer1, tracer2, l, m, z, mask_mz):
        r"""
        Compute the 2-halo angular power spectrum with a user-supplied
        :math:`(M, z)` selection mask applied to each bias-weighted bracket.

        .. math::

            C_\ell^{2h,\mathrm{masked}} = \int dz \,
                \frac{dV}{dz\,d\Omega}\, W_1 W_2 \,
                P_{\mathrm{lin}}(k_\ell, z) \,
                I_1^{\mathrm{masked}}(k_\ell, z)\,
                I_2^{\mathrm{masked}}(k_\ell, z)

        with

        .. math::

            I_a^{\mathrm{masked}}(k, z) = \int d\ln M \,
                \frac{dn}{d\ln M}(M, z)\, b_1(M, z)\,
                \mathcal{W}_1(M, z)\, u_a(k \mid M, z),

        and :math:`k_\ell = (\ell + 1/2)/\chi(z)`.

        Notes
        -----
        Unlike the 1-halo term, the 2-halo term is a product of two
        *separate* mass integrals, each linear in the profile and each
        sourced by a *distinct* halo. With log-normal intrinsic scatter
        :math:`\ln A \sim \mathcal{N}(0, \sigma_{\ln Y}^2)` the per-halo
        amplitudes are independent, so the scatter expectation factorizes
        and **each bracket** carries the conditional *first* moment

        .. math::

            \mathcal{W}_1(M, z) =
                \langle A\, \mathbf{1}(q_{\mathrm{obs}} < q_{\mathrm{cat}})\rangle.

        Pass this ``n_power=1`` weight as ``mask_mz`` (e.g.
        :func:`hmfast.tracers.tsz_completeness.conditional_An_undetected`
        with ``n_power=1``), in contrast to the ``n_power=2`` weight used by
        :meth:`cl_1h_masked`. The halo-model consistency counterterm is
        dropped, consistent with :meth:`cl_1h_masked`.

        For the tSZ auto-spectrum the same mask weights both brackets. A
        cross-spectrum in which only one tracer is masked would instead
        apply ``mask_mz`` to a single bracket; that case is not handled here.

        Parameters
        ----------
        tracer1, tracer2 : Tracer or None
            Same conventions as :meth:`cl_2h`.
        l : array-like
            Multipole grid.
        m : array
            Halo-mass grid in physical :math:`M_\odot`. Must match the first
            axis of ``mask_mz``.
        z : array
            Redshift grid. Must match the second axis of ``mask_mz``.
        mask_mz : array
            Mask of shape :math:`(N_m, N_z)`, the conditional first moment
            :math:`\mathcal{W}_1`. Values in :math:`[0, 1]` are expected for
            a Heaviside selection; larger values occur with scatter because
            :math:`\langle A\rangle = e^{\sigma_{\ln Y}^2/2} > 1`.

        Returns
        -------
        cl_2h_masked : array
            2-halo angular power spectrum on the masked map with shape
            :math:`(N_\ell,)`.
        """
        tracer2 = tracer1 if tracer2 is None else tracer2

        l = jnp.atleast_1d(l)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        logm = jnp.log(m)

        dndlnm = self.halo_mass_function.halo_mass_function(self, m, z)  # (Nm, Nz)
        bias = self.halo_bias.halo_bias(self, m, z)                     # (Nm, Nz)

        is_same_tracer = tracer1 is tracer2

        def slice_z(i):
            zi = z[i]
            chi_i = self.cosmology.angular_diameter_distance(zi) * (1.0 + zi)
            ki = (l + 0.5) / chi_i

            # Bias- and mask-weighted mass weight, shared by both brackets.
            wz = (dndlnm[:, i] * bias[:, i] * mask_mz[:, i])[None, :]  # (1, Nm)

            u1 = tracer1.profile.u_k(self, ki, m, jnp.atleast_1d(zi))[:, :, 0]  # (Nl, Nm)
            # Mass integral via trapezoid in ln M, matching cl_1h_masked.
            I1 = jnp.trapezoid(u1 * wz, x=logm, axis=-1)                       # (Nl,)
            if is_same_tracer:
                I2 = I1
            else:
                u2 = tracer2.profile.u_k(self, ki, m, jnp.atleast_1d(zi))[:, :, 0]
                I2 = jnp.trapezoid(u2 * wz, x=logm, axis=-1)

            p_lin = log_interp1d_extrap(ki, *self.cosmology.pk(zi, linear=True))
            return p_lin * I1 * I2  # (Nl,)

        pk_grid = jax.vmap(slice_z)(jnp.arange(z.shape[0]))  # (Nz, Nl)

        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        comov_vol = self.cosmology.comoving_volume_element(z)
        integrand = pk_grid * (comov_vol * kernel1 * kernel2)[:, None]
        return jnp.trapezoid(integrand, x=z, axis=0)


    def bias_weighted_I(self, tracer, k, m, z):
        """Bias-weighted halo integral :math:`I_1(k, z)` used by the 2-halo term.

        .. math::

            I_1(k, z) = \\int d\\ln M \\, \\frac{dn}{d\\ln M}\\, b_1(M, z)\\,
                u(k \\mid M, z)

        plus the same halo-model consistency counterterm as :meth:`pk_2h`.
        The profile :math:`u` is the existing Fourier-space tracer profile
        (for tSZ this is the implemented angular :math:`y_\\ell` evaluated at
        :math:`\\ell = k\\chi-1/2`).

        Parameters
        ----------
        tracer : Tracer
            Tracer whose profile enters the integral.
        k : array-like
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        m : array
            Halo-mass grid in physical :math:`M_\\odot`.
        z : array-like
            Redshift grid.

        Returns
        -------
        I : array
            :math:`I_1(k, z)` with shape :math:`(N_k, N_z)`.
        """
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        dndlnm = self.halo_mass_function.halo_mass_function(self, m, z)
        bias = self.halo_bias.halo_bias(self, m, z)
        total_weights = dndlnm * bias * w[:, None]

        uk = tracer.profile.u_k(self, k, m, z)
        integral = jnp.sum(uk * total_weights[None, :, :], axis=1)

        n_min, b1_min, _ = self._counter_terms(m, z)
        correction = b1_min[None, :] * n_min[None, :] * uk[:, 0, :]
        return integral + self.hm_consistency * correction


    def _emulator_pk(self, k, z, linear=True):
        r"""Matter :math:`P(k,z)` from :meth:`Cosmology.pk` (physical Mpc, no \(h\) rescaling)."""
        k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)
        return jax.vmap(
            lambda zi: log_interp1d_extrap(
                k, *self.cosmology.pk(zi, linear=linear)
            )
        )(z).T


    def radial_transfer_R_ell(self, tracer, l, k, m, z, linear=True, n_chi=None,
                              I=None, P_m=None):
        r"""Non-Limber radial transfer :math:`R_\ell(k)` in existing :math:`C_\ell` units.

        The PDF defines
        :math:`R_\ell(k)=\int d\chi\, W_y(\chi)\, I_{11}(k,z)\sqrt{P_{\mathrm{lin}}}\,
        j_\ell(k\chi)`. Mapped onto this repo's Limber measure
        :math:`\int dz\,(dV/dz\,d\Omega)\, W(z)^2 P_m I_1^2` with
        :math:`P_m` from :meth:`Cosmology.pk` that becomes

        .. math::

            R_\ell(k) = \int dz\, \frac{dV}{dz\,d\Omega}\, W(z)\,
                I_1(k, z)\, \sqrt{P_m(k, z)}\, j_\ell\bigl(k\,\chi(z)\bigr)

        with the same :math:`I_1` as :meth:`pk_2h` and :math:`P_m` from
        :meth:`Cosmology.pk`. This is **not** a Limber evaluation at
        :math:`k=(\ell+1/2)/\chi`.

        The slow weight is evaluated on the supplied :math:`(k,z)` grid and
        interpolated onto a finer radial mesh (``n_chi``) so :math:`j_\ell`
        is resolved. At large :math:`L` or high :math:`\ell` (small scales)
        the lattice sum of these :math:`R_\ell` approaches the usual Limber
        2-halo term.

        Parameters
        ----------
        tracer : Tracer
            Line-of-sight kernel and halo profile.
        l : array-like
            Multipoles :math:`\ell` (integer values are expected).
        k : array-like
            Three-dimensional wavenumbers in :math:`\mathrm{Mpc}^{-1}`
            (lattice magnitudes :math:`k_s`, not Limber :math:`k_\ell`).
        m : array
            Halo-mass grid in physical :math:`M_\odot`.
        z : array
            Redshift grid on which :math:`I_1` and :math:`P_m` are computed.
        linear : bool, optional
            If True (default) use the linear matter power spectrum.
        n_chi : int, optional
            Number of radial samples for the Bessel integral. ``None``
            chooses a grid that resolves :math:`k_{\\max}\\chi`.
        I, P_m : array, optional
            Precomputed :math:`I_1(k,z)` and :math:`P_m(k,z)` with shape
            :math:`(N_k, N_z)`. Used to avoid repeating the mass integral.

        Returns
        -------
        R : array
            :math:`R_\ell(k)` with shape :math:`(N_\ell, N_k)`.
        """
        import numpy as np

        l = jnp.atleast_1d(l)
        k = jnp.atleast_1d(k)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)

        if I is None:
            I = self.bias_weighted_I(tracer, k, m, z)
        if P_m is None:
            P_m = self._emulator_pk(k, z, linear=linear)
        sqrtP = jnp.sqrt(jnp.clip(jnp.asarray(P_m), 0.0, None))

        chi = self.cosmology.angular_diameter_distance(z) * (1.0 + z)
        comov_vol = self.cosmology.comoving_volume_element(z)
        W = tracer.kernel(self.cosmology, z)
        weight = (comov_vol * W)[None, :] * jnp.asarray(I) * sqrtP

        R = integrate_radial_bessel(l, k, z, chi, weight, n_chi=n_chi)
        return jnp.asarray(R)


    def cl_2h_nonlimber(
        self,
        tracer1,
        tracer2,
        l,
        m,
        z,
        l_limber=0.0,
        k=None,
        n_k=128,
        n_chi=None,
        linear=True,
    ):
        r"""Continuum non-Limber 2-halo spectrum with a Limber switch.

        This exposes the same l_limber convention as upstream hmfast:
        multipoles below the threshold use the exact radial-Bessel projection,
        while multipoles at or above it use the existing Limber method. The default
        l_limber=0 therefore preserves the existing Limber result exactly.

        The exact continuum projection is

        .. math::

            C_\ell^{2h} = \frac{2}{\pi}\int dk\,k^2
                R_\ell^{(1)}(k)R_\ell^{(2)}(k),

        using the same transfer function as the periodic 2-halo method. If k is
        omitted, n_k logarithmic samples span the cosmology's native
        power-spectrum grid.
        """
        import numpy as np

        tracer2 = tracer1 if tracer2 is None else tracer2
        ell = np.atleast_1d(np.asarray(l, dtype=float))
        low = ell < float(l_limber)
        result = np.empty(ell.size, dtype=float)

        if np.any(~low):
            result[~low] = np.asarray(
                self.cl_2h(
                    tracer1,
                    tracer2,
                    jnp.asarray(ell[~low]),
                    m,
                    z,
                    linear=linear,
                )
            )
        if not np.any(low):
            return jnp.asarray(result)

        if k is None:
            n_k = int(n_k)
            if n_k < 2:
                raise ValueError("n_k must be >= 2")
            k_native, _ = self.cosmology.pk(jnp.atleast_1d(z)[0], linear=linear)
            k = np.geomspace(float(k_native[0]), float(k_native[-1]), n_k)
        else:
            k = np.atleast_1d(np.asarray(k, dtype=float))
            if k.size < 2 or np.any(~np.isfinite(k)) or np.any(k <= 0.0):
                raise ValueError("k must contain at least two positive finite values.")
            if np.any(np.diff(k) <= 0.0):
                raise ValueError("k must be strictly increasing.")

        z = jnp.atleast_1d(z)
        m = jnp.atleast_1d(m)
        I1 = self.bias_weighted_I(tracer1, k, m, z)
        P_m = self._emulator_pk(k, z, linear=linear)
        R1 = np.asarray(
            self.radial_transfer_R_ell(
                tracer1,
                ell[low],
                k,
                m,
                z,
                linear=linear,
                n_chi=n_chi,
                I=I1,
                P_m=P_m,
            )
        )
        if tracer1 is tracer2:
            R2 = R1
        else:
            I2 = self.bias_weighted_I(tracer2, k, m, z)
            R2 = np.asarray(
                self.radial_transfer_R_ell(
                    tracer2,
                    ell[low],
                    k,
                    m,
                    z,
                    linear=linear,
                    n_chi=n_chi,
                    I=I2,
                    P_m=P_m,
                )
            )

        k = np.asarray(k)
        result[low] = (2.0 / np.pi) * trapezoid(
            k[:, None] ** 3 * R1.T * R2.T,
            x=np.log(k),
            axis=0,
        )
        return jnp.asarray(result)


    def cl_2h_periodic(self, tracer1, tracer2, l, m, z, L, n_max=None, s_max=None,
                       s=None, g=None, linear=True, k_max=None, n_k=None, n_chi=None):
        r"""Exact non-Limber periodic-universe 2-halo angular spectrum.

        .. math::

            C_{\ell,L}^{2h} = \frac{4\pi}{L^3}\sum_{s\ge 1} g_s\,
                R_\ell(k_s)\, R'_\ell(k_s),
            \qquad k_s = \frac{2\pi}{L}\sqrt{s}

        with :math:`R_\ell` from :meth:`radial_transfer_R_ell` (PDF eqs. 36–41
        and 105–106). The DC mode :math:`p=0` is excluded. This is **not**
        the coarse-grained Limber cutoff :math:`\Theta(k_\ell-2\pi/L)`.

        The existing :meth:`cl_1h` path is unchanged. :math:`P_m` is
        :meth:`Cosmology.pk` in physical :math:`\\mathrm{Mpc}^3` (same
        convention as :meth:`pk_1h` / :meth:`pk_2h`). At large :math:`L`
        or high :math:`\\ell` this approaches Limber :meth:`cl_2h`,
        provided the lattice extends to :math:`k \\gtrsim k_\\ell`.

        Parameters
        ----------
        tracer1 : Tracer
            First tracer.
        tracer2 : Tracer or None
            Second tracer (``None`` means auto-spectrum).
        l : array-like
            Multipole grid.
        m : array
            Halo-mass grid in physical :math:`M_\odot`.
        z : array
            Redshift grid.
        L : float
            Periodic box side length in physical :math:`\mathrm{Mpc}`.
        n_max : int, optional
            Keep lattice vectors with :math:`|p_i|\le n_{\max}`.
        s_max : int, optional
            Keep shells with :math:`s\le s_{\max}`. Required together with
            ``n_max`` unless ``s`` and ``g`` are supplied.
        s, g : array-like, optional
            Precomputed shells and multiplicities from
            :func:`hmfast.halos.periodic.cubic_lattice_shells`.
        linear : bool, optional
            Forwarded to the matter power spectrum (default linear).
        k_max : float, optional
            If ``s`` is not supplied, drop shells with
            :math:`k_s > k_{\\max}` (sets ``n_max`` if it is omitted).
        n_k : int, optional
            If set and smaller than the number of shells, evaluate
            :math:`I_1` on this many log-\:math:`k` nodes and interpolate
            onto each :math:`k_s` (cheaper for a large box).
        n_chi : int, optional
            Radial samples for :math:`j_\\ell`; see
            :meth:`radial_transfer_R_ell`.

        Returns
        -------
        cl : array
            Periodic 2-halo :math:`C_\ell` with shape :math:`(N_\ell,)`.
        """
        import numpy as np

        tracer2 = tracer1 if tracer2 is None else tracer2
        L = float(L)
        if not np.isfinite(L) or L <= 0.0:
            raise ValueError("L must be a positive finite box side in Mpc.")
        if s is None:
            if k_max is not None:
                n_from_k = n_max_for_kmax(L, k_max)
                n_max = n_from_k if n_max is None else min(int(n_max), n_from_k)
            s, g = cubic_lattice_shells(s_max=s_max, n_max=n_max)
        s_np = np.asarray(s)
        g_np = np.asarray(g)
        k_s = lattice_wavenumbers(L, s_np)
        if k_max is not None:
            keep = k_s <= float(k_max) + 1.0e-15
            s_np, g_np, k_s = s_np[keep], g_np[keep], k_s[keep]

        I1 = self._I1_on_lattice(tracer1, k_s, m, z, n_k=n_k)
        P_m = self._emulator_pk(k_s, z, linear=linear)

        R1 = self.radial_transfer_R_ell(
            tracer1, l, k_s, m, z, linear=linear, n_chi=n_chi, I=I1, P_m=P_m
        )
        if tracer1 is tracer2:
            R2 = R1
        else:
            I2 = self._I1_on_lattice(tracer2, k_s, m, z, n_k=n_k)
            R2 = self.radial_transfer_R_ell(
                tracer2, l, k_s, m, z, linear=linear, n_chi=n_chi, I=I2, P_m=P_m
            )
        g = jnp.asarray(g_np)
        return (4.0 * jnp.pi / L**3) * jnp.sum(g[None, :] * R1 * R2, axis=-1)


    def _I1_on_lattice(self, tracer, k_s, m, z, n_k=None):
        """:math:`I_1(k_s,z)`, optionally interpolated from a log-k grid."""
        import numpy as np

        k_s = np.asarray(k_s, dtype=float)
        if n_k is None or int(n_k) >= k_s.size:
            return np.asarray(self.bias_weighted_I(tracer, k_s, m, z))
        k_grid = np.geomspace(float(np.min(k_s)), float(np.max(k_s)), int(n_k))
        I_grid = np.asarray(self.bias_weighted_I(tracer, k_grid, m, z))
        return interpolate_kz(k_grid, I_grid, k_s)


    def connected_1h_cl_variance(self, tracer1, tracer2, l, m, z, k_damp=0.0):
        r"""Full-sky 1-halo connected contribution to \(\mathrm{Var}(\hat C_\ell)\).

        Reuses :meth:`trispectrum_1h` on the diagonal and applies the PDF
        \(1/4\pi\) factor (eq. 97) so the result shares units with
        \(2C_\ell^2/(2\ell+1)\):

        .. math::

            \mathrm{Var}^{1h}(\hat C_\ell)
                = T^{1h}_{\ell\ell} / (4\pi).

        The same number is used for the usual and periodic universes.
        """
        l = jnp.atleast_1d(l)
        T = self.connected_1h_cl_covariance(tracer1, tracer2, l, m, z, k_damp=k_damp)
        return jnp.diag(T)

    def connected_1h_cl_covariance(self, tracer1, tracer2, l, m, z, k_damp=0.0):
        r"""Full-sky 1-halo connected \(\mathrm{Cov}(\hat C_\ell,\hat C_{\ell'})\).

        .. math::

            \mathrm{Cov}^{1h}_{\ell\ell'} = T^{1h}_{\ell\ell'} / (4\pi)

        using the existing :meth:`trispectrum_1h` (PDF eq. 97).
        """
        l = jnp.atleast_1d(l)
        T = self.trispectrum_1h(tracer1, tracer2, l, l, m, z, k_damp=k_damp)
        return T / (4.0 * jnp.pi)


    def var_cl(self, tracer1, tracer2, l, m, z, k_damp=0.0, linear=True):
        r"""Usual-universe full-sky \(\mathrm{Var}(\hat C_\ell)\).

        .. math::

            \mathrm{Var}(\hat C_\ell^{12})
                = \frac{C_\ell^{11}C_\ell^{22} + (C_\ell^{12})^2}{2\ell+1}
                + \frac{T^{1h}_{\ell\ell}}{4\pi},
            \qquad C_\ell = C_\ell^{1h} + C_\ell^{2h}

        with Limber :meth:`cl_1h` / :meth:`cl_2h` and the existing
        :meth:`trispectrum_1h`.

        Returns
        -------
        result : dict of arrays
            ``cl``, ``cl_1h``, ``cl_2h``, ``var_gaussian``, ``var_1h``,
            ``var_total``, each shape :math:`(N_\ell,)`.
        """
        import numpy as np

        tracer2 = tracer1 if tracer2 is None else tracer2
        l = jnp.atleast_1d(l)
        cl_1h = np.asarray(
            self.cl_1h(tracer1, tracer2, l, m, z, k_damp=k_damp)
        )
        cl_2h = np.asarray(self.cl_2h(tracer1, tracer2, l, m, z, linear=linear))
        cl = cl_1h + cl_2h
        if tracer1 is tracer2:
            var_g = gaussian_cl_variance(cl, l)
        else:
            cl_11 = np.asarray(
                self.cl_1h(tracer1, tracer1, l, m, z, k_damp=k_damp)
                + self.cl_2h(tracer1, tracer1, l, m, z, linear=linear)
            )
            cl_22 = np.asarray(
                self.cl_1h(tracer2, tracer2, l, m, z, k_damp=k_damp)
                + self.cl_2h(tracer2, tracer2, l, m, z, linear=linear)
            )
            var_g = (cl_11 * cl_22 + cl * cl) / (2.0 * np.asarray(l) + 1.0)
        var_c = np.asarray(
            self.connected_1h_cl_variance(tracer1, tracer2, l, m, z, k_damp=k_damp)
        )
        return {
            "cl": cl,
            "cl_1h": cl_1h,
            "cl_2h": cl_2h,
            "var_gaussian": var_g,
            "var_1h": var_c,
            "var_total": var_g + var_c,
        }


    def _periodic_mode_amplitudes(
        self, tracer1, tracer2, l, m, z, L, n_max=None, s_max=None,
        k_max=None, n_k=None, n_chi=None, linear=True, n_max_aniso=None,
    ):
        """Per-mode :math:`R_\\ell(k_p)` and the periodic 2-halo mean.

        The mean :math:`C_{\\ell,L}^{2h}` is always the full-lattice shell sum.
        ``n_max_aniso`` keeps only a low-:math:`n` cube of directions for
        :math:`Q_\\ell`; leftover high-:math:`k` power is returned as
        ``A_iso`` and treated as isotropic in :func:`lattice_Q_ell`.
        """
        import numpy as np

        tracer2 = tracer1 if tracer2 is None else tracer2
        L = float(L)
        if k_max is not None:
            n_from_k = n_max_for_kmax(L, k_max)
            n_max = n_from_k if n_max is None else min(int(n_max), n_from_k)
        if n_max is None and s_max is None:
            if n_max_aniso is None:
                raise ValueError("Provide n_max, s_max, k_max, or n_max_aniso.")
            n_max = int(n_max_aniso)

        s_u, g_u = cubic_lattice_shells(s_max=s_max, n_max=n_max)
        k_u = lattice_wavenumbers(L, s_u)
        if k_max is not None:
            keep = k_u <= float(k_max) + 1.0e-15
            s_u, g_u, k_u = s_u[keep], g_u[keep], k_u[keep]

        I1 = self._I1_on_lattice(tracer1, k_u, m, z, n_k=n_k)
        P_m = self._emulator_pk(k_u, z, linear=linear)
        R1 = np.asarray(
            self.radial_transfer_R_ell(
                tracer1, l, k_u, m, z, linear=linear, n_chi=n_chi, I=I1, P_m=P_m
            )
        )
        if tracer1 is tracer2:
            R2 = R1
        else:
            I2 = self._I1_on_lattice(tracer2, k_u, m, z, n_k=n_k)
            R2 = np.asarray(
                self.radial_transfer_R_ell(
                    tracer2, l, k_u, m, z, linear=linear, n_chi=n_chi, I=I2, P_m=P_m
                )
            )
        cl_2h = (4.0 * np.pi / L**3) * np.sum(
            np.asarray(g_u)[None, :] * R1 * R2, axis=1
        )

        n_full = int(n_max) if n_max is not None else int(np.floor(np.sqrt(s_max)))
        n_ex = n_full if n_max_aniso is None else min(int(n_max_aniso), n_full)
        p = cubic_lattice_vectors(s_max=s_max, n_max=n_ex)
        s_p = np.sum(p * p, axis=1)
        if k_max is not None:
            keep = lattice_wavenumbers(L, s_p) <= float(k_max) + 1.0e-15
            p, s_p = p[keep], s_p[keep]
        inv = np.searchsorted(s_u, s_p)
        R1_mode = R1[:, inv] if p.shape[0] else np.zeros((R1.shape[0], 0))
        R2_mode = R2[:, inv] if p.shape[0] else np.zeros((R2.shape[0], 0))
        A_ex = np.sum(R1_mode * R2_mode, axis=1)
        A_tot = cl_2h * (L**3 / (4.0 * np.pi))
        A_iso = np.clip(A_tot - A_ex, 0.0, None)
        return p, R1_mode, R2_mode, cl_2h, A_iso


    def var_cl_periodic(
        self, tracer1, tracer2, l, m, z, L, n_max=None, s_max=None,
        k_max=None, n_k=None, n_chi=None, linear=True, k_damp=0.0,
        n_max_aniso=None,
    ):
        r"""Periodic-universe full-sky \(\mathrm{Var}(\hat C_\ell)\).

        .. math::

            \mathrm{Var}^G_L(\hat C_\ell) = 2 C_{\ell,L}^2 Q_\ell,
            \qquad
            Q_\ell = \sum_{p,q\neq 0} w_{\ell p} w_{\ell q} P_\ell^2(\mu_{pq}),

        with \(C_{\ell,L}=C_\ell^{1h}+C_{\ell,L}^{2h}\) (existing Limber
        1-halo plus shipped non-Limber periodic 2-halo) and the **same**
        1-halo connected piece as :meth:`var_cl`:

        .. math::

            \mathrm{Var}_L = \mathrm{Var}^G_L + T^{1h}_{\ell\ell}/(4\pi).

        As \(L\to\infty\) or at high \(\ell\), many lattice directions
        contribute and \(Q_\ell\to 1/(2\ell+1)\), so the Gaussian piece
        approaches the usual \(2C_\ell^2/(2\ell+1)\).  ``n_max_aniso``
        evaluates the pair sum on a low-\(n\) cube and treats leftover
        high-\(k\) power as isotropic (required for a large box).

        Returns
        -------
        result : dict of arrays
            ``cl``, ``cl_1h``, ``cl_2h``, ``Q``, ``A_iso``,
            ``var_gaussian``, ``var_1h``, ``var_total``.
        """
        import numpy as np

        tracer2 = tracer1 if tracer2 is None else tracer2
        l = jnp.atleast_1d(l)
        cl_1h = np.asarray(
            self.cl_1h(tracer1, tracer2, l, m, z, k_damp=k_damp)
        )
        p, R1_mode, R2_mode, cl_2h, A_iso = self._periodic_mode_amplitudes(
            tracer1, tracer2, l, m, z, L, n_max=n_max, s_max=s_max,
            k_max=k_max, n_k=n_k, n_chi=n_chi, linear=linear,
            n_max_aniso=n_max_aniso,
        )
        cl = cl_1h + cl_2h
        Q = lattice_Q_ell(l, p, R1_mode * R2_mode, A_iso=A_iso)
        var_g = 2.0 * cl * cl * Q
        var_c = np.asarray(
            self.connected_1h_cl_variance(tracer1, tracer2, l, m, z, k_damp=k_damp)
        )
        return {
            "cl": cl,
            "cl_1h": cl_1h,
            "cl_2h": cl_2h,
            "Q": Q,
            "A_iso": A_iso,
            "var_gaussian": var_g,
            "var_1h": var_c,
            "var_total": var_g + var_c,
        }


    def var_cl_binned(
        self, tracer1, tracer2, l, m, z, ell_edges, k_damp=0.0, linear=True,
    ):
        r"""Usual-universe binned \(\mathrm{Var}(\hat C_b)\) (PDF §7).

        \(\hat C_b=\sum_\ell W_{b\ell}\hat C_\ell\) with uniform
        \(W_{b\ell}=1/N_b\). The Gaussian piece is diagonal in \(\ell\);
        the 1-halo piece uses the full \(T^{1h}_{\ell\ell'}/4\pi\).
        """
        import numpy as np

        tracer2 = tracer1 if tracer2 is None else tracer2
        l = jnp.atleast_1d(l)
        W, ell_eff, n_ell = multipole_bin_weights(l, ell_edges)
        unb = self.var_cl(tracer1, tracer2, l, m, z, k_damp=k_damp, linear=linear)
        cov_g = np.diag(unb["var_gaussian"])
        cov_1h = np.asarray(
            self.connected_1h_cl_covariance(tracer1, tracer2, l, m, z, k_damp=k_damp)
        )
        cov_tot = cov_g + cov_1h
        cl_b = W @ unb["cl"]
        var_g = np.einsum("bi,ij,bj->b", W, cov_g, W)
        var_c = np.einsum("bi,ij,bj->b", W, cov_1h, W)
        ell_f = np.asarray(l, dtype=float)
        f = ell_f * (ell_f + 1.0) / (2.0 * np.pi)
        WD = W * f
        return {
            "ell_eff": ell_eff,
            "n_ell": n_ell,
            "weights": W,
            "cl": cl_b,
            "cl_1h": W @ unb["cl_1h"],
            "cl_2h": W @ unb["cl_2h"],
            "var_gaussian": var_g,
            "var_1h": var_c,
            "var_total": var_g + var_c,
            "cov_gaussian": W @ cov_g @ W.T,
            "cov_1h": W @ cov_1h @ W.T,
            "cov_total": W @ cov_tot @ W.T,
            "dell": WD @ unb["cl"],
            "var_dell_gaussian": np.einsum("bi,ij,bj->b", WD, cov_g, WD),
            "var_dell_1h": np.einsum("bi,ij,bj->b", WD, cov_1h, WD),
            "var_dell": np.einsum("bi,ij,bj->b", WD, cov_tot, WD),
        }


    def var_cl_periodic_binned(
        self, tracer1, tracer2, l, m, z, L, ell_edges, n_max=None, s_max=None,
        k_max=None, n_k=None, n_chi=None, linear=True, k_damp=0.0,
        n_max_aniso=None,
    ):
        r"""Periodic-universe binned \(\mathrm{Var}(\hat C_b)\) (PDF §7).

        Projects the lattice Gaussian covariance (eq. 107), rescaled to the
        total \(C_{\ell,L}=C_\ell^{1h}+C_{\ell,L}^{2h}\) so a one-multipole
        bin recovers :meth:`var_cl_periodic`, plus the same 1-halo
        \(T^{1h}/4\pi\) as :meth:`var_cl_binned`.
        """
        import numpy as np

        tracer2 = tracer1 if tracer2 is None else tracer2
        l = jnp.atleast_1d(l)
        W, ell_eff, n_ell = multipole_bin_weights(l, ell_edges)
        cl_1h = np.asarray(
            self.cl_1h(tracer1, tracer2, l, m, z, k_damp=k_damp)
        )
        p, R1_mode, R2_mode, cl_2h, A_iso = self._periodic_mode_amplitudes(
            tracer1, tracer2, l, m, z, L, n_max=n_max, s_max=s_max,
            k_max=k_max, n_k=n_k, n_chi=n_chi, linear=linear,
            n_max_aniso=n_max_aniso,
        )
        cl = cl_1h + cl_2h
        cov_2h = lattice_gaussian_cl_cov(
            l, p, R1_mode, float(L), R2=R2_mode
        )
        # Isotropic remainder of Q contributes only on the diagonal.
        if np.any(np.asarray(A_iso) > 0.0):
            Q = lattice_Q_ell(l, p, R1_mode * R2_mode, A_iso=A_iso)
            var_diag = 2.0 * cl_2h * cl_2h * Q
            np.fill_diagonal(cov_2h, var_diag)
        scale = np.divide(cl, cl_2h, out=np.ones_like(cl), where=cl_2h > 0.0)
        cov_g = cov_2h * scale[:, None] * scale[None, :]
        cov_1h = np.asarray(
            self.connected_1h_cl_covariance(tracer1, tracer2, l, m, z, k_damp=k_damp)
        )
        cov_tot = cov_g + cov_1h
        var_g = np.einsum("bi,ij,bj->b", W, cov_g, W)
        var_c = np.einsum("bi,ij,bj->b", W, cov_1h, W)
        ell_f = np.asarray(l, dtype=float)
        f = ell_f * (ell_f + 1.0) / (2.0 * np.pi)
        WD = W * f
        return {
            "ell_eff": ell_eff,
            "n_ell": n_ell,
            "weights": W,
            "cl": W @ cl,
            "cl_1h": W @ cl_1h,
            "cl_2h": W @ cl_2h,
            "var_gaussian": var_g,
            "var_1h": var_c,
            "var_total": var_g + var_c,
            "cov_gaussian": W @ cov_g @ W.T,
            "cov_1h": W @ cov_1h @ W.T,
            "cov_total": W @ cov_tot @ W.T,
            "dell": WD @ cl,
            "var_dell_gaussian": np.einsum("bi,ij,bj->b", WD, cov_g, WD),
            "var_dell_1h": np.einsum("bi,ij,bj->b", WD, cov_1h, WD),
            "var_dell": np.einsum("bi,ij,bj->b", WD, cov_tot, WD),
        }


jax.tree_util.register_pytree_node(
    HaloModel,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: HaloModel._tree_unflatten(aux_data, children)
)
