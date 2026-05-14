"""
Core halo model implementation using JAX for differentiability.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Dict, Any, Optional, Callable
from functools import partial
from mcfit import TophatVar

from hmfast.halos.massfunc import T08HaloMass, TW10SubHaloMass
from hmfast.halos.bias import T10HaloBias
from hmfast.halos.concentration import D08Concentration, B13Concentration
from hmfast.halos.mass_definition import MassDefinition
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
        h = self.cosmology.H0 / 100.0
        self._tophat_instance = partial(TophatVar(dummy_k / h, lowring=True, backend='jax'), extrap=True)


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
        h = self.cosmology.H0 / 100.0
        comov_vol = self.cosmology.comoving_volume_element(z) * h**3

        # Integrate over redshift
        integrand = P_1h_grid * (comov_vol[:, None] * kernel1[:, None] * kernel2[:, None])
        
        return jnp.trapezoid(integrand, x=z, axis=0)
    


    @jax.jit
    def pk_2h(self, tracer1, tracer2, k, m, z):
        """
        Compute the 2-halo contribution to the 3D power spectrum in
        physical units.

        .. math::
        
            P_{2h}(k, z) = P_{\\mathrm{lin}}(k, z) \\, I_1(k, z) \\, I_2(k, z)
        
        with
        
        .. math::
        
            I_i(k, z) = \\int d\\ln M \\, \\frac{dn}{d\\ln M}(M, z) \\, b(M, z) \\, u_i(k \\mid M, z),
        
        where :math:`u_i(k \\mid M, z)` is the Fourier-space tracer profile,
        :math:`dn/d\\ln M` is the halo mass function, and :math:`b(M, z)` is the
        linear halo bias.

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

        Returns
        -------
        pk_2h : array
            2-halo power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`.
        """
        
        cparams = self.cosmology._cosmo_params()
        h, k, m, z = cparams["h"], jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
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
        
        # Reconstruct the legacy linear spectrum normalization used by the
        # current halo-model projection chain so outputs remain unchanged.
        P_lin = jax.vmap(lambda zi: jnp.interp(h * k, *self.cosmology.pk(zi, linear=True)))(z).T * h**6
        
        return P_lin * I1 * I2


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

        def trisp_slice(zi):
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

            # Mass integral: weighted by dn/dlnM
            weight = dndlnm[:, 0] * w  # (Nm,)
            integrand = usq_a[:, None, :] * usq_b[None, :, :] * weight[None, None, :]
            T_z = jnp.sum(integrand, axis=-1)  # (Nl1, Nl2)
            return T_z

        T_grid = jax.vmap(trisp_slice)(z)  # (Nz, Nl1, Nl2)

        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        h = self.cosmology.H0 / 100.0
        comov_vol = self.cosmology.comoving_volume_element(z) * h**3

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
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        dndlnm = self.halo_mass_function.halo_mass_function(self, m, z)  # (Nm, Nz)
        total_weights = dndlnm * w[:, None]  # (Nm, Nz)

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
            weight_i = total_weights[:, i] * mask_mz[:, i]  # (Nm,)
            return jnp.sum(u_sq * weight_i[None, :], axis=-1)  # (Nl,)

        pk_grid = jax.vmap(slice_z)(jnp.arange(z.shape[0]))  # (Nz, Nl)

        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        h = self.cosmology.H0 / 100.0
        comov_vol = self.cosmology.comoving_volume_element(z) * h**3
        weight_z = comov_vol * kernel1 * kernel2
        integrand = pk_grid * weight_z[:, None]
        return jnp.trapezoid(integrand, x=z, axis=0)


    @jax.jit
    def cl_2h(self, tracer1, tracer2, l, m, z):
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
            return self.pk_2h(tracer1, tracer2, k=ki, m=m, z=jnp.atleast_1d(zi)).flatten()
    
        # Map over redshift to get P(k=l/chi, z)
        P_2h_grid = jax.vmap(get_pk_slice)(z) 
        
        # Get individual kernels
        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        
        h = self.cosmology.H0 / 100.0
        comov_vol = self.cosmology.comoving_volume_element(z) * h**3
    
        # Limber Integral: C_l = int dz P(k,z) * [W1 * W2 * dV/dz]
        integrand = P_2h_grid * (comov_vol[:, None] * kernel1[:, None] * kernel2[:, None])
        
        return jnp.trapezoid(integrand, x=z, axis=0)


jax.tree_util.register_pytree_node(
    HaloModel,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: HaloModel._tree_unflatten(aux_data, children)
)
