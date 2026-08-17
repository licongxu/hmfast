import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
from functools import partial
from abc import ABC, abstractmethod
from mcfit import TophatVar


class HaloMass(ABC):
    """
    Abstract base class for halo mass function models.

    Subclasses provide the dimensionless fitting function :math:`f(\\sigma)`
    entering :math:`dn / d\\ln M`, together with a public evaluator for the
    halo mass function on a mass-redshift grid.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _compute_hmf_grid(self, halo_model):
        """
        Compute the internal interpolation grid for the halo mass function.

        The interpolation mass grid returned here is in physical
        :math:`M_\\odot`.

        Returns
        -------
        ln_x : array_like
            :math:`\\ln(1+z)` grid.
        ln_M : array_like
            :math:`\\ln M` grid.
        dn_dlnM_grid : array_like
            Halo mass function grid :math:`dn/d\\ln M` in comoving
            :math:`\\mathrm{Mpc}^{-3}`.
        """
        
        z_grid = halo_model.cosmology._z_grid_pk()
        cparams = halo_model.cosmology._cosmo_params()

        # Power spectra for all redshifts, shape: (n_k, n_z)
        pk_grid = jax.vmap(lambda zp: halo_model.cosmology.pk(zp, linear=True)[1].flatten())(z_grid).T
    
        # Compute σ²(R, z) and dσ²/dR using TophatVar
        R_grid, var = jax.vmap(halo_model._tophat_instance, in_axes=1, out_axes=(0, 0))(pk_grid)
        R_grid = R_grid[0].flatten()  # shape: (n_R,)
    
        # Compute dσ²/dR for each z, output shape: (n_z, n_R)
        dvar_grid = jax.vmap(lambda v: jnp.gradient(v, R_grid), in_axes=0)(var)
    
        # Compute σ(R, z)
        ln_sigma_grid = 0.5 * jnp.log(var)
        sigma_grid = jnp.exp(ln_sigma_grid)
    
        # Mass grid, shape: (n_R,)
        rho_crit_0 = cparams["Rho_crit_0"]
        Omega0_cb = cparams['Omega0_cb']
        M_grid = 4.0 * jnp.pi / 3.0 * Omega0_cb * rho_crit_0 * (R_grid ** 3)
    
    
        # Halo mass function grid, shape: (n_z, n_R)
        hmf_grid = halo_model.halo_mass_function._f_sigma(halo_model, sigma_grid, z_grid)
    
        # Compute d n / d ln(M). Standard Tinker08 formula:
        #   dn/dlnM = -f(σ) × R × dσ²/dR / (8π R³ σ²)
        # gives dn/dlnM in 1/Mpc³ (with R in Mpc, ρ in M_sun/Mpc³, M in M_sun).
        # The factor of 2 vs 8π is absorbed in the 0.5 prefactor in _f_sigma.
        # Previously divided by h³ which then cancelled the ×h³ on the volume
        # element in cl_1h_masked; removing both for direct 1/Mpc³ × Mpc³
        # arithmetic to avoid spurious sub-percent floating-point cross-talk.
        dlnnu_dlnR_grid = -dvar_grid * R_grid / jnp.exp(2. * ln_sigma_grid)
        dn_dlnM_grid = dlnnu_dlnR_grid * hmf_grid / (4.0 * jnp.pi * R_grid**3)
    
        # Grids for interpolation
        ln_x = jnp.log(1. + z_grid)
        ln_M = jnp.log(M_grid)
    
        return ln_x, ln_M, dn_dlnM_grid



    @abstractmethod
    def _f_sigma(self, halo_model, sigma, z):
        """
        Evaluate the internal dimensionless fitting function entering the halo
        mass function.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model instance supplying the cosmology and mass definition
            used to evaluate the fitting function.
        sigma : array-like
            Root-mean-square linear density fluctuation
            :math:`\\sigma(R, z)`.
        z : float or array-like, optional
            Redshift(s), when required by the specific mass-function model.

        Returns
        -------
        array-like
            Values of the internal fitting function evaluated at the requested
            inputs.
        """
        pass


class T08HaloMass(HaloMass):
    """
    Halo mass function from `Tinker et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJ...688..709T/abstract>`_.

    Calibrated for spherical-overdensity halo masses. 
    In this implementation, the fitting coefficients are interpolated over the
    tabulated overdensity grid spanning :math:`\\Delta_\\mathrm{m} = 200`
    to :math:`3200`.
    """

    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _f_sigma(self, halo_model, sigma, z):
        """
        Evaluate the internal Tinker et al. (2008) fitting function.
    
        Parameters
        ----------
        halo_model : HaloModel
            Halo model instance supplying the cosmology and mass definition
            used to evaluate the fitting function.
        sigma : jnp.ndarray
            Root-mean-square linear density fluctuation
            :math:`\\sigma(R, z)`.
        z : float or jnp.ndarray
            Redshift(s) corresponding to ``sigma``.
        
    
        Returns
        -------
        f_sigma : jnp.ndarray
            Values of the internal fitting function with shape matching
            ``sigma``.
        """
        
        # Overdensity threshold converted to log scale
        delta_numeric = halo_model.mass_definition._delta_numeric(halo_model.cosmology, z)
        delta_mean = halo_model.mass_definition._convert_reference(
            halo_model.cosmology,
            z,
            delta_numeric,
            from_ref=halo_model.mass_definition.reference,
            to_ref='mean',
        )
        delta_mean = jnp.log10(delta_mean)
        
        # Define parameters as JAX arrays
        delta_mean_tab = jnp.log10(jnp.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]))
        A_tab = jnp.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
        aa_tab = jnp.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
        b_tab = jnp.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
        c_tab = jnp.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
    
        # Linear interpolation using jnp.interp
        Ap = jnp.interp(delta_mean, delta_mean_tab, A_tab) * (1 + z) ** -0.14
        a = jnp.interp(delta_mean, delta_mean_tab, aa_tab) * (1 + z) ** -0.06
        b = jnp.interp(delta_mean, delta_mean_tab, b_tab) * (1 + z) ** -jnp.power(10, -jnp.power(0.75 / jnp.log10(jnp.power(10, delta_mean) / 75), 1.2))
        c = jnp.interp(delta_mean, delta_mean_tab, c_tab)
        
        # Calculate final result f(σ)
        f_sigma = 0.5 * Ap[:, None] * (jnp.power(sigma / b[:, None], -a[:, None]) + 1) * jnp.exp(-c[:, None] / sigma**2)
        return f_sigma


    @partial(jax.jit, static_argnums=(0,))
    def halo_mass_function(self, halo_model, m, z) -> jnp.ndarray:
        """
        Compute the halo mass function :math:`dn/d\\ln M`.
    
        The halo mass function gives the comoving number density of halos per logarithmic mass interval:
    
        .. math::
    
            \\frac{dn}{d\\ln M} = f(\\sigma) \\frac{\\rho_{m,0}}{M} \\left| \\frac{d\\ln \\sigma^{-1}}{d\\ln M} \\right|

        In this model,

        .. math::

            f(\\sigma) = 0.5 A \\left[\\left(\\frac{\\sigma}{b}\\right)^{-a} + 1\\right]
            \\exp\\left(-\\frac{c}{\\sigma^2}\\right),
    
        where :math:`f(\\sigma)` is the Tinker et al. (2008) fitting function,
        calibrated over a tabulated overdensity grid spanning
        :math:`\\Delta_\\mathrm{m} = 200` to :math:`3200`,
        :math:`A`, :math:`a`, :math:`b`, and :math:`c` are redshift-dependent
        fitting parameters, and :math:`\\sigma(M)` is the variance of the
        density field smoothed on the mass scale :math:`M`.
    
        Parameters
        ----------
        halo_model : HaloModel
            Halo model instance supplying the cosmology and mass definition
            used to evaluate the fitting function.
        m : array-like
            Halo mass grid in physical :math:`M_\\odot`.
        z : array-like
            Redshift grid.
    
        Returns
        -------
        dndlnM : array-like
            Halo mass function values :math:`dn/d\\ln M` in comoving
            :math:`\\mathrm{Mpc}^{-3}`, with shape :math:`(N_m, N_z)`.
        """
       
        
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)

        ln_x_grid, ln_M_grid, dn_dlnM_grid = self._compute_hmf_grid(halo_model)

        # Interpolate in LOG-LOG space: dn/dlnM varies over many orders of
        # magnitude (e.g. 1e-30 at high M, z), so linear interpolation of the
        # raw value introduces ~1% bias at intermediate (M, z). Interpolating
        # log(dn/dlnM) on the same (ln(1+z), ln M) grid and exp-ing back
        # matches tszpower (massfuncs.py:272-274 and 290-298).
        log_dn_grid = jnp.log(jnp.clip(dn_dlnM_grid, 1e-300, None))
        _hmf_interp = jscipy.interpolate.RegularGridInterpolator(
            (ln_x_grid, ln_M_grid), log_dn_grid)
        mm, zz = jnp.meshgrid(m, z, indexing='ij')
        pts = jnp.stack([jnp.log(1. + zz), jnp.log(mm)], axis=-1)

        return jnp.exp(_hmf_interp(pts))






class ST99HaloMass(HaloMass):
    """
    Halo mass function from `Sheth & Tormen (1999)
    <https://ui.adsabs.harvard.edu/abs/1999MNRAS.308..119S/abstract>`_
    (sometimes referred to as ST98 from the arXiv date).

    Implements equation (10) of that paper with the standard parameters
    :math:`A = 0.3222`, :math:`a = 0.707`, :math:`p = 0.3`, and
    :math:`\\delta_c = 1.686`. Unlike :class:`T08HaloMass`, this multiplicity
    function does not depend on the spherical-overdensity mass definition.

    Notes
    -----
    The internal :math:`f(\\sigma)` returned by :meth:`_f_sigma` includes the
    conventional factor of :math:`1/2` used throughout hmfast so that
    :meth:`HaloMass._compute_hmf_grid` produces
    :math:`dn/d\\ln M` in comoving :math:`\\mathrm{Mpc}^{-3}`.
    """

    def __init__(self, A=0.3222, a=0.707, p=0.3, delta_c=1.686):
        """
        Parameters
        ----------
        A : float, optional
            Overall amplitude. Default ``0.3222`` (ST99 normalisation).
        a : float, optional
            Barrier rescaling (often denoted :math:`q`). Default ``0.707``.
        p : float, optional
            Low-mass slope. Default ``0.3``.
        delta_c : float, optional
            Spherical-collapse barrier. Default ``1.686``.
        """
        self.A = float(A)
        self.a = float(a)
        self.p = float(p)
        self.delta_c = float(delta_c)

    @partial(jax.jit, static_argnums=(0,))
    def _f_sigma(self, halo_model, sigma, z):
        """
        Evaluate the internal Sheth–Tormen (1999) fitting function.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model instance (cosmology unused beyond the shared grid).
        sigma : jnp.ndarray
            Root-mean-square linear density fluctuation :math:`\\sigma(R, z)`,
            shape ``(N_z, N_R)``.
        z : float or jnp.ndarray
            Redshift grid corresponding to ``sigma`` (unused; ST99 has no
            explicit redshift evolution beyond :math:`\\sigma(z)`).

        Returns
        -------
        f_sigma : jnp.ndarray
            Internal fitting function with shape matching ``sigma``.
        """
        # Standard f(σ) (colossus / ST99 eq. 10), times 1/2 for hmfast grid
        # convention (same 0.5 prefactor as T08/T10).
        nu_p = self.a * (self.delta_c / sigma) ** 2
        f_std = (
            self.A
            * jnp.sqrt(nu_p * 2.0 / jnp.pi)
            * jnp.exp(-0.5 * nu_p)
            * (1.0 + nu_p ** (-self.p))
        )
        return 0.5 * f_std

    @partial(jax.jit, static_argnums=(0,))
    def halo_mass_function(self, halo_model, m, z) -> jnp.ndarray:
        """
        Compute the halo mass function :math:`dn/d\\ln M`.

        .. math::

            \\frac{dn}{d\\ln M} = f(\\sigma)\\,\\frac{\\rho_{m,0}}{M}
            \\left|\\frac{d\\ln\\sigma^{-1}}{d\\ln M}\\right|

        with the Sheth–Tormen (1999) multiplicity

        .. math::

            f(\\sigma) = A\\sqrt{\\frac{2a}{\\pi}}\\,\\frac{\\delta_c}{\\sigma}
            \\left[1 + \\left(\\frac{a\\delta_c^2}{\\sigma^2}\\right)^{-p}\\right]
            \\exp\\left(-\\frac{a\\delta_c^2}{2\\sigma^2}\\right),

        or equivalently
        :math:`f = A\\sqrt{2\\nu'/\\pi}\\,(1+\\nu'^{-p})\\,e^{-\\nu'/2}` with
        :math:`\\nu' = a\\delta_c^2/\\sigma^2`.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model instance supplying the cosmology.
        m : array-like
            Halo mass grid in physical :math:`M_\\odot`.
        z : array-like
            Redshift grid.

        Returns
        -------
        dndlnM : array-like
            Halo mass function values :math:`dn/d\\ln M` in comoving
            :math:`\\mathrm{Mpc}^{-3}`, with shape :math:`(N_m, N_z)`.
        """
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)

        ln_x_grid, ln_M_grid, dn_dlnM_grid = self._compute_hmf_grid(halo_model)

        log_dn_grid = jnp.log(jnp.clip(dn_dlnM_grid, 1e-300, None))
        _hmf_interp = jscipy.interpolate.RegularGridInterpolator(
            (ln_x_grid, ln_M_grid), log_dn_grid
        )
        mm, zz = jnp.meshgrid(m, z, indexing="ij")
        pts = jnp.stack([jnp.log(1.0 + zz), jnp.log(mm)], axis=-1)

        return jnp.exp(_hmf_interp(pts))




class SubHaloMass(ABC):
    """
    Abstract base class for subhalo mass function models.
    """
    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def dndlnmu(self, halo_model, m_host, m_sub):
        """
        Compute the subhalo abundance per logarithmic mass ratio.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing any cosmology- or mass-definition-dependent context.
        m_host : float or array_like
            Host halo mass in physical :math:`M_\\odot`.
        m_sub : float or array_like
            Subhalo mass in physical :math:`M_\\odot`.

        Returns
        -------
        array-like
            Dimensionless subhalo abundance :math:`dN/d\\ln \\mu`, i.e.
            number of subhalos per host per logarithmic mass ratio.
        """
        pass

class TW10SubHaloMass(SubHaloMass):
    """
    Subhalo mass function from `Tinker & Wetzel (2010) <https://ui.adsabs.harvard.edu/abs/2010ApJ...719...88T/abstract>`_.

    Valid for all host halo masses.
    """
    def __init__(self):
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def dndlnmu(self, halo_model, m_host, m_sub):
        """
        Compute the Tinker and Wetzel (2010) subhalo mass function.

        .. math::

            \\frac{dN}{d\\ln \\mu} = 0.30 \\mu^{-0.7} \\exp(-9.9 \\mu^{2.5})

        where :math:`\\mu = M_{\\rm sub} / M_{\\rm host}`.
    
        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing any cosmology- or mass-definition-dependent context.
        m_host : float or array_like
            Host halo mass in physical :math:`M_\\odot`.
        m_sub : float or array_like
            Subhalo mass in physical :math:`M_\\odot`.
    
        Returns
        -------
        dN_dlnmu : float or array_like
                Dimensionless number of subhalos per host per :math:`d\\ln \\mu`.
                The output is dimensionless.
        """
        mu = m_sub / m_host
        dN_dlnmu = 0.30 * mu ** (-0.7) * jnp.exp(-9.9 * mu ** 2.5)
        return dN_dlnmu




