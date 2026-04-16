import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from functools import partial
from abc import ABC, abstractmethod
from mcfit import TophatVar


class HaloMass(ABC):
    """
    Abstract base class for all halo mass functions.
    All subclasses must implement the f_sigma method.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_hmf_grid(self, halo_model):
        """
        Compute :math:`\sigma(R, z)` and the halo mass function grid for use in interpolation.

        Returns
        -------
        ln_x : array_like
            :math:`\ln(1+z)` grid.
        ln_M : array_like
            :math:`\ln M` grid.
        dn_dlnM_grid : array_like
            :math:`dn/d\ln M` grid.
        sigma_grid : array_like
            :math:`\sigma(R, z)` values.
        """
        
        z_grid = halo_model.cosmology._z_grid_pk()
        cparams = halo_model.cosmology.get_all_cosmo_params()
        h = cparams["h"]
    
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
        M_grid = 4.0 * jnp.pi / 3.0 * Omega0_cb * rho_crit_0 * (R_grid ** 3) * h ** 3
    
        # Overdensity threshold
        delta_numeric = halo_model._delta_numeric(z_grid)
        delta_mean = halo_model._convert_reference(z_grid, delta_numeric, from_ref=halo_model.mass_definition.reference, to_ref='mean') 
    
        # Halo mass function grid, shape: (n_z, n_R)
        hmf_grid = halo_model.mass_model.f_sigma(sigma_grid, z_grid, delta_mean)
    
        # Compute d n / d ln(M)
        dlnnu_dlnR_grid = -dvar_grid * R_grid / jnp.exp(2. * ln_sigma_grid)
        dn_dlnM_grid = dlnnu_dlnR_grid * hmf_grid / (4.0 * jnp.pi * R_grid**3 * h**3)
    
        # Grids for interpolation
        ln_x = jnp.log(1. + z_grid)
        ln_M = jnp.log(M_grid)
    
        return ln_x, ln_M, dn_dlnM_grid



    @abstractmethod
    def f_sigma(self, sigmas, z, delta_mean):
        """
        Compute :math:`f_\\sigma`.
        """
        pass


class T08HaloMass(HaloMass):
    """
    Halo mass function from `Tinker et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJ...688..709T/abstract>`_.

    Valid for spherical overdensity masses (e.g., 200m, 500c).
    """

    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def f_sigma(self, sigmas, z, delta_mean):
        """
        Computes the differential mass function :math:`dn/d\\ln\\sigma` for given variance :math:`\\sigma(R)` and redshift.
    
        The relation is:
    
            .. math::
    
                f(\\sigma) = 0.5 A \\left[ \\left( \\frac{\\sigma}{b} \\right)^{-a} + 1 \\right] \\exp\\left( -\\frac{c}{\\sigma^2} \\right)
    
            where :math:`A`, :math:`a`, :math:`b`, and :math:`c` are parameters interpolated as a function of overdensity and redshift.
    
        Parameters
        ----------
        sigmas : jnp.ndarray
            Variance of the linear density field :math:`\\sigma(R, z)`, shape (n_R, n_z) or (n_R,)
        z : float or jnp.ndarray
            Redshift(s) corresponding to sigmas
        delta_mean : float or jnp.ndarray
            Halo overdensity :math:`\\Delta` (e.g., 200, 500, 1600). Can be scalar or shape (n_z,)
    
        Returns
        -------
        f_sigma : jnp.ndarray
            Halo mass function values, shape matching sigmas
            (:math:`dn/d\\ln\\sigma`) in units consistent with Tinker et al. (2008)
        """
        
        # Convert delta_mean to log scale
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
        f_sigma = 0.5 * Ap[:, None] * (jnp.power(sigmas / b[:, None], -a[:, None]) + 1) * jnp.exp(-c[:, None] / sigmas**2)
        return f_sigma


    @partial(jax.jit, static_argnums=(0,))
    def halo_mass_function(self, halo_model, m, z) -> jnp.ndarray:
        """
        Compute the halo mass function :math:`\frac{dn}{d\ln M}` for arbitrary mass and redshift arrays.

        Parameters
        ----------
        m : array-like
            Halo mass grid.
        z : array-like
            Redshift grid.

        Returns
        -------
        dndlnM : array-like
            Halo mass function values, shape (len(m), len(z)).
        """
       
        
        ln_x_grid, ln_M_grid, dn_dlnM_grid = self._compute_hmf_grid(halo_model)

        # Create the interpolator, the meshgrid, and then stack the points
        _hmf_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_M_grid), dn_dlnM_grid)
        mm, zz = jnp.meshgrid(jnp.atleast_1d(m), jnp.atleast_1d(z), indexing='ij')
        pts = jnp.stack([jnp.log(1. + zz), jnp.log(mm)], axis=-1)
        
        return _hmf_interp(pts)




class T10HaloMass(HaloMass):
    """
    Halo mass function from `Tinker et al. (2010) <https://ui.adsabs.harvard.edu/abs/2010ApJ...724..878T/abstract>`_.

    Valid for spherical overdensity masses (e.g., 200m, 500c).
    """

    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def f_sigma(self, sigmas, z, delta_mean):
        """
        Computes the mass function :math:`f(\\nu, z)` as a function of peak height :math:`\\nu` and redshift.
    
        The relation is:
    
            .. math::
    
                f(\\nu, z) = 0.5 \\alpha (1 + (\\beta^2 \\nu)^{-\\phi}) \\nu^{\\eta} \\exp\\left(-\\frac{\\gamma \\nu}{2}\\right) \\sqrt{\\nu}
    
            where the parameters :math:`\\alpha`, :math:`\\beta`, :math:`\\gamma`, :math:`\\eta`, :math:`\\phi` are redshift-dependent.
    
        Parameters
        ----------
        sigmas : jnp.ndarray
            Variance of the linear density field :math:`\\sigma(R, z)`, shape (n_R, n_z) or (n_R,)
        z : float or jnp.ndarray
            Redshift(s) corresponding to sigmas
        delta_mean : float or jnp.ndarray
            Halo overdensity :math:`\\Delta` (e.g., 200, 500, 1600). Can be scalar or shape (n_z,)
    
        Returns
        -------
        f_nu : jnp.ndarray
            Halo mass function values, shape matching sigmas
            (:math:`f(\\nu, z)`)
        """
        delta_mean = jnp.log10(delta_mean)
        delta_c = 1.686 
        log_nu = 2.0 * jnp.log(delta_c) - 2.0 * jnp.log(sigmas)
        nu = jnp.exp(log_nu)
        
        # Base parameters
        alpha0, beta0, gamma0, eta0, phi0 = 0.368, 0.589, 0.864, -0.243, -0.729
        # Redshift exponents
        alpha_z, beta_z, gamma_z, eta_z, phi_z = 0.0, 0.2, -0.01, 0.27, -0.08

        # Compute z-dependent parameters
        alpha = alpha0 * (1 + z)**alpha_z
        beta  = beta0  * (1 + z)**beta_z
        gamma = gamma0 * (1 + z)**gamma_z
        eta   = eta0   * (1 + z)**eta_z
        phi   = phi0   * (1 + z)**phi_z

        # Reshape for broadcasting
        alpha = alpha[:, None]
        beta  = beta[:, None]
        gamma = gamma[:, None]
        eta   = eta[:, None]
        phi   = phi[:, None]

        beta_term = (beta ** 2 * nu) ** (-phi)  # (beta^2 * nu)^(-phi)
        eta_term  = nu ** eta
        exp_term  = jnp.exp(-gamma * nu / 2)


        f_nu = 0.5 * alpha * (1 + beta_term) * eta_term * exp_term * jnp.sqrt(nu)
        return f_nu

    @partial(jax.jit, static_argnums=(0,))
    def halo_mass_function(self, halo_model, m, z) -> jnp.ndarray:
        """
        Compute the halo mass function :math:`\frac{dn}{d\ln M}` for arbitrary mass and redshift arrays.

        Parameters
        ----------
        m : array-like
            Halo mass grid.
        z : array-like
            Redshift grid.

        Returns
        -------
        dndlnM : array-like
            Halo mass function values, shape (len(m), len(z)).
        """
       
        
        ln_x_grid, ln_M_grid, dn_dlnM_grid = self._compute_hmf_grid(halo_model)

        # Create the interpolator, the meshgrid, and then stack the points
        _hmf_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_M_grid), dn_dlnM_grid)
        mm, zz = jnp.meshgrid(jnp.atleast_1d(m), jnp.atleast_1d(z), indexing='ij')
        pts = jnp.stack([jnp.log(1. + zz), jnp.log(mm)], axis=-1)
        
        return _hmf_interp(pts)





class SubHaloMass(ABC):
    """
    Abstract base class for all subhalo mass functions.
    All subclasses must implement the dndlnmu method.
    """
    @abstractmethod
    def dndlnmu(self, halo_model, m, z):
        """
        Compute :math:`d\\ln\\mu`
        """
        pass

class TW10SubHaloMass(SubHaloMass):
    """
    Subhalo mass function from `Tinker & Wetzel (2010) <https://ui.adsabs.harvard.edu/abs/2010ApJ...719...88T/abstract>`_.

    Valid for all host halo masses.
    """
    def __init__(self):
        pass
    
    def dndlnmu(self, M_host, M_sub):
        """
        Computes the number of subhalos per host per :math:`d\\ln\\mu`, where :math:`\\mu = M_{\\rm sub} / M_{\\rm host}`.
    
        The relation is:
    
            .. math::
    
                \\frac{dN}{d\\ln\\mu} = 0.30 \\mu^{-0.7} \\exp(-9.9 \\mu^{2.5})
    
        Parameters
        ----------
        M_host : float or array_like
            Host halo mass [Msun]
        M_sub : float or array_like
            Subhalo mass [Msun]
    
        Returns
        -------
        dN_dlnmu : float or array_like
            Number of subhalos per host per :math:`d\\ln\\mu`
        """
        mu = M_sub / M_host
        dN_dlnmu = 0.30 * mu ** (-0.7) * jnp.exp(-9.9 * mu ** 2.5)
        return dN_dlnmu



class JvdB14SubHaloMass(SubHaloMass):
    """
    Subhalo mass function from `Jiang & van den Bosch (2014) <https://ui.adsabs.harvard.edu/abs/2014MNRAS.440..193J/abstract>`_.

    Valid for all host halo masses.
    """
    def __init__(self):
        # Jiang & van den Bosch (2014) parameters
        self.gamma1 = 0.13
        self.alpha1 = -0.83
        self.gamma2 = 1.33
        self.alpha2 = -0.02
        self.beta = 5.67
        self.zeta = 1.19

    def dndlnmu(self, M_host, M_sub):
        """
        Computes the number of subhalos per host per :math:`d\\ln\\mu`, where :math:`\\mu = M_{\\rm sub} / M_{\\rm host}`.
    
        The relation is:
    
            .. math::
    
                \\frac{dN}{d\\ln\\mu} = \\gamma_1 \\mu^{\\alpha_1} + \\gamma_2 \\mu^{\\alpha_2} \\exp(-\\beta \\mu^{\\zeta})
    
            where the parameters are taken from Eq. 21 of the paper.
    
        Parameters
        ----------
        M_host : float or array_like
            Host halo mass [Msun]
        M_sub : float or array_like
            Subhalo mass [Msun]
    
        Returns
        -------
        dN_dlnmu : float or array_like
            Number of subhalos per host per :math:`d\\ln\\mu`
        """
        
        mu = M_sub / M_host
        dN_dlnmu = (self.gamma1 * mu**self.alpha1 + self.gamma2 * mu**self.alpha2) * \
                    jnp.exp(-self.beta * mu**self.zeta)
        return dN_dlnmu



