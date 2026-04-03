import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import mcfit
import functools
from jax.scipy.special import sici, erf
from jax.tree_util import register_pytree_node_class

from hmfast.download import get_default_data_path
from hmfast.defaults import merge_with_defaults
from hmfast.utils import lambertw, Const
from hmfast.halo_model.mass_definition import MassDefinition


class HankelTransform:
    """
    Reusable Hankel transform wrapper for JAX-based computation.
    """
    def __init__(self, x, nu=0.5):
        
        self._hankel = mcfit.Hankel(x, nu=nu, lowring=True, backend='jax')
        self._hankel_jit = jax.jit(functools.partial(self._hankel, extrap=False))

    def transform(self, f_theta):
        """
        Perform the Hankel transform on a profile sampled on self._x_grid
        """
        k, y_k = self._hankel_jit(f_theta)
        return k, y_k


class HaloProfile:

    @property
    def has_central_contribution(self):
        """ 
        Indicates whether the profile has a contribution from central terms, such as:
        
            - HOD, which has profile = N_sat * u_k + N_sat 
            - CIB, which has profile = L_sat * u_k + L_sat * L_cen

        For most profiles, profile = prefactor * u_k, meaning that this will be set to False.
        """
        return False

        
    def u_k_hankel(self, halo_model, x, m, z, params=None):
        """
        Hankel-transform a 3D halo/tracer profile to u_ell for halo model use.
    
        Parameters
        ----------
        x : arrat like
            Radius r scaled by the scale radius x = r / r_s
        z : float or array_like
            Redshift(s).
        m : float or array_like
            Halo mass(es).
        k : array_like, optional
            k values over which the hankel transform will be evaluated. 
            If None, the transform's natural k grid will be output.
            If not None, the transform will be inteprolated to match this k
        params : dict, optional
            Parameter dictionary

        Returns ell, u_ell_m
    
        """

        params = merge_with_defaults(params)
        cparams = halo_model.emulator.get_all_cosmo_params(params=params)
        h = params['H0']/100
       
        W_x = jnp.where((x >= x[0]) & (x <= x[-1]), 1.0, 0.0)

        def single_m_z(m_val, z_val):
            profile = jnp.squeeze(self.profile(halo_model, x, m_val, z_val, params=params))  # remove extra axes
            return profile * x**0.5 * W_x  # shape (Nx,)

        hankel_integrand = jax.vmap(jax.vmap(single_m_z, in_axes=(None, 0)), in_axes=(0, None) )(m, z)
            
        # We need u_k_native to have shape (Nx, Nm, Nz)
        k_native, u_k_native = self._hankel.transform(hankel_integrand)
        u_k_native = jnp.swapaxes(u_k_native, 2, 0)
        u_k_native = jnp.swapaxes(u_k_native, 2, 1)
 
        return k_native, u_k_native

    def u_k_matter(self, halo_model, k, m, z, params=None):
        """
        Calculate u^m(k, M, z) supporting independent dimensions for k, m, and z.
        
        Returns u_k_m with shape (N_k, N_m, N_z).
        """
        params = merge_with_defaults(params)
        
        # Ensure all inputs are 1D arrays
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        
        # Get c_delta and r_delta
        c_delta = halo_model.c_delta(m, z, params=params)
        r_delta = halo_model.r_delta(m, z, params=params)
        lambda_val = 1.0 
        
        # Compute analytical profile q terms with shape: (N_k, N_m, N_z)
        q = k[:, None, None] * r_delta[None, :, :] / c_delta[None, :, :] * (1 + z[None, None, :])
        q_scaled = (1 + lambda_val * c_delta[None, :, :]) * q
        
        Si_q, Ci_q = sici(q)
        Si_q_scaled, Ci_q_scaled = sici(q_scaled)
        
        # NFW normalization
        f_nfw = lambda x: 1.0 / (jnp.log1p(x) - x / (1 + x))
        f_nfw_val = f_nfw(lambda_val * c_delta)
        f_nfw_val = f_nfw_val[None, :, :]  
        
        # Fourier-space profile calculation
        u_k_m = (jnp.cos(q) * (Ci_q_scaled - Ci_q)
                   + jnp.sin(q) * (Si_q_scaled - Si_q)
                   - jnp.sin(lambda_val * c_delta[None,:,:] * q) / q_scaled) * f_nfw_val 
    
        return k, u_k_m
    


class MatterProfile(HaloProfile):
    pass


class GalaxyHODProfile(HaloProfile):
    pass

class DensityProfile(HaloProfile):
    
    def u_k(self, halo_model, k, m, z, moment=1, params=None):
        """
        Compute the kSZ tracer u_ell (Nk, Nm, Nz).
        Supports arbitrary input shapes for k, m, and z.
        """
        
        params = merge_with_defaults(params)
        h = params['H0'] / 100.0
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Compute r_delta and ell_delta
        delta = halo_model.mass_definition.delta
        r_delta = halo_model.r_delta(m, z, params=params)
        d_A_z = jnp.atleast_1d(halo_model.emulator.angular_diameter_distance(z, params=params)) * h
        ell_delta = d_A_z[None, :] / r_delta
        
        # chi: (Nz,) -> Target ell grid: (Nk, Nz)
        chi = d_A_z * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5 
    
        # Calculate kSZ Prefactor as (Nm, Nz)
        vrms = jnp.sqrt(halo_model.emulator.v_rms_squared(z, params=params))
        mu_e = 1.14
        f_free = 1.0
        prefactor = (4 * jnp.pi * r_delta**3 * f_free / mu_e * (1 + z)[None, :]**3 / chi[None, :]**2 * vrms[None, :])
    
        # Get native Hankel transform outputs, which may not align with the k from this function's input
        #k_native, u_k_native = self.u_k_hankel(m, z, params=params) 
        k_native, u_k_native = self.u_k_hankel(halo_model, self.x, m, z, params=params)   # New way
        
        # Calculate native u_ell and the native ell grid
        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None]))
        ell_native = k_native[:, None, None] * ell_delta[None, :, :]
        
        # Apply prefactor and moment logic
        u_ell_base = prefactor[None, :, :] * u_ell_native
        u_ell_val = jax.lax.select(moment == 1, u_ell_base, u_ell_base**2)
    
        # 5. Vectorized Interpolation (Double vmap)
        def interp_single_column(target_x, native_x, native_y):
            return jnp.interp(target_x, native_x, native_y)
    
        # Map over Redshift (Nz) then Mass (Nm)
        vmapped_interp = jax.vmap(
            jax.vmap(interp_single_column, in_axes=(None, 1, 1), out_axes=1),
            in_axes=(1, 2, 2), out_axes=2
        )
        
        u_ell_interp = vmapped_interp(ell_target, ell_native, u_ell_val)
        
        return ell_target, u_ell_interp
    

class PressureProfile(HaloProfile):
     def u_k(self, halo_model, k, m, z, moment=1, params=None):
        
        params = merge_with_defaults(params)
        h = params['H0']/100
        B = self.B
        delta = halo_model.mass_definition.delta
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        
        r_delta = halo_model.r_delta(m, z, params=params) / B**(1/3) # (Nm, Nz)
        d_A = jnp.atleast_1d(halo_model.emulator.angular_diameter_distance(z, params=params)) * h
        ell_delta = d_A[None, :] / r_delta  # (Nm, Nz)
        
        Mpc_per_h_to_cm = Const._Mpc_over_m_ / h # This is actually Mpc_per_h_to_m, but the math is currently working
        prefactor = (1 + z)[None, :] * 4 * jnp.pi * r_delta * Mpc_per_h_to_cm / (ell_delta**2)  # (Nm, Nz)
        
        # Target ell grid for interpolation: (Nk, Nz)
        chi = d_A * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5 
        
        # Get native Hankel transform outputs, which may not align with the k from this function's input
        k_native, u_k_native = self.u_k_hankel(halo_model, self.x, m, z, params=params)  
        
        # Calculate native u_ell and the native ell grid
        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None])) 
        ell_native = k_native[:, None, None] * ell_delta[None, :, :] # (Nk_native, Nm, Nz)
        
        # Apply prefactor and moment
        u_ell_base = prefactor[None, :, :] * u_ell_native # (Nk_native, Nm, Nz)
        u_ell_val = jax.lax.select(moment == 1, u_ell_base, u_ell_base**2)
    
        # Interpolate over the native k-axis (axis 0) for every combination of m and z    
        def interp_at_z(ell_t, ell_n, u_n):
            return jnp.interp(ell_t, ell_n, u_n)
       
        vmap_interp = jax.vmap(
            jax.vmap(interp_at_z, in_axes=(None, 1, 1), out_axes=1), 
            in_axes=(1, 2, 2), out_axes=2
        )
        
        # Resulting shape: (Nk, Nm, Nz)
        u_ell_interp = vmap_interp(ell_target, ell_native, u_ell_val)
        
        return ell_target, u_ell_interp




class CIBProfile(HaloProfile):
    pass



########################################### Density for kSZ Tracers ###########################################

class B16DensityProfile(DensityProfile):
    def __init__(self, x=None, model="agn"):
        # Grid initialization (triggers the x.setter)
        self.x = x if x is not None else jnp.logspace(-4, 1, 256)
        # Model initialization (triggers the model.setter)
        self.model = model

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self._hankel = HankelTransform(self._x, nu=0.5)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        """
        Updates physics attributes by switching between 'agn' and 'shock'.
        Strictly case-insensitive and validates input.
        """
        val_lower = value.lower()
        if val_lower not in ["agn", "shock"]:
            raise ValueError(f"Invalid model '{value}'. Must be 'agn' or 'shock'.")
            
        self._model = val_lower
        b16_configs = self._get_model_configs(self._model)
        
        # Physics parameters reset in throuples
        self.A_rho0, self.A_alpha, self.A_beta = b16_configs['A_rho0'], b16_configs['A_alpha'], b16_configs['A_beta']
        self.alpha_m_rho0, self.alpha_m_alpha, self.alpha_m_beta = b16_configs['alpha_m_rho0'], b16_configs['alpha_m_alpha'], b16_configs['alpha_m_beta']
        self.alpha_z_rho0, self.alpha_z_alpha, self.alpha_z_beta = b16_configs['alpha_z_rho0'], b16_configs['alpha_z_alpha'], b16_configs['alpha_z_beta']

    def _get_model_configs(self, model_key):
        """Internal lookup for Battaglia 2016 Table 2 parameters."""
        AGN = {
            'A_rho0': 4000.0, 'A_alpha': 0.88, 'A_beta': 3.83,
            'alpha_m_rho0': 0.29, 'alpha_m_alpha': -0.03, 'alpha_m_beta': 0.04,
            'alpha_z_rho0': -0.66, 'alpha_z_alpha': 0.19, 'alpha_z_beta': -0.025
        }
        SHOCK = {
            'A_rho0': 1.9e4, 'A_alpha': 0.70, 'A_beta': 4.43,
            'alpha_m_rho0': 0.09, 'alpha_m_alpha': -0.017, 'alpha_m_beta': 0.005,
            'alpha_z_rho0': -0.95, 'alpha_z_alpha': 0.27, 'alpha_z_beta': 0.037
        }
        return SHOCK if model_key == "shock" else AGN


    def profile(self, halo_model, x, m, z, params=None):
        """
        Battaglia et al. 2016 gas density profile (AGN feedback model).
        Fully vectorized to support:
            x.shape = (Nx,)
            m.shape = (Nm,)
            z.shape = (Nz,)
        Output shape: (Nx, Nm, Nz)
        """
        params = merge_with_defaults(params)
        cparams = halo_model.emulator.get_all_cosmo_params(params)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        h = cparams["h"]

        gamma = -0.2
        xc = 0.5
        
        # Ensure 1D and setup broadcasting shapes
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m),  jnp.atleast_1d(z)  # (Nx,)
        x_b, m_b, z_b = x[:, None, None], m[None, :, None], z[None, None, :]      # (Nx, 1, 1), (1, Nm, 1), (1, 1, Nz)
        
        # Critical density broadcast to (1, 1, Nz)
        rho_crit_z = jnp.atleast_1d(halo_model.emulator.critical_density(z, params=params))[None, None, :]
        
        # Mass scaling logic
        m_200c_msun = m_b / h
        mass_ratio = m_200c_msun / 1e14 
       
        # Compute Shape Parameters (Equations A1, A2 from B16)
        rho0 = self.A_rho0 * mass_ratio**self.alpha_m_rho0 * (1 + z_b)**self.alpha_z_rho0 
        alpha = self.A_alpha * mass_ratio**self.alpha_m_alpha * (1 + z_b)**self.alpha_z_alpha 
        beta = self.A_beta * mass_ratio**self.alpha_m_beta * (1 + z_b)**self.alpha_z_beta 
        
        # Profile Shape Function (Nx, Nm, Nz)
        p_x = (x_b / xc)**gamma * (1 + (x_b / xc)**alpha)**(-(beta + gamma) / alpha)
        
        # Final result: M_sun h^2 / Mpc^3 
        rho_gas = rho0 * rho_crit_z * f_b * p_x 
        
        return rho_gas

        

class NFWDensityProfile(DensityProfile):
    def __init__(self, x=None):
        self.x = x if x is not None else jnp.logspace(jnp.log10(1e-4), jnp.log10(1.0), 256)
    

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        """
        Whenever x is modified, immediately rebuild the hankel transform object
        """
        self._x = value
        self._hankel = HankelTransform(self._x, nu=0.5)
        

    def profile(self, halo_model, x, m, z, params=None):
        params = merge_with_defaults(params)
        cparams = halo_model.emulator.get_all_cosmo_params(params)
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m), jnp.atleast_1d(z)
       
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        
        # Get scale radius r_s
        r_delta = halo_model.r_delta(m, z, params=params)
        c_delta = halo_model.c_delta(m, z, params=params)
        r_s = r_delta / c_delta # (Nm, Nz)
        
        # Calculate rho_s
        m_nfw = jnp.log(1 + c_delta) - c_delta / (1 + c_delta) # (Nm, Nz)
        rho_s = m[:, None] / (4 * jnp.pi * r_s**3 * m_nfw)    # (Nm, Nz)
        
        # Final broadcast to (Nx, Nm, Nz)
        # x needs to be (Nx, 1, 1) and rho_s (1, Nm, Nz)
        rho_gas = f_b * rho_s[None, :, :] / (x[:, None, None] * (1 + x[:, None, None])**2)
        
        return rho_gas
   

########################################### Pressure for tSZ Tracers ###########################################


@register_pytree_node_class
class GNFWPressureProfile(PressureProfile):
    def __init__(self, x=None, P0_GNFW=8.130, alpha_GNFW=1.0620, beta_GNFW=5.4807, gamma_GNFW=0.3292, B=1.4):

        self.P0_GNFW = P0_GNFW
        self.alpha_GNFW = alpha_GNFW
        self.beta_GNFW = beta_GNFW
        self.gamma_GNFW = gamma_GNFW
        self.B = B

        self.x = x if x is not None else jnp.logspace(jnp.log10(1e-5), jnp.log10(4.0), 256) 


    @property
    def x(self):
       return self._x

    @x.setter
    def x(self, value):
        """
        Whenever x is modified, immediately rebuild the hankel transform object
        """
        self._x = value
        self._hankel = HankelTransform(self._x, nu=0.5)


    def tree_flatten(self):
        # The dynamic parameters JAX should track
        leaves = (self.P0_GNFW, self.alpha_GNFW, self.beta_GNFW, self.gamma_GNFW, self.B)
        # Static metadata: the grid and the Hankel object
        aux_data = (self._x, self._hankel)
        return (leaves, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        x, hankel = aux_data
        # Create object without calling __init__ to avoid rebuilding Hankel
        obj = cls.__new__(cls)
        obj.P0_GNFW, obj.alpha_GNFW, obj.beta_GNFW, obj.gamma_GNFW, obj.B = leaves
        obj._x = x
        obj._hankel = hankel
        return obj

    def update_params(self, **kwargs):
        """Helper to return a NEW profile with updated leaf values."""
        names = ["P0_GNFW", "alpha_GNFW", "beta_GNFW", "gamma_GNFW", "B"]
        
        # STRICT CHECK: Block typos immediately
        if not set(kwargs).issubset(names):
            invalid = set(kwargs) - set(names)
            raise ValueError(f"Invalid GNFW parameter(s): {invalid}. Expected: {names}")

        leaves, treedef = jax.tree_util.tree_flatten(self)
        new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)
        

    def profile(self, halo_model, x, m, z, params=None):
        """
        GNFW pressure profile as a function of dimensionless scaled radius x = r/r_delta.
        
        Fully vectorized: supports
            x.shape = (Nx,)
            m.shape = (Nm,)
            z.shape = (Nz,)
        Output shape: (Nx, Nm, Nz)
        """
       
    
        # Retrieve all required parameters and ensure all inputs are 1D  
        params = merge_with_defaults(params)
        H0 = params["H0"]
        
        P0, alpha, beta, gamma, B = self.P0_GNFW, self.alpha_GNFW, self.beta_GNFW, self.gamma_GNFW, self.B 
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m), jnp.atleast_1d(z) 
       
        # Helper variables for normalization
        h = H0 / 100.0
        c_km_s = Const._c_ / 1e3
        H = halo_model.emulator.hubble_parameter(z, params=params) * c_km_s  # (Nz,)
        H = jnp.atleast_1d(H)[None, None, :]  # (1, 1, Nz)

        # Corrected mass given the hydrostatic mass bias
        m_delta_tilde = (m / B)[None, :, None]  # (1, Nm, 1)
    
        P_500c = (1.65 * (h / 0.7) ** 2 * (H / H0) ** (8 / 3) * (m_delta_tilde / (0.7 * 3e14)) ** (2 / 3 + 0.12) * (0.7 / h) ** 1.5)  # (1, Nm, Nz)
    
        # Scaled radius and GNFW formula
        c_delta = halo_model.c_delta(m, z, params=params)  # (Nm, Nz)
        scaled_x = c_delta[None, :, :] * x[:, None, None]   # (Nx, Nm, Nz)
        Pe = P_500c * P0 * scaled_x ** (-gamma) * (1 + scaled_x ** alpha) ** ((gamma - beta) / alpha)
    
        return Pe  # shape: (Nx, Nm, Nz)




class NFWMatterProfile(MatterProfile):
    def __init__(self):
        pass


    def u_k(self, halo_model, k, m, z, moment=1, params=None):
        """ 
        Compute either the first or second moment of the CMB lensing tracer u_ell.
        For CMB lensing:, 
            First moment:     W_k_cmb * u_ell_m
            Second moment:    W_k_cmb^2 * u_ell_m^2 
        """

        params = merge_with_defaults(params)
        cparams = halo_model.emulator.get_all_cosmo_params(params)

        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Compute u_m_k from BaseTracer
        k, u_m = self.u_k_matter(halo_model, k, m, z, params=params) 
        
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"]
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # shape (N_m, 1)
        m_over_rho_mean = jnp.broadcast_to(m_over_rho_mean, u_m.shape)

        u_m *= m_over_rho_mean
    
        moment_funcs = [
            lambda _:   u_m ,
            lambda _:   u_m**2,
        ]
    
        u_k = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return k, u_k


########################################### CIB Profiles ###########################################


@register_pytree_node_class
class ShangCIBProfile(CIBProfile):
    def __init__(self, nu, L0_cib=6.4e-8, alpha_cib=0.36, beta_cib=1.75, gamma_cib=1.7,
                 T0_cib=24.4, m_eff_cib=10**12.6, sigma2_LM_cib=0.5, 
                 delta_cib=3.6, z_plateau_cib=1e100, M_min_cib=10**11.5):

        self.nu = nu
        self.L0_cib, self.alpha_cib, self.beta_cib, self.gamma_cib = L0_cib, alpha_cib, beta_cib, gamma_cib
        self.T0_cib, self.m_eff_cib, self.sigma2_LM_cib = T0_cib, m_eff_cib, sigma2_LM_cib
        self.delta_cib, self.z_plateau_cib, self.M_min_cib = delta_cib, z_plateau_cib, M_min_cib

    @property
    def has_central_contribution(self):
        return True

    def tree_flatten(self):
        leaves = (self.nu, self.L0_cib, self.alpha_cib, self.beta_cib, self.gamma_cib, self.T0_cib, 
                  self.m_eff_cib, self.sigma2_LM_cib, self.delta_cib, self.z_plateau_cib, self.M_min_cib)
        return (leaves, None)
        

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        return cls(*leaves)

    def update_params(self, **kwargs):
        names = [
            'nu', 'L0_cib', 'alpha_cib', 'beta_cib', 'gamma_cib', 'T0_cib', 'm_eff_cib',
            'sigma2_LM_cib', 'delta_cib', 'z_plateau_cib', 'M_min_cib'
        ]
        # Check for typos/invalid names
        if not set(kwargs).issubset(names):
            raise ValueError(f"Invalid CIB parameter(s): {set(kwargs) - set(names)}")
    
        leaves, treedef = jax.tree_util.tree_flatten(self)
        new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)


    def sigma(self, m, params=None):
        params = merge_with_defaults(params)
        M_eff_cib, sigma2_LM_cib = self.m_eff_cib, self.sigma2_LM_cib
       
        # Log-normal in mass
        log10_m = jnp.log10(m)
        log10_M_eff = jnp.log10(M_eff_cib)
        Sigma_M = m / jnp.sqrt(2 * jnp.pi * sigma2_LM_cib)  *  jnp.exp( -(log10_m - log10_M_eff)**2 / (2 * sigma2_LM_cib) )
        return Sigma_M


    def phi(self, z, params=None):
        ''' 
        Implementation of Φ(z) = (1 + z)^(δ_CIB) for z < z_plateau, 1 for z >= z_plateau from the Shang model'''
        params = merge_with_defaults(params)
        delta_cib = self.delta_cib
        z_p = self.z_plateau_cib

        Phi_z = jnp.where(z < z_p, (1 + z) ** delta_cib, 1.0)

        return Phi_z


    def theta(self,  z, nu, params=None):
        """Spectral energy distribution function Theta(nu,z) for CIB, analogous to class_sz."""
        params = merge_with_defaults(params)
        T0, alpha_cib, beta_cib, gamma_cib = self.T0_cib, self.alpha_cib, self.beta_cib, self.gamma_cib
    
        h = Const._h_P_  # Planck [J s]
        k_B = Const._k_B_ #1.380649e-23  # Boltzmann [J/K]
        c = Const._c_  #2.99792458e8    # speed of light [m/s]
    
        T_d_z = T0 * (1 + z) ** alpha_cib
    
        x = -(3. + beta_cib + gamma_cib) * jnp.exp(-(3. + beta_cib + gamma_cib))
        # nu0 in GHz
        nu0_GHz = 1e-9 * k_B * T_d_z / h * (3. + beta_cib + gamma_cib + lambertw(x))
        # convert all nu, nu0 to Hz for Planck
        nu_Hz   = nu * 1e9      # If input is GHz!
        nu0_Hz  = nu0_GHz * 1e9
    
        def B_nu(nu_Hz, T):
            return (2 * h * nu_Hz ** 3 / c ** 2) / (jnp.exp(h * nu_Hz / (k_B * T)) - 1)
    
        
        Theta = jnp.where(
            nu_Hz >= nu0_Hz,
            (nu_Hz / nu0_Hz) ** (-gamma_cib),
            (nu_Hz / nu0_Hz) ** beta_cib * (B_nu(nu_Hz, T_d_z) / B_nu(nu0_Hz, T_d_z))
        )
        
        return Theta


    def l_gal(self, halo_model, m, z, nu, params=None):
        # Shang model logic: L0 * Phi(z) * Sigma(m) * Theta(nu_eff)
        phi_z = jnp.atleast_1d(self.phi(z, params=params))[None, :]
        sigma_m = jnp.atleast_1d(self.sigma(m, params=params))[:, None]
        theta_val = jnp.atleast_1d(self.theta(z, nu * (1 + z), params=params))[None, :]
        return self.L0_cib * phi_z * sigma_m * theta_val



    def l_sat(self, halo_model, m, z, nu, params=None):
        def integrate_single_halo(m_single):
            ms_min = self.M_min_cib
            ms_max = m_single
            ngrid = 200
            
            ms_grid = jnp.logspace(jnp.log10(ms_min), jnp.log10(ms_max), ngrid)
            dlnms = jnp.log(ms_grid[1] / ms_grid[0])
            
            # Subhalo mass function
            dn_dlnms = halo_model.subhalo_mass_model.dndlnmu(m_single, ms_grid)
            # Standard Shang luminosity
            l_gal_grid = self.l_gal(halo_model, ms_grid, z, nu, params=params)
            
            return jnp.sum(dn_dlnms[:, None] * l_gal_grid * dlnms, axis=0)

        return jax.vmap(integrate_single_halo)(m)


     
    def l_cen(self, halo_model, m, z, nu, params=None):
        # Shang: Central mass is the full halo mass
        n_cen = jnp.where(m > self.M_min_cib, 1.0, 0.0)
        l_gal = self.l_gal(halo_model, m, z, nu, params=params)
        return n_cen[:, None] * l_gal


     
    def j_bar_nu(self, halo_model, m, z, nu, params=None):
        """
        Compute the mean comoving emissivity j_bar_nu(z) in [Lsun / Mpc^3].
        Integral of (L_cen + L_sat) over the halo mass function.
        """
        params = merge_with_defaults(params)
        h = params["H0"] / 100

        # Get the luminosities (ensure physical mass if needed)
        m_phys = m / h
        lc = self.l_cen(halo_model, m_phys, z, nu, params=params) # Shape: (Nm, Nz)
        ls = self.l_sat(halo_model, m_phys, z, nu, params=params) # Shape: (Nm, Nz)
        
        # Get the halo mass function dn/dlnm 
        dndlnm = halo_model.halo_mass_function(m, z, params=params) # Shape: (Nm, Nz)

        # Correct for Maniyar if needed
        chi = halo_model.emulator.angular_diameter_distance(z, params=params) * (1 + z) 
        
        # Integrate: j_bar = integral [dn/dlnm * (L_c + L_s)] dlnm
        integrand = dndlnm * (lc + ls)
        j_bar = jnp.trapezoid(integrand, x=jnp.log(m), axis=0)

        # Add the consistency counter-term (correction for unbound mass) if hm_consistency is True
        j_bar = jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model.counter_terms(m, z, params=params)[0] * lc[0], lambda x: x, j_bar)
        
        return j_bar * h**3 / (4 * jnp.pi) 


    def monopole(self, halo_model, m, z, nu, params=None):
        """
        Compute total CIB intensity I_nu [Jy/sr] using the line-of-sight integral.
        I_nu = integral [ dchi/dz * a(z) * j_bar_nu(z) ] dz
        """
        params = merge_with_defaults(params)
    
        # Get the mean comoving emissivity (Shape: Nz)
        j_bar = self.j_bar_nu(halo_model, m, z, nu, params=params)
        
        # dchi/dz = c / H(z), a(z) = 1/(1+z)
        dchi_dz = 1.0 / halo_model.emulator.hubble_parameter(z, params=params)
        a = 1.0 / (1.0 + z)
        
        # Final Integral over redshift
        integrand = dchi_dz * a * j_bar
        intensity = jnp.trapezoid(integrand, x=z) 
        
        return intensity


    def sat_and_cen_contribution(self, halo_model, k, m, z, params=None):

        params = merge_with_defaults(params)
        cparams = halo_model.emulator.get_all_cosmo_params(params)
        nu = self.nu
        h = params["H0"]/100
       
        #nu = self.nu 
        chi = halo_model.emulator.angular_diameter_distance(z, params=params) * (1 + z) 

        # Compute the physical mass for ls and lc and then u_k_matter from BaseTracer
        m_physical = m/h
        ls = self.l_sat(halo_model, m_physical, z, nu, params=params)
        lc = self.l_cen(halo_model, m_physical, z, nu , params=params)

        # Apply flux cut if flux cut is not None
        #mask = ((ls + lc) / (4 * jnp.pi * (1 + z) * chi**2) * 1e3 > self.flux_cut) 
        #lc, ls = jax.lax.cond(self.flux_cut is not None, lambda _: (jnp.where(mask, 0.0, lc), jnp.where(mask, 0.0, ls)), lambda _: (lc, ls), operand=None)

        _, u_m = self.u_k_matter(halo_model, k, m, z, params=params)

        # Compute central and satellite terms
        sat_term =  1  / (4*jnp.pi)    *   (ls[None, :, :] * u_m ) 
        cen_term =  1  / (4*jnp.pi)    *   (lc[None, :, :])       

        return sat_term, cen_term


    def u_k(self, halo_model, k, m, z, moment=1, params=None):
        """ 
        Compute either the first or second moment of the CIB tracer.
        Refactored to use sat_and_cen_contribution to avoid redundant math.
        """
        # Get the individual components (scaled correctly by h_factors and 4pi)
        params = merge_with_defaults(params)
        sat_term, cen_term = self.sat_and_cen_contribution(halo_model, k, m, z, params=params)

        moment_funcs = [
            lambda _: cen_term + sat_term,                         # prefactor * (lc[None, :, :] + ls[None, :, :] * u_m ) 
            lambda _: sat_term**2 + 2 * sat_term * cen_term,       # prefactor * (ls[None, :, :]**2 * u_m**2 + 2 * ls[None, :, :] * lc[None, :, :] * u_m ) 
        ]

        u_k = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return k, u_k


        



@register_pytree_node_class
class ManiyarCIBProfile(CIBProfile):
    def __init__(self, nu, eta_max_cib=0.4028, zc_cib=1.5, tau_cib=1.204, fsub_cib=0.134, 
                 M_min_cib=10**11.5, m_eff_cib=10**12.6, sigma2_LM_cib=0.5, s_nu_data=None):
        self.nu = nu
        self.eta_max_cib, self.zc_cib, self.tau_cib, self.fsub_cib = eta_max_cib, zc_cib, tau_cib, fsub_cib
        self.M_min_cib, self.m_eff_cib, self.sigma2_LM_cib = M_min_cib, m_eff_cib, sigma2_LM_cib
        self.s_nu_data = s_nu_data # Passed from Tracer


        if s_nu_data is None:
            s_nu_z_path = os.path.join(get_default_data_path(), "auxiliary_files", "filtered_snu_planck_z_fine.txt")
            s_nu_nu_path = os.path.join(get_default_data_path(), "auxiliary_files", "filtered_snu_planck_nu_fine.txt")
            s_nu_path = os.path.join(get_default_data_path(), "auxiliary_files", "filtered_snu_planck_fine.txt")
            self.s_nu_data = (np.loadtxt(s_nu_z_path), np.loadtxt(s_nu_nu_path), np.loadtxt(s_nu_path))
        else:
            self.s_nu_data = s_nu_data

    @property
    def has_central_contribution(self):
        return True
        
    def tree_flatten(self):
        leaves = (self.nu, self.eta_max_cib, self.zc_cib, self.tau_cib, self.fsub_cib, 
                  self.M_min_cib, self.m_eff_cib, self.sigma2_LM_cib)
        aux = self.s_nu_data
        return (leaves, aux)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        return cls(*leaves, s_nu_data=aux)


    def update_params(self, **kwargs):
        names = ['self.nu', 'eta_max_cib', 'zc_cib', 'tau_cib', 'fsub_cib', 'M_min_cib', 'm_eff_cib', 'sigma2_LM_cib']
        # Check for typos/invalid names
        if not set(kwargs).issubset(names):
            raise ValueError(f"Invalid CIB parameter(s): {set(kwargs) - set(names)}")
    
        leaves, treedef = jax.tree_util.tree_flatten(self)
        new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

    
    def m_dot(self, halo_model, m, z, params=None):
        ''' Mdot =  46.1(1 + 1.11z)E(z)(m /10^12Msun)^1.1 from the Maniyar model'''

        params = merge_with_defaults(params)
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        c_km_s = Const._c_ / 1e3
        E_z = jnp.atleast_1d(halo_model.emulator.hubble_parameter(z, params=params)) * c_km_s / params["H0"]
        
        return 46.1 * (1.0 + 1.11 * z[None, :]) * E_z[None, :] * (m[:, None] / 1e12) ** 1.1


    def sfr_maniyar(self, halo_model, m, z, params=None):
        """
        Compute Maniyar et al. CIB galaxy luminosity from halo mass and redshift.
    
        Returns
        -------
        L_gal : array
            Galaxy luminosity [lsun] per halo
        """

        # Gather all relevant parameters 
        params = merge_with_defaults(params)
        cparams = halo_model.emulator.get_all_cosmo_params(params)
        M_eff, sigma2_LM, eta_max, tau, z_c, f_sub = self.m_eff_cib, self.sigma2_LM_cib, self.eta_max_cib, self.tau_cib, self.zc_cib, self.fsub_cib 
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # sigma^2 depends on whether M < M_eff or > M_eff
        sigma2_lnM = jnp.where(m[:, None] < M_eff,sigma2_LM, (jnp.sqrt(sigma2_LM) - tau * jnp.maximum(0.0, z_c - z[None, :]))**2,)

        # Get the halo accretion rate, baryon fraction, and also take log of relevant quantities
        Mdot = self.m_dot(halo_model, m, z, params=params)
        logM = jnp.log(m)[:, None]
        logMeff = jnp.log(M_eff)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
    
        # Get SFR_c and then use that to get SFR
        sfr_c = eta_max * jnp.exp(- ((logM - logMeff)**2) / (2.0 * sigma2_lnM))
        sfr = 1e10 * Mdot * f_b * sfr_c

        return sfr

    def s_nu_maniyar(self, z, nu, params=None):
        ln_x_grid, ln_nu_grid, ln_s_nu_grid = jnp.log(1 + self.s_nu_data[0]), jnp.log(self.s_nu_data[1]), jnp.log(self.s_nu_data[2])
        _s_nu_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_nu_grid), ln_s_nu_grid)  
        s_nu = jnp.exp(_s_nu_interp((jnp.log(1 + z), jnp.log(nu))))
        return s_nu

        

    def l_gal(self, halo_model, m, z, nu, params=None):
        # Maniyar model logic: 4pi * s_nu * SFR
        s_nu = self.s_nu_maniyar(z, nu, params=params)[None, :]
        sfr = self.sfr_maniyar(halo_model, m, z, params=params)
        return 4 * jnp.pi * s_nu * sfr



    def l_sat(self, halo_model, m, z, nu, params=None):
        def integrate_single_halo(m_single):
            ms_min = self.M_min_cib
            # Host efficiency scaling uses mass corrected by fsub
            ms_max = m_single * (1 - self.fsub_cib)
            ngrid = 200
            
            ms_grid = jnp.logspace(jnp.log10(ms_min), jnp.log10(ms_max), ngrid)
            dlnms = jnp.log(ms_grid[1] / ms_grid[0])
            
            dn_dlnms = halo_model.subhalo_mass_model.dndlnmu(m_single, ms_grid)
            
            # Maniyar Clamping Logic
            sfr_i = self.l_gal(halo_model, ms_grid, z, nu, params=params)
            sfr_ii = self.l_gal(halo_model, ms_max, z, nu, params=params) * ms_grid[:, None] / ms_max
            l_gal_grid = jnp.minimum(sfr_i, sfr_ii)
            
            return jnp.sum(dn_dlnms[:, None] * l_gal_grid * dlnms, axis=0)

        return jax.vmap(integrate_single_halo)(m)


    def l_cen(self, halo_model, m, z, nu, params=None):
        # Maniyar: Central mass is reduced by the subhalo fraction
        m_eff = m * (1 - self.fsub_cib)
        n_cen = jnp.where(m_eff > self.M_min_cib, 1.0, 0.0)
        l_gal = self.l_gal(halo_model, m_eff, z, nu, params=params)
        return n_cen[:, None] * l_gal

    
    
    def j_bar_nu(self, halo_model, m, z, nu, params=None):
        """
        Compute the mean comoving emissivity j_bar_nu(z) in [Lsun / Mpc^3].
        Integral of (L_cen + L_sat) over the halo mass function.
        """
        params = merge_with_defaults(params)
        h = params["H0"] / 100

        # Get the luminosities (ensure physical mass if needed)
        m_phys = m / h
        lc = self.l_cen(halo_model, m_phys, z, nu, params=params) # Shape: (Nm, Nz)
        ls = self.l_sat(halo_model, m_phys, z, nu, params=params) # Shape: (Nm, Nz)
        
        # Get the halo mass function dn/dlnm 
        dndlnm = halo_model.halo_mass_function(m, z, params=params) # Shape: (Nm, Nz)

        # Correct for Maniyar if needed
        chi = halo_model.emulator.angular_diameter_distance(z, params=params) * (1 + z) 
        maniyar_factor = (1+z) * chi**2 #if self.cib_model == 'maniyar' else 1
        
        # Integrate: j_bar = integral [dn/dlnm * (L_c + L_s)] dlnm
        integrand = dndlnm * (lc + ls)
        j_bar = jnp.trapezoid(integrand, x=jnp.log(m), axis=0)

        # Add the consistency counter-term (correction for unbound mass) if hm_consistency is True
        j_bar = jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model.counter_terms(m, z, params=params)[0] * lc[0], lambda x: x, j_bar)
        
        return j_bar * h**3 / (4 * jnp.pi) * maniyar_factor


    def monopole(self, halo_model, m, z, nu, params=None):
        """
        Compute total CIB intensity I_nu [Jy/sr] using the line-of-sight integral.
        I_nu = integral [ dchi/dz * a(z) * j_bar_nu(z) ] dz
        """
        params = merge_with_defaults(params)
    
        # Get the mean comoving emissivity (Shape: Nz)
        j_bar = self.j_bar_nu(halo_model, m, z, nu, params=params)
        
        # dchi/dz = c / H(z), a(z) = 1/(1+z)
        dchi_dz = 1.0 / halo_model.emulator.hubble_parameter(z, params=params)
        a = 1.0 / (1.0 + z)
        
        # Final Integral over redshift
        integrand = dchi_dz * a * j_bar
        intensity = jnp.trapezoid(integrand, x=z) 
        
        return intensity

    
    def sat_and_cen_contribution(self, halo_model, k, m, z, params=None):

        params = merge_with_defaults(params)
        cparams = halo_model.emulator.get_all_cosmo_params(params)
        nu = self.nu
        h = params["H0"]/100
       
        #nu = self.nu 
        chi = halo_model.emulator.angular_diameter_distance(z, params=params) * (1 + z) 

        # Compute the physical mass for ls and lc and then u_k_matter from BaseTracer
        m_physical = m/h
        ls = self.l_sat(halo_model, m_physical, z, nu, params=params)
        lc = self.l_cen(halo_model, m_physical, z, nu , params=params)

        # Apply flux cut if flux cut is not None
        #mask = ((ls + lc) / (4 * jnp.pi * (1 + z) * chi**2) * 1e3 > self.flux_cut) 
        #lc, ls = jax.lax.cond(self.flux_cut is not None, lambda _: (jnp.where(mask, 0.0, lc), jnp.where(mask, 0.0, ls)), lambda _: (lc, ls), operand=None)

        _, u_m = self.u_k_matter(halo_model, k, m, z, params=params)

        # Compute central and satellite terms
        sat_term =  1  / (4*jnp.pi)    *   (ls[None, :, :] * u_m ) 
        cen_term =  1  / (4*jnp.pi)    *   (lc[None, :, :])       

        return sat_term, cen_term


    def u_k(self, halo_model, k, m, z, moment=1, params=None):
        """ 
        Compute either the first or second moment of the CIB tracer.
        Refactored to use sat_and_cen_contribution to avoid redundant math.
        """
        # Get the individual components (scaled correctly by h_factors and 4pi)
        params = merge_with_defaults(params)

        nu = self.nu
        sat_term, cen_term = self.sat_and_cen_contribution(halo_model, k, m, z, nu, params=params)

        moment_funcs = [
            lambda _: cen_term + sat_term,                         # prefactor * (lc[None, :, :] + ls[None, :, :] * u_m ) 
            lambda _: sat_term**2 + 2 * sat_term * cen_term,       # prefactor * (ls[None, :, :]**2 * u_m**2 + 2 * ls[None, :, :] * lc[None, :, :] * u_m ) 
        ]

        u_k = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return k, u_k

     

################################### Galaxy HOD profiles ################################################




@register_pytree_node_class
class StandardGalaxyHODProfile(GalaxyHODProfile):
    """
    Galaxy HOD tracer implementing central + satellite occupation.
    Refactored with individual float attributes to support JAX JIT and Grad.
    """

    def __init__(self, sigma_log10M_HOD=0.68, alpha_s_HOD=1.30, M1_prime_HOD=10**12.7, M_min_HOD=10**11.8, M0_HOD=0.0):        
        
        self.sigma_log10M_HOD, self.alpha_s_HOD, self.M1_prime_HOD, self.M_min_HOD, self.M0_HOD  = sigma_log10M_HOD, alpha_s_HOD, M1_prime_HOD, M_min_HOD, M0_HOD

    @property
    def has_central_contribution(self):
        return True
    
  
    # --- JAX PyTree Registration ---

    def tree_flatten(self):
        # Dynamic leaves (JAX will track these for gradients/jit) and static metadata (changes will trigger a recompile)
        leaves = (self.sigma_log10M_HOD, self.alpha_s_HOD, self.M1_prime_HOD, self.M_min_HOD, self.M0_HOD)
        return (leaves, None)


    @classmethod
    def tree_unflatten(cls, aux, leaves):
        return cls(*leaves)


    def update_params(self, **kwargs):
        names = ['sigma_log10M_HOD', 'alpha_s_HOD', 'M1_prime_HOD', 'M_min_HOD', 'M0_HOD']
        
        # Block typos immediately
        if not set(kwargs).issubset(names):
            raise ValueError(f"Invalid galaxy HOD parameter(s): {set(kwargs) - set(names)}")
    
        leaves, treedef = jax.tree_util.tree_flatten(self)
        new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

    # --- Physics Implementations ---

    def n_cen(self, m, params=None):
        """Mean central occupation."""
        # Using attributes directly as they are now JAX-traced leaves
        x = (jnp.log10(m) - jnp.log10(self.M_min_HOD)) / self.sigma_log10M_HOD
        return 0.5 * (1.0 + erf(x))

    def n_sat(self, m, params=None):
        """Mean satellite occupation."""
        pow_term = jnp.maximum((m - self.M0_HOD) / self.M1_prime_HOD, 0.0)**self.alpha_s_HOD
        return self.n_cen(m) * pow_term

    def ng_bar(self, halo_model, m, z, params=None):
        """Comoving galaxy number density ng(z)."""
        params = merge_with_defaults(params)
        logm = jnp.log(m)
        z = jnp.atleast_1d(z)

        Ntot = self.n_cen(m) + self.n_sat(m)
        dndlnm = halo_model.halo_mass_function(m, z, params=params)
        ng_val = jnp.trapezoid(dndlnm * Ntot[:, None], x=logm, axis=0)

        # HM Consistency check
        return jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model.counter_terms(m, z, params=params)[0] * Ntot[0], lambda x: x, ng_val)

    def galaxy_bias(self, halo_model, m, z, params=None):
        """Compute the large-scale galaxy bias b_g(z)."""
        params = merge_with_defaults(params)
        logm = jnp.log(m)
        z = jnp.atleast_1d(z)

        Ntot = self.n_cen(m) + self.n_sat(m)
        dndlnm = halo_model.halo_mass_function(m, z, params=params)
        bh = halo_model.halo_bias(m, z, order=1, params=params)
        ng = self.ng_bar(halo_model, m, z, params=params)

        bg_num = jnp.trapezoid(dndlnm * bh * Ntot[:, None], x=logm, axis=0)
        bg_num = jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model.counter_terms(m, z, params=params)[1] * Ntot[0], lambda x: x, bg_num)
        return bg_num / ng


    def sat_and_cen_contribution(self, halo_model, k, m, z, params=None):
        """ 
        Compute either the first or second moment of the galaxy HOD tracer u_ell.
        For galaxy HOD:, 
            First moment:     W_g / ng_bar * [Nc + Ns * u_ell_m]
            Second moment:    W_g^2 / ng_bar^2 * [Ns^2 * u_ell_m^2 + 2 * Ns * u_ell_m]
        You cannot simply take u_ell_g**2.
        """

        params = merge_with_defaults(params)
        Ns = self.n_sat(m, params=params)
        Nc = self.n_cen(m, params=params)
        ng = self.ng_bar(halo_model, m, z, params=params) * (params["H0"]/100)**3

        _, u_m = self.u_k_matter(halo_model, k, m, z, params=params)  

        sat_term = (1/ng) * (Ns[None, :, None] * u_m)
        cen_term = (1/ng) * (Nc[None, :, None]**0)
    
        return sat_term, cen_term


    def u_k(self, halo_model, k, m, z, moment=1, params=None):
        """Compute 1st or 2nd moment of the galaxy HOD tracer."""
        params = merge_with_defaults(params)
        Ns = self.n_sat(m, params=params)
        Nc = self.n_cen(m, params=params)
        ng = self.ng_bar(halo_model, m, z, params=params) * (params["H0"]/100)**3

        _, u_m = self.u_k_matter(halo_model, k, m, z, params=params)
    
        moment_funcs = [
            
            lambda _: (1/ng) * (Nc[None, :, None] + Ns[None, :, None] * u_m),
            lambda _: (1/ng**2) * (Ns[None, :, None]**2 * u_m**2 + 2 * Ns[None, :, None] * u_m),
        ]
    
        u_k_res = jax.lax.switch(moment - 1, moment_funcs, None)
        return k, u_k_res




