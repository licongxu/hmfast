import jax
import jax.numpy as jnp
from functools import partial


class T10HaloBias:
    """
    Tinker et al. (2010) large-scale linear bias, JAX-friendly.

    Parameters
    ----------
    sigmas : jnp.ndarray
        sigma(R,z) or sigma(M,z), shape (nM, n_z)
    z : scalar or array_like
        Redshift(s) (kept for API compatibility)
    delta_mean : scalar or array_like
        Halo overdensity Δ, shape (n_z,) or scalar

    Returns
    -------
    b_nu : jnp.ndarray
        Halo bias, shape same as sigmas
    """

    def __init__(self):
        pass


    @partial(jax.jit, static_argnums=(0,))
    def b1_nu(self, sigmas, z, delta_mean):
        y = jnp.log10(delta_mean)
        delta_c = 1.686  # the critical overdensity (slightly redshift-dependent in LCDM), so this is approximate
        
        # Tinker (2010) parameters
        A  = jnp.array(1.0 + 0.24 * y * jnp.exp(-(4.0 / y) ** 4))
        a  = jnp.array(0.44 * y - 0.88)
        B  = jnp.array(0.183)
        b_ = jnp.array(1.5)
        C  = jnp.array((0.019 + 0.107 * y + 0.19 * jnp.exp(-(4.0 / y) ** 4)))
        c  = jnp.array(2.4)
    
        nu = delta_c / sigmas
        nu_a = jnp.power(nu, a)
        first = A * (nu_a / (nu_a + delta_c ** a))
        b_nu = 1.0 - first + B * jnp.power(nu, b_) + C * jnp.power(nu, c)
    
        return b_nu


    @partial(jax.jit, static_argnums=(0,))
    def b2_nu(self, sigmas, z, delta_mean):

        delta_c =  1.686
        nu = (delta_c / sigmas)**2

        z = jnp.atleast_1d(z)
        
        # Base parameters followed by redshift exponents
        alpha0, beta0, gamma0, eta0, phi0 = 0.368, 0.589, 0.864, -0.243, -0.729
        alpha_z, beta_z, gamma_z, eta_z, phi_z = 0.0, 0.2, -0.01, 0.27, -0.08

        # Compute z-dependent parameters
        alpha = alpha0 * (1 + z)**alpha_z
        beta  = beta0  * (1 + z)**beta_z
        gamma = gamma0 * (1 + z)**gamma_z
        eta   = eta0   * (1 + z)**eta_z
        phi   = phi0   * (1 + z)**phi_z


        a = -phi
        b = beta**2
        c = gamma
        d = eta + 0.5

        a2 = -17/21
        

        eps1 = (c * nu - 2 * d) / delta_c
        eps2 = (c * nu * (c * nu - 4 * d - 1) + 2 * d * (2 * d - 1)) /  delta_c**2
        
        E1 = - 2 * a / (delta_c * ((b * nu)**(-a) + 1))
        E2 = E1  * (-2 * a + 2 * c * nu - 4 * d + 1) / delta_c

        b2_nu = 2 * (1 + a2) * (eps1 + E1) + eps2 + E2

        return b2_nu

