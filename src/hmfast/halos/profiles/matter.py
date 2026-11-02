import os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import sici
from jax.tree_util import register_pytree_node_class

from hmfast.halos.massdef import MassDefinition
from hmfast.halos.profiles import HaloProfile


class MatterProfile(HaloProfile):
    """
    Parent matter density profile class from which matter profile classes inherit.

    Child profile classes must implement :meth:`real` and :meth:`fourier`.
    """
    pass



class NFWMatterProfile(MatterProfile):
    """
    Matter density profile from `Navarro, Frenk & White (1997) <https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract>`_.

    The real-space mass-weighted matter profile is written as

    .. math::

        u_r(r, M, z) = \\frac{1}{\\bar{\\rho}_{m,0}} \\,
        \\frac{\\rho_s}{(r/r_s) \\left(1+r/r_s\\right)^2}
        \\tag{1}

    .. math::

        \\rho_s = \\frac{M}{4\\pi r_s^3}
        \\left[\\ln(1+c) - \\frac{c}{1+c}\\right]^{-1}
        \\tag{2}

    with :math:`r_s = r_\\Delta / c`.

    The Fourier-space mass-weighted matter profile is written as

    .. math::

        u(k, M, z)
        = \\frac{M}{\\bar{\\rho}_{m,0}}
        \\left[\\ln(1+c) - \\frac{c}{1+c}\\right]^{-1}
        \\Bigg[
        \\cos(q) \\left(\\mathrm{Ci}[(1+c)q] - \\mathrm{Ci}(q)\\right)
        + \\sin(q) \\left(\\mathrm{Si}[(1+c)q] - \\mathrm{Si}(q)\\right)
        - \\frac{\\sin(cq)}{(1+c)q}
        \\Bigg]
        \\tag{3}

    with :math:`q = k \\, r_s \\, (1+z)`.
    """
    def __init__(self):
        pass

    def _tree_flatten(self):
        return ((), None)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        return cls()

    @jax.jit
    def real(self, halo_model, r, m, z):
        """
        Compute the real-space mass-weighted NFW matter profile.

        This evaluates Eqs. (1) and (2).

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, concentration relation, and halo
            radius.
        r : float or jnp.ndarray
            Radius or radii in :math:`\\mathrm{Mpc}`.
        m : float or jnp.ndarray
            Halo mass(es) in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Real-space profile with shape :math:`(N_r, N_m, N_z)`, where
            singleton dimensions get squeezed before return.
        """
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)
        c = jnp.reshape(halo_model.concentration.c_delta(halo_model.cosmology, m, z, mass_def=halo_model.mass_def), (len(m), len(z)))
        r_delta = jnp.reshape(halo_model.mass_def.r_delta(halo_model.cosmology, m, z), (len(m), len(z)))
        r_s = r_delta * (1.0 + z[None, :]) / c

        # Unit-mass profile of Eq. (2), truncated at r_Delta (x = c).
        f_nfw = 1.0 / (jnp.log1p(c) - c / (1.0 + c))
        x = r[:, None, None] / r_s[None, :, :]
        u_r = (1.0 / (4.0 * jnp.pi * r_s**3))[None, :, :] * f_nfw[None, :, :] / (x * (1.0 + x) ** 2)
        # NaN-safe: a bare `x <= c` would silently zero out NaN instead of propagating it.
        u_r = jnp.where(jnp.isnan(x) | jnp.isnan(c[None, :, :]) | (x <= c[None, :, :]), u_r, 0.0)

        rho_mean_0 = halo_model.cosmology.rho_x(0.0, "matter")  # delta_m = rho_halo / rho_m; neutrinos stay smooth
        return jnp.squeeze((m[:, None] / rho_mean_0)[None, :, :] * u_r)


    @jax.jit
    def fourier(self, halo_model, k, m, z):
        """
        Compute the mass-weighted NFW matter profile in Fourier space.

        This evaluates Eq. (3).
    
        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, concentration relation, and halo
            radius.
        k : float or jnp.ndarray
            Comoving wavenumber(s) in :math:`\\mathrm{Mpc}^{-1}`.
        m : float or jnp.ndarray
            Halo mass(es) in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).
            
        Returns
        -------
        jnp.ndarray
            Fourier-space profile with shape :math:`(N_k, N_m, N_z)`, where
            singleton dimensions get squeezed before return.
        """

        
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        c = jnp.reshape(halo_model.concentration.c_delta(halo_model.cosmology, m, z, mass_def=halo_model.mass_def), (len(m), len(z)))
        r_delta = jnp.reshape(halo_model.mass_def.r_delta(halo_model.cosmology, m, z), (len(m), len(z)))

        # Unit-mass truncated NFW of Eq. (3), shape (N_k, N_m, N_z).
        c_b = c[None, :, :]
        q = k[:, None, None] * r_delta[None, :, :] / c_b * (1 + z[None, None, :])
        q_scaled = (1 + c_b) * q
        Si_q, Ci_q = sici(q)
        Si_q_scaled, Ci_q_scaled = sici(q_scaled)
        f_nfw = 1.0 / (jnp.log1p(c_b) - c_b / (1 + c_b))
        u_k = (jnp.cos(q) * (Ci_q_scaled - Ci_q) + jnp.sin(q) * (Si_q_scaled - Si_q) - jnp.sin(c_b * q) / q_scaled) * f_nfw

        rho_mean_0 = halo_model.cosmology.rho_x(0.0, "matter")  # delta_m = rho_halo / rho_m; neutrinos stay smooth
        return jnp.squeeze(u_k * (m / rho_mean_0)[None, :, None])


jax.tree_util.register_pytree_node(
    NFWMatterProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: NFWMatterProfile._tree_unflatten(aux_data, children),
)
