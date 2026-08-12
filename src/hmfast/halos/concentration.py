import jax
import jax.numpy as jnp
from functools import partial
from abc import ABC, abstractmethod

from hmfast.halos.mass_definition import MassDefinition, convert_m_delta



class Concentration(ABC):
    """
    Abstract base class for all concentration-mass relations.
    All subclasses must implement the c_delta method.
    """
    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def c_delta(self, halo_model, m, z):
        """
        Compute the concentration parameter :math:`c_\\Delta`.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology and target mass definition.
        m : array-like
            Halo masses in physical :math:`M_\\odot`.
        z : array-like
            Redshifts.
        """
        pass


class D08Concentration(Concentration):
    """
    Concentration-mass relation from `Duffy et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008MNRAS.390L..64D/abstract>`_.

    The fitted relation is

    .. math::

        c_\\Delta(M, z) = A \\left(\\frac{M}{M_\\mathrm{pivot}}\\right)^B (1+z)^C

    where :math:`A`, :math:`B`, :math:`C`, and :math:`M_\\mathrm{pivot}` are fit parameters.

    Calibrated for 200c, 200m, and virial mass definitions.
    """
    def __init__(self):
        pass


    @partial(jax.jit, static_argnums=(0,))
    def c_delta(self, halo_model, m, z):
        """
        Compute the concentration parameter.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology and target mass definition.
        m : array-like
            Halo masses in physical :math:`M_\\odot`.
        z : array-like
            Redshifts.

        Returns
        -------
        array-like
            Concentration values with shape :math:`(N_m, N_z)`.
        """
        
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        h = halo_model.cosmology.H0 / 100.0
        m_internal = m * h
        mdef = halo_model.mass_definition

        # Parameter Lookup Table
        coeffs = {
            (200, "critical"):       (5.71, -0.084, -0.47, 2e12),
            (200, "mean"):           (10.14, -0.081, -1.01, 2e12),
            ("vir", "critical"):     (7.85, -0.081, -0.71, 2e12),
        }
        
        # Determine if we have a direct match or need conversion
        key = (mdef.delta, mdef.reference) 
        
        if key in coeffs:
            A, B, C, M_pivot = coeffs[key]
            return A * (m_internal[:, None] / M_pivot)**B * (1 + z[None, :])**C

        if not halo_model.convert_masses:
            raise ValueError(f"Mass definition {key} incompatible with the selected concentration-mass relation.")

        # Conversion Logic (Native 200c)
        A, B, C, M_pivot = coeffs[(200, "critical")]
        native_def = MassDefinition(200, "critical")
        c_seed = A * (m_internal[:, None] / M_pivot)**B * (1 + z[None, :])**C
        m_200c = convert_m_delta(halo_model.cosmology, m, z, mass_def_old=mdef, mass_def_new=native_def, c_old=c_seed)
        
        # Compute r_s from native 200c mesh
        c_200c = A * ((m_200c * h) / M_pivot)**B * (1 + z[None, :])**C
        r_200c = jax.vmap(lambda mc, zi: native_def.r_delta(halo_model.cosmology, mc, zi), (1, 0))(m_200c, z).T
        
        # Final Target Radius / r_s
        r_target = mdef.r_delta(halo_model.cosmology, m, z)
        return (r_target * c_200c / r_200c).reshape(len(m), len(z))

