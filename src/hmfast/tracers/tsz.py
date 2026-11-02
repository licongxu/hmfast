import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from hmfast.tracers.base_tracer import Tracer
from hmfast.utils import Const
from hmfast.halos.profiles import PressureProfile, GNFWPressureProfile

jax.config.update("jax_enable_x64", True)


class tSZTracer(Tracer):
    """
    thermal Sunyaev-Zeldovich effect tracer.

    The kernel has a single term:

    .. math::

        W_{\\mathrm{tSZ}}(\\chi) = \\frac{\\sigma_T}{m_e c^2}\\, \\frac{1}{1+z}

    for :math:`z_{\\min} \\leq z \\leq z_{\\max}`, and zero otherwise, where :math:`\\sigma_T`
    is the Thomson cross-section and :math:`m_e c^2` is the electron
    rest-mass energy. See :meth:`kernel` for how this is returned.

    Attributes
    ----------
    profile : PressureProfile
        Pressure profile used to model the thermal Sunyaev-Zeldovich signal.
    z_range : tuple of float
        ``(z_min, z_max)`` over which the kernel has support; projections start
        at :math:`z = 10^{-5}` if ``z_min`` is lower.
    """

    _required_profile_type = PressureProfile

    def __init__(self, profile=None, *, z_range=(0.0, 5.0)):
        super().__init__(profile=profile or GNFWPressureProfile())
        self._z_range = (jnp.asarray(z_range[0]), jnp.asarray(z_range[1]))

    @property
    def z_range(self):
        return self._z_range

    # --- Begin JAX PyTree Registration ---

    def _tree_flatten(self):
        # z_range is a leaf, so redshift slices can be traced and vmapped without recompiling.
        leaves = (self.profile, self._z_range)
        aux_data = None
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        profile, z_range = leaves
        obj = cls.__new__(cls)
        obj.profile = profile
        obj._z_range = z_range
        return obj

    def update(self, *, profile=None, z_range=None):
        """
        Return a new tSZTracer instance with updated attributes.

        Parameters
        ----------
        profile : PressureProfile, optional
            New pressure profile to use for the tracer. If None, the profile is unchanged.
        z_range : tuple of float, optional
            New ``(z_min, z_max)`` support of the kernel. If None, z_range is unchanged.

        Returns
        -------
        tSZTracer
            New tracer instance with updated attributes.
        """
        (old_profile, old_z_range), aux = self._tree_flatten()
        new_profile = profile if profile is not None else old_profile
        new_z_range = (jnp.asarray(z_range[0]), jnp.asarray(z_range[1])) if z_range is not None else old_z_range
        return self._tree_unflatten(aux, (new_profile, new_z_range))

    # --- End JAX PyTree Registration ---

    def _kernel_primary(self, cosmology, z):

        """
        Compute the tSZ kernel as a function of redshift.

        The kernel is given by:

            .. math::

               W_{\\mathrm{tSZ}}(\\chi) = \\frac{\\sigma_T}{m_e c^2} \\frac{1}{1+z}

        for :math:`z_{\\min} \\leq z \\leq z_{\\max}`, and zero otherwise.

        where :math:`\\sigma_T` is the Thomson cross-section, :math:`m_e c^2` is
        the electron rest-mass energy, and :math:`z` is the redshift.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object.
        z : float or array-like
            Redshift(s).

        Returns
        -------
        W_tsz : float or array-like
            tSZ kernel evaluated at redshift(s) :math:`z`.
        """

        z = jnp.atleast_1d(z)

        m_e = Const.m_e * Const.c**2 / Const.eV
        sigma_T = Const.sigma_T * 1e6
        W = (sigma_T / m_e) * Const.Mpc_over_m / (1.0 + z)
        # Tolerance guards against callers' z grids (e.g. jnp.geomspace) not landing bit-exactly on the ends.
        z_min, z_max = self._z_range
        return jnp.squeeze(jnp.where((z >= z_min - 1e-8) & (z <= z_max + 1e-8), W, 0.0))

    def kernel(self, cosmology, z):
        """
        Radial kernel terms of the thermal Sunyaev-Zeldovich tracer.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object.
        z : float or array_like
            Redshift(s) at which to evaluate the kernels.

        Returns
        -------
        list of tuple of (array_like, int, int)
            One ``(W, n, a)`` per term, specifying how it is projected into :math:`C_\\ell`:
            :math:`W` is the radial kernel, :math:`n` the order of the spherical Bessel
            derivative :math:`j^{(n)}_\\ell(k\\chi)` (with :math:`n=-1` meaning
            :math:`j_\\ell(k\\chi)/(k\\chi)^2`), and :math:`a` the order of the angular derivative,
            which sets the :math:`\\ell`-dependent prefactor (:math:`1`, :math:`\\ell(\\ell+1)` or
            :math:`\\sqrt{(\\ell+2)!/(\\ell-2)!}` for :math:`a = 0, 1, 2`).

            - ``(W_tSZ, 0, 0)``: tSZ kernel :math:`W_{\\rm tSZ}` above.
        """
        return [(self._kernel_primary(cosmology, z), 0, 0)]



jax.tree_util.register_pytree_node(
    tSZTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: tSZTracer._tree_unflatten(aux_data, children)
)