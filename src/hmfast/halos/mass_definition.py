import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from functools import partial

from hmfast.utils import newton_root

class MassDefinition:
    """
    Mass definition for halos, specifying the overdensity threshold and reference density.

    This class encapsulates the definition of a halo mass in terms of an overdensity
    parameter (`delta`) and a reference density (`reference`). The mass can be defined
    with respect to either the critical density or the mean matter density of the universe,
    and can use a fixed overdensity (e.g., 200) or the redshift-dependent 'virial' value.

    Parameters
    ----------
    delta : int, float, or str, optional
        Overdensity parameter. Can be a numeric value (e.g., 200, 500) or the string 'vir'
        for the redshift-dependent virial overdensity. Default is 200.
    reference : {'critical', 'mean'}, optional
        Reference density for the overdensity threshold. Must be either 'critical' (for
        critical density) or 'mean' (for mean matter density). Default is 'critical'.

    Attributes
    ----------
    delta : int, float, or str
        The overdensity parameter. If 'vir', the virial overdensity is used.
    reference : str
        The reference density, either 'critical' or 'mean'.

    Raises
    ------
    ValueError
        If an invalid combination of `delta` and `reference` is provided, or if either
        parameter is set to an unsupported value.
    """

    def __init__(self, delta=200, reference="critical"):
        self._delta = None
        self._reference = None
        self.reference = reference
        self.delta = delta
        
    # Ensure that reference is only ever critical or mean
    @property
    def reference(self):
        return self._reference
    
    @reference.setter
    def reference(self, value):
        value = str(value).lower()
        if value not in ("critical", "mean"):
            raise ValueError("reference must be either 'critical' or 'mean'")
            
        # Prevent changing reference if delta == "vir"
        if getattr(self, "_delta", None) == "vir" and value != "critical":
            raise ValueError("'vir' is only allowed with 'critical' reference")
        self._reference = value

        
    @property
    def delta(self):
        return self._delta

        
    @delta.setter
    def delta(self, value):
        if isinstance(value, str):
            value = value.lower()
            
        # If 'vir', reference must be 'critical'
        if value == "vir":
            if getattr(self, "_reference", None) != "critical":
                raise ValueError("'vir' is only allowed with 'critical' reference")
            self._delta = value
            return

        # Otherwise, it must be numeric
        if isinstance(value, (int, float)):
            self._delta = value
            return

        raise ValueError("delta must be numeric or 'vir'")


    def _tree_flatten(self):
        # delta can be a tracer (numeric) or a static string ('vir')
        # reference is always a static string for critical/mean
        children = () 
        aux_data = (self.delta, self.reference) 
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


jax.tree_util.register_pytree_node(
    MassDefinition,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: MassDefinition._tree_unflatten(aux_data, children)
)


