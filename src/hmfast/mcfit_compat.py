"""Import mcfit and restore hmfast JAX dtype settings afterward."""

from hmfast.jax_platform import configure_jax

import mcfit  # noqa: F401  # mcfit sets jax_enable_x64=True on import

configure_jax(force=True)

from mcfit import Hankel, TophatVar  # noqa: F401

__all__ = ["Hankel", "TophatVar", "mcfit"]
