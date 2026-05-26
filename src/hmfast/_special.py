"""
Special-function shims for hmfast.

``jax.scipy.special.sici`` was only added in recent JAX releases (see
https://github.com/jax-ml/jax/pull/32052). The jax[tpu] wheel that ships with
``libtpu`` on Cloud TPU v6e is jax 0.6.2, which does **not** expose ``sici``.
Importing hmfast on that VM then fails with::

    ImportError: cannot import name 'sici' from 'jax.scipy.special'

This module provides a ``sici`` callable that:

1. Prefers ``jax.scipy.special.sici`` when it is available.
2. Falls back to a ``jax.pure_callback`` wrapper around
   ``scipy.special.sici``. ``pure_callback`` is JIT- and vmap-safe across
   CPU/GPU/TPU backends; on TPU it shuttles the array to the host, runs SciPy,
   and ships the result back. That is slower than a fused TPU kernel but keeps
   the import (and the rest of the model) working on older JAX.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

try:
    from jax.scipy.special import sici as _jax_sici  # type: ignore[attr-defined]

    _HAS_JAX_SICI = True
except ImportError:  # pragma: no cover - exercised on older JAX (e.g. jax 0.6.2)
    _jax_sici = None
    _HAS_JAX_SICI = False


def _make_scipy_sici_host(out_dtype):
    def _host(x):
        import scipy.special  # imported lazily so the dep is optional at import

        si, ci = scipy.special.sici(x)
        # scipy.special.sici always promotes to float64. Cast back to the
        # caller's requested dtype so the pure_callback boundary doesn't have
        # to do a hidden truncation (which on TPU can overflow to NaN).
        return si.astype(out_dtype), ci.astype(out_dtype)

    return _host


def _sici_via_callback(x: Array) -> Tuple[Array, Array]:
    x = jnp.asarray(x)
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    si, ci = jax.pure_callback(
        _make_scipy_sici_host(x.dtype),
        (out_shape, out_shape),
        x,
        vmap_method="sequential",
    )
    return si, ci


def sici(x: Array) -> Tuple[Array, Array]:
    """Sine and cosine integrals ``Si(x), Ci(x)``.

    Drop-in replacement for :func:`jax.scipy.special.sici` that also works on
    older JAX releases by falling back to :func:`scipy.special.sici` through
    :func:`jax.pure_callback`.
    """
    if _HAS_JAX_SICI:
        return _jax_sici(x)
    return _sici_via_callback(x)


__all__ = ["sici"]
