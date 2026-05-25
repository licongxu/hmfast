"""
JAX platform and dtype configuration for hmfast.

TPU accelerators do not support 64-bit types (float64 / complex128). hmfast and
mcfit both default to ``jax_enable_x64=True``, which makes mcfit's FFT-based
Hankel / Tophat paths use complex128 kernels and fail on TPU with errors such as
``Element type C128 is not supported on TPU``.

This module centralizes x64 selection and provides dtype helpers used when
building mcfit grids and emulator weights.
"""

from __future__ import annotations

import os
from typing import Optional

import jax
import jax.numpy as jnp

_CONFIGURED = False


def is_tpu_platform() -> bool:
    """Return True when the default JAX device is a TPU."""
    try:
        devices = jax.devices()
        return bool(devices) and devices[0].platform == "tpu"
    except Exception:
        return False


def _env_x64_preference() -> Optional[bool]:
    """Parse HMFAST_JAX_ENABLE_X64 / JAX_ENABLE_X64 if set."""
    for key in ("HMFAST_JAX_ENABLE_X64", "JAX_ENABLE_X64"):
        raw = os.environ.get(key, "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
    return None


def default_enable_x64() -> bool:
    """
    Whether hmfast should enable JAX x64 mode.

    Defaults to True on CPU/GPU and False on TPU unless overridden by environment
    variables ``HMFAST_JAX_ENABLE_X64`` or ``JAX_ENABLE_X64``.
    """
    pref = _env_x64_preference()
    if pref is not None:
        return pref
    return not is_tpu_platform()


def configure_jax(enable_x64: Optional[bool] = None, *, force: bool = False) -> bool:
    """
    Set ``jax_enable_x64`` for hmfast.

    Call again with ``force=True`` after importing mcfit, which unconditionally
    enables x64 in its module body.
    """
    global _CONFIGURED
    if enable_x64 is None:
        enable_x64 = default_enable_x64()
    if _CONFIGURED and not force and enable_x64 == jax.config.jax_enable_x64:
        return enable_x64
    jax.config.update("jax_enable_x64", enable_x64)
    _CONFIGURED = True
    return enable_x64


def jax_enable_x64() -> bool:
    return bool(jax.config.jax_enable_x64)


def float_dtype():
    return jnp.float64 if jax_enable_x64() else jnp.float32


def complex_dtype():
    return jnp.complex128 if jax_enable_x64() else jnp.complex64
