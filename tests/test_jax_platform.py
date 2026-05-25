"""Tests for JAX platform / TPU-safe dtype configuration."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hmfast.jax_platform import configure_jax, default_enable_x64, float_dtype


@pytest.fixture(autouse=True)
def _restore_x64():
    prev = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", prev)


def test_mcfit_kernel_dtype_follows_x64_setting():
    """mcfit must use complex64 kernels when x64 is disabled (TPU-safe path)."""
    configure_jax(False, force=True)
    from hmfast.mcfit_compat import Hankel

    x = jnp.geomspace(1e-2, 1e2, 32, dtype=float_dtype())
    h = Hankel(x, nu=0.5, lowring=True, backend="jax")
    assert h._u.dtype == jnp.complex64

    f = jnp.ones(32, dtype=float_dtype())
    y, g = jax.jit(lambda arr: h(arr, extrap=False))(f)
    assert jnp.all(jnp.isfinite(g))


def test_mcfit_compat_restores_x64_off_after_import(monkeypatch):
    monkeypatch.setenv("HMFAST_JAX_ENABLE_X64", "0")
    configure_jax(force=True)
    import importlib
    import hmfast.mcfit_compat as mc

    importlib.reload(mc)
    assert jax.config.jax_enable_x64 is False


def test_default_enable_x64_false_on_tpu(monkeypatch):
    class _FakeDevice:
        platform = "tpu"

    monkeypatch.setattr(
        "hmfast.jax_platform.jax.devices",
        lambda: [_FakeDevice()],
    )
    monkeypatch.delenv("HMFAST_JAX_ENABLE_X64", raising=False)
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    assert default_enable_x64() is False


def test_hankel_transform_wrapper_uses_float32_grid():
    configure_jax(False, force=True)
    from hmfast.halos.profiles.base_profile import HankelTransform

    x = np.logspace(-2, 2, 32)
    ht = HankelTransform(x, nu=0.5)
    assert ht._hankel._u.dtype == jnp.complex64
