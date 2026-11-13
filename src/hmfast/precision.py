"""Working floating-point precision.

TPU float64 arithmetic overflows inside the float64 range. Products such as
``1e20 * 1e20`` and ``exp`` of arguments that float64 can represent return
inf or nan, so hmfast computes in float32 on TPU.

CPU runs use JAX float64. That is enabled here unless ``JAX_ENABLE_X64`` is
already set in the environment. ``mcfit`` turns x64 on at import;
:func:`configure_precision` runs after that import and puts the flag back.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np


def _env_x64():
    """Return the JAX_ENABLE_X64 override, or None when the variable is unset."""
    value = os.environ.get("JAX_ENABLE_X64")
    if value is None:
        return None
    return value.strip().lower() not in ("0", "false", "no")


def configure_precision():
    """Select float32 on TPU and float64 on CPU.

    ``mcfit`` enables x64 at import, so this applies the choice again afterwards.
    ``JAX_ENABLE_X64`` overrides the CPU choice. TPU always stays in float32.
    """
    on_tpu = any(getattr(device, "platform", None) == "tpu" for device in jax.devices())
    if on_tpu:
        # Forced off even if JAX_ENABLE_X64=1: the TPU float64 results are wrong.
        jax.config.update("jax_enable_x64", False)
    else:
        requested = _env_x64()
        jax.config.update("jax_enable_x64", True if requested is None else requested)
    # TPU's default matmul is bfloat16. That shifts log10 P(k) enough to miss
    # the power spectrum by about a percent, and the mass-function tail by a
    # few percent, with no speedup at these sizes. Full precision matches
    # float64 to a few parts in a million.
    jax.config.update("jax_default_matmul_precision", "highest")


def available_backends():
    """Platforms JAX can see in this process, such as ``("tpu",)`` or ``("cpu", "gpu")``."""
    found = []
    for device in jax.devices():
        if device.platform not in found:
            found.append(device.platform)
    return tuple(found)


def backend():
    """Platform used for new arrays and jit calls.

    This is the device set with :func:`set_backend`, or the process default
    when that has not been called.
    """
    chosen = jax.config.jax_default_device
    if chosen is None:
        return jax.devices()[0].platform
    if isinstance(chosen, str):
        return chosen
    return chosen.platform


def set_backend(name):
    """Run new calculations on ``"cpu"``, ``"gpu"``, or ``"tpu"``.

    The numerics are the same program on every backend. XLA compiles it for
    the device. There is no second, less accurate kernel to pick by accident.

    Precision follows the device. A TPU stays in float32, because its float64
    arithmetic overflows inside the float64 range. A GPU uses float32 with
    full matrix-multiply precision. A CPU uses float64 unless
    ``JAX_ENABLE_X64`` says otherwise. Matrix multiplies stay at full
    precision everywhere: the device default is bfloat16, which moves the
    power spectrum by about a percent at these sizes.

    A process that already has a TPU cannot turn float64 on for a CPU
    selection. For a float64 CPU reference, start another process with
    ``JAX_PLATFORMS=cpu`` and ``JAX_ENABLE_X64=1`` before importing hmfast.
    JAX only sees devices that existed when it started, so a missing platform
    is reported instead of being initialized later.

    Returns the platform name that is now active.
    """
    if not isinstance(name, str):
        raise TypeError(f"backend must be 'cpu', 'gpu', or 'tpu', got {type(name).__name__}")
    name = name.lower()
    if name not in ("cpu", "gpu", "tpu"):
        raise ValueError(
            f"backend must be 'cpu', 'gpu', or 'tpu', got {name!r}. "
            f"Visible devices: {', '.join(available_backends()) or 'none'}."
        )
    visible = available_backends()
    if name not in visible:
        raise RuntimeError(
            f"No {name} device is visible (available: {', '.join(visible) or 'none'}). "
            "Set JAX_PLATFORMS to that backend before importing hmfast."
        )
    if "tpu" in visible:
        jax.config.update("jax_enable_x64", False)
    elif name == "cpu":
        requested = _env_x64()
        jax.config.update("jax_enable_x64", True if requested is None else requested)
    else:
        jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_default_device", name)
    return name


def real_dtype():
    """JAX floating dtype for the active backend."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def numpy_real_dtype():
    """NumPy floating dtype matching :func:`real_dtype`."""
    return np.float64 if jax.config.jax_enable_x64 else np.float32
