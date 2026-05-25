---
description: JAX, floating-point, and numerical conventions for hmfast
---

# JAX and numerics

- Use **`jax.numpy` (`jnp`)** for arrays that must compose with JIT, gradients, or the halo model; align with existing modules (`cosmology.py`, `halos/halo_model.py`).
- **`jax_enable_x64`** defaults via `hmfast.jax_platform.configure_jax()`: **True** on CPU/GPU, **False** on TPU (mcfit FFT needs complex64). Override with `HMFAST_JAX_ENABLE_X64`. See `docs/tpu.md`.
- Import mcfit through **`hmfast.mcfit_compat`** (not raw `mcfit`) so x64 settings are restored after mcfit import.
- **`mcfit`** is used with JAX backend in halo-model code (e.g. `TophatVar` with `backend='jax'`). Pass log-spaced grids with `float_dtype()` so convolution kernels stay TPU-safe.
- Prefer **small, shape-focused tests** (like `tests/test_halo_model.py`) when adding numerical paths: assert shapes and finite outputs before claiming physical correctness.
