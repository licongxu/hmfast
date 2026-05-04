---
description: JAX, floating-point, and numerical conventions for hmfast
---

# JAX and numerics

- Use **`jax.numpy` (`jnp`)** for arrays that must compose with JIT, gradients, or the halo model; align with existing modules (`cosmology.py`, `halos/halo_model.py`).
- The codebase sets **`jax_enable_x64`** where needed — do not silently switch to float32 in core physics unless explicitly requested and validated.
- **`mcfit`** is used with JAX backend in halo-model code (e.g. `TophatVar` with `backend='jax'`). Keep interfaces compatible with existing `partial` / closure patterns.
- Prefer **small, shape-focused tests** (like `tests/test_halo_model.py`) when adding numerical paths: assert shapes and finite outputs before claiming physical correctness.
