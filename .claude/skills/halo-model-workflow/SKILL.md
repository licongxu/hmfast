---
name: halo-model-workflow
description: >-
  Guides edits to hmfast's differentiable JAX halo model (HaloModel, mass
  function, bias, concentration, profiles). Use when changing src/hmfast/halos/,
  power-spectrum methods, or halo-model consistency flags.
---

# Halo model workflow

## Before editing

1. Read `src/hmfast/halos/halo_model.py` for how **`Cosmology`** is wired (multiple `_load_emulator` calls at init) and how **`TophatVar`** / mcfit is instantiated.
2. Check **`MassDefinition`**, **`halo_mass_function`**, **`halo_bias`**, **`subhalo_mass_function`**, and **`concentration`** defaults — `HaloModel.update()` preserves the immutable-style pattern via pytree flatten/unflatten.

## Implementation tips

- Keep new quantities **JAX-friendly** and consistent with **x64** usage elsewhere.
- If adding parameters, thread them through **`update()`** and **`_tree_flatten` / `_tree_unflatten`** so the model remains a valid **pytree** where intended.
- Add or extend **pytest** cases in `tests/test_halo_model.py` (shapes, initialization, basic spectra).

## Verification

Run from repository root:

```bash
pytest tests/test_halo_model.py -v
```

Extend tests if you add public methods on `HaloModel`.
