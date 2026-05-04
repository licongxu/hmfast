---
name: tracers-and-profiles
description: >-
  Guides adding or modifying hmfast tracers and halo profiles. Use when editing
  src/hmfast/tracers/ or src/hmfast/halos/profiles/, or integrating new
  observables with HaloModel.
---

# Tracers and profiles

## New tracer

1. Subclass **`Tracer`** from `tracers/base_tracer.py`.
2. Set **`_required_profile_type`** to the correct **`HaloProfile`** subclass (pressure vs density vs matter, etc.).
3. Implement **`kernel(self, cosmology, z)`** with the same JAX/numpy conventions as sibling tracers.
4. Register the class in **`tracers/__init__.py`** and **`__all__`**.
5. Reuse **`_load_dndz_data`** / **`_normalize_dndz`** when working with **n(z)** files like existing galaxy tracers.

## New or changed profile

1. Place implementations under **`halos/profiles/`**, inheriting from the appropriate base (`HaloProfile`, pressure, density, matter, HOD modules).
2. Export from **`halos/profiles/__init__.py`** if needed for clean imports.
3. Pair the profile with tracers that enforce the type via **`Tracer.profile`**.

## Integration

- Follow existing files (**`tsz.py`**, **`galaxy_hod.py`**, **`cmb_lensing.py`**) for how **`Cosmology`**, **data paths**, and **`Const`** are used.

## Tests

- Add focused tests under **`tests/`** (kernels shapes, normalization, or finite outputs at reference **z**).
