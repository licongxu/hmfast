# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**hmfast** is a Python package for **machine-learning–accelerated, differentiable halo-model calculations** for cosmology. It uses **JAX** (`jax`, `jax.numpy`), **emulators** for cosmological backgrounds and spectra, and **mcfit** for correlation-function transforms. Target audience: research code (alpha quality, version 0.1.0).

## Layout

| Path | Role |
|------|------|
| `src/hmfast/cosmology.py` | `Cosmology` — parameters + emulator-backed distances, H(z), P(k), CMB, etc. |
| `src/hmfast/halos/` | `HaloModel`, mass function, bias, concentration, profiles |
| `src/hmfast/tracers/` | Observable tracers (tSZ, kSZ, HOD, lensing, CIB, …) inheriting `Tracer` |
| `src/hmfast/emulator_load.py` | Loading emulator weights |
| `src/hmfast/download.py` | Downloading emulator `.npz` assets |
| `src/hmfast/data/` | Bundled text data (e.g. n(z), filter curves) |

## Public imports

```python
from hmfast import Cosmology                         # top-level export
from hmfast.halos import HaloModel                   # NOT in top-level __init__ (commented out)
from hmfast.tracers import tSZTracer, kSZTracer, GalaxyHODTracer, GalaxyLensingTracer, CMBLensingTracer, CIBTracer
```

**Package import triggers `download_emulators(models=["lcdm", "ede-v2"])` automatically** — be aware for CI or air-gapped machines.

## Architecture

### HaloModel computation methods

`HaloModel` provides four JIT-compiled methods: `pk_1h`, `pk_2h`, `cl_1h`, `cl_2h`. All use `@partial(jax.jit, static_argnums=(1, 2))` — tracer arguments are **static** (traced once per unique tracer pair). The `update()` method returns a new `HaloModel` instance with replaced components, enabling functional-style parameter sweeps.

**Unit conventions**: `k` in Mpc⁻¹, `m` in physical M☉. Internally the code converts to h-units where needed; the linear power spectrum is returned in legacy (Mpc/h)³ units in `pk_2h`.

### JAX pytree registration

`HaloModel` is registered as a JAX pytree (`jax.tree_util.register_pytree_node`). The `Cosmology` instance is the sole child (differentiable leaf); all other components (mass function, bias, concentration, mass definition, flags) are auxiliary data. This is required for `jax.jit`/`jax.grad` to work through `HaloModel`.

### Emulator pre-loading

`HaloModel.__init__` explicitly pre-loads four emulators (`DAZ`, `HZ`, `PKL`, `DER`) via `cosmology._load_emulator(...)` **before any JIT boundary** — emulator loading must never happen inside a jitted function. `TophatVar` (mcfit) is also pre-initialized here.

### Tracer–Profile pairing

Each tracer subclass declares `_required_profile_type` which is enforced at profile assignment time. The pairings are:

| Tracer | Profile type |
|--------|-------------|
| `tSZTracer` | `PressureProfile` |
| `kSZTracer` | `DensityProfile` |
| `GalaxyHODTracer` | `HODProfile` |
| `GalaxyLensingTracer` / `CMBLensingTracer` | `MatterProfile` |
| `CIBTracer` | `CIBProfile` |

Profiles with a central contribution (HOD, CIB) set `has_central_contribution = True` and implement `_sat_and_cen_contribution()`; `pk_1h` dispatches differently for these.

### Cosmology emulator sets

Valid `emulator_set` strings: `"lcdm:v1"`, `"mnu:v1"`, `"neff:v1"`, `"wcdm:v1"`, `"ede:v1"`, `"mnu-3states:v1"`, `"ede:v2"`. Invalid keys raise with an explicit list. `jax_enable_x64` is set globally in both `cosmology.py` and `halo_model.py`.

## Conventions

1. **JAX**: Use `jnp` inside numeric hot paths; do not silently switch to float32 in core physics.
2. **mcfit `==0.0.22`** is pinned — do not bump without compatibility checks.
3. **New tracers**: subclass `Tracer`, set `_required_profile_type`, implement `kernel(cosmology, z)`.
4. **New profiles**: subclass the appropriate `HaloProfile` subclass; implement `u_k` and `u_r`.
5. **Tests**: assert shapes and finite outputs; run `pytest` from repo root.

## Related agent assets

`.claude/rules/` contains always-on conventions for JAX/numerics, emulators/data, and API layering. `.claude/skills/` contains task workflows for halo model, tracers, and cosmology emulators — read the relevant skill before large edits.
