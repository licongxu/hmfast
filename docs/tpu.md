# Running hmfast on TPU

## Root cause

hmfast uses [mcfit](https://github.com/cosmodesi/mcfit) for Hankel transforms (`Hankel`) and top-hat variance integrals (`TophatVar`). With the JAX backend, mcfit performs `rfft` / `hfft` convolutions.

Both **hmfast** and **mcfit** historically forced `jax_enable_x64=True`. On TPU that implies **complex128** FFT kernels, which the TPU driver rejects, for example:

```text
XlaRuntimeError: INVALID_ARGUMENT: Element type C128 is not supported on TPU.
```

or:

```text
RuntimeError: 64-bit data types are not yet supported on the TPU driver API.
```

GPU and CPU backends tolerate float64/complex128; TPU does not.

## Fix in hmfast

hmfast now configures JAX dtypes via `hmfast.jax_platform`:

- On **TPU** (detected from `jax.devices()[0].platform == "tpu"`), `jax_enable_x64` defaults to **False** (float32 / complex64).
- On **CPU/GPU**, the default remains **True** (float64), preserving existing accuracy.
- Override with environment variables:
  - `HMFAST_JAX_ENABLE_X64=0` or `1`
  - `JAX_ENABLE_X64=0` or `1` (also respected)

Imports go through `hmfast.mcfit_compat`, which re-applies hmfast's x64 setting after mcfit import (mcfit unconditionally enables x64 in its `__init__.py`).

## Usage on Cloud TPU VM

```bash
export JAX_PLATFORMS=tpu
# optional explicit override (auto-detected on TPU anyway):
export HMFAST_JAX_ENABLE_X64=0

python -c "import jax; print(jax.devices()); import hmfast; from hmfast.cosmology import Cosmology; print(jax.config.jax_enable_x64)"
```

Expect `jax_enable_x64` to be `False` on TPU. Numerical results will differ slightly from GPU float64 runs; compare key observables if you need sub-percent agreement.

## Accuracy trade-off

TPU mode uses float32 throughout emulators and mcfit transforms. For publication-grade MCMC on TPU, validate a subset of spectra against a GPU float64 reference. For maximum accuracy, keep inference on **GPU** with the default x64 settings.
