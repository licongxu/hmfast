# hmfast

Fast, differentiable halo-model predictions for cosmology, built on JAX and
neural-network emulators.

## Overview

**hmfast** provides JIT-compilable and autodiff-compatible halo model
calculations.  It combines:

- **JAX** for just-in-time compilation and automatic differentiation.
- **Neural emulators** (trained on CLASS) for background quantities and linear
  power spectra.
- **mcfit** (v0.0.22) for fast Hankel transforms used in Fourier-space profile
  projections.
- A modular split between **halo ingredients** (mass function, bias,
  concentration, profiles) and **observable tracers** (tSZ, kSZ, CMB lensing,
  CIB, galaxy HOD, galaxy lensing).

## Installation

```bash
pip install -e ".[dev]"
```

Importing `hmfast` may download emulator weights on first use (~few MB).

## Quick start

```python
from hmfast.halos import HaloModel
from hmfast.tracers import tSZTracer, CMBLensingTracer

hm = HaloModel()

# Access cosmology
print("h =", hm.cosmology.H0 / 100.0)

# Linear power spectrum
k, pk = hm.cosmology.pk(z=0.0, linear=True)

# 1-halo power spectrum for CMB lensing
tracer = CMBLensingTracer()
import jax.numpy as jnp
m = jnp.logspace(11, 15, 20)
k = jnp.logspace(-3, 1, 50)
z = jnp.array([0.5])
p1h = hm.pk_1h(tracer, tracer, k, m, z)
```

## Pressure profiles

hmfast ships three electron-pressure profile models for thermal SZ
calculations:

| Profile | Reference | Description |
|---------|-----------|-------------|
| `GNFWPressureProfile` | Nagai, Kravtsov & Vikhlinin (2007) | Generalised NFW with Arnaud P500c normalization |
| `ParametricGNFWPressureProfile` | Parametric y0 rescaling | GNFW shape with a parametric Compton-y0 amplitude (A_SZ, alpha_SZ) |
| `B12PressureProfile` | Battaglia et al. (2012) | Mass- and redshift-dependent shape parameters |

### Using the parametric pressure profile

```python
from hmfast.halos import HaloModel
from hmfast.halos.profiles import ParametricGNFWPressureProfile
from hmfast.tracers import tSZTracer

hm = HaloModel()
profile = ParametricGNFWPressureProfile(
    A_SZ=-4.97,       # log10 amplitude
    alpha_SZ=0.7867,  # mass scaling exponent
)
tracer = tSZTracer(profile=profile)

# Real-space pressure profile
import jax.numpy as jnp
r = jnp.logspace(-2, 1, 50)
m = jnp.logspace(13, 15, 10)
z = jnp.array([0.3, 0.7, 1.0])
Pe = profile.u_r(hm, r, m, z)  # shape (Nr, Nm, Nz)
```

The profile supports JAX automatic differentiation through its parameters:

```python
import jax

def loss(A_SZ):
    p = profile.update(A_SZ=A_SZ)
    return jnp.sum(p.u_r(hm, r, m, z) ** 2)

grad = jax.grad(loss)(profile.A_SZ)
```

## Tracers

| Tracer | Profile | Observable |
|--------|---------|------------|
| `tSZTracer` | `PressureProfile` | Thermal Sunyaev-Zeldovich |
| `kSZTracer` | `PressureProfile` | Kinematic Sunyaev-Zeldovich |
| `CMBLensingTracer` | `MatterProfile` | CMB lensing convergence |
| `CIBTracer` | `CIBProfile` | Cosmic infrared background |
| `GalaxyHODTracer` | `GalaxyHODProfile` | Galaxy clustering (HOD) |
| `GalaxyLensingTracer` | `DensityProfile` | Galaxy-galaxy lensing |

## Halo model ingredients

| Component | Module | Models |
|-----------|--------|--------|
| Mass function | `halos.massfunc` | Tinker et al. (2008), Tinker et al. (2010) |
| Halo bias | `halos.bias` | Tinker et al. (2010) |
| Concentration | `halos.concentration` | Duffy et al. (2008) |
| Mass definition | `halos.mass_definition` | Spherical overdensity (critical/mean) |

## Testing

```bash
pytest                                    # unit tests
pytest tests/stress_test_parametric_gnfw.py  # parametric profile stress tests
```

## Project structure

```
src/hmfast/
  cosmology.py          Cosmology class with emulator-backed predictions
  emulator_load.py      Neural emulator loading and inference
  download.py           Emulator weight download utilities
  utils.py              Physical constants
  halos/
    halo_model.py       HaloModel orchestrator
    mass_definition.py  Spherical-overdensity mass definitions
    massfunc.py         Halo mass functions
    bias.py             Halo bias models
    concentration.py    Concentration-mass relations
    profiles/
      base_profile.py   HaloProfile base class and Hankel transform
      pressure.py       Pressure profiles (GNFW, ParametricGNFW, B12)
      matter.py         Matter profiles (NFW)
      density.py        Density profiles (NFW, BCM, B16)
      cib.py            CIB luminosity profiles
      hod.py            HOD galaxy profiles
  tracers/
    base_tracer.py      Tracer base class
    tsz.py              Thermal SZ tracer
    ksz.py              Kinematic SZ tracer
    cmb_lensing.py      CMB lensing tracer
    cib.py              CIB tracer
    galaxy_hod.py       Galaxy HOD tracer
    galaxy_lensing.py   Galaxy lensing tracer
```

## Authors

Patrick Janulewicz (PJ), Licong Xu (LX), Boris Bolliet (BB)

## License

MIT
