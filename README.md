# hmfast

**Machine learning accelerated and differentiable halo model code**

Authors: Patrick Janulewicz, Licong Xu, Boris Bolliet

## Installation

```bash
pip install "git+https://github.com/licongxu/hmfast.git"
```

## Quick Start

```python
import hmfast

cosmo = hmfast.Cosmology(
    Omega_c=0.25, 
    Omega_b=0.05, 
    h=0.67, 
    sigma8=0.8, 
    n_s=0.96
)
print(cosmo)
```

## Features
- JAX-based **differentiable** halo model
- ML emulators for fast predictions
- Comprehensive cosmology and tracers support
- Gradient-based inference ready

See `examples/` or docs for more.

## License
Apache-2.0