"""
HMFast: Machine learning accelerated and differentiable halo model code.

This package provides fast, differentiable halo model calculations using JAX
and machine learning emulators for cosmological applications.
"""

__version__ = "0.2.0"
__author__ = "Patrick Janulewicz, Licong Xu, Boris Bolliet"
__email__ = "pj407@cam.ac.uk"


# mcfit enables JAX x64 at import. Lock precision before any hmfast array is built.
import mcfit  # noqa: F401

from .precision import available_backends, backend, configure_precision, set_backend

configure_precision()

from .download import data_path, download_emulators

download_emulators(emulator_set=["lcdm:v1", "ede:v2"], skip_existing=True)


from . import halos
from . import cosmology
from . import tracers


__all__ = [
    "available_backends",
    "backend",
    "cosmology",
    "data_path",
    "download_emulators",
    "halos",
    "set_backend",
    "tracers",
]