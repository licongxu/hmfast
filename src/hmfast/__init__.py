"""
HMFast: Machine learning accelerated and differentiable halo model code.

This package provides fast, differentiable halo model calculations using JAX
and machine learning emulators for cosmological applications.
"""

__version__ = "0.1.0"
__author__ = "Patrick Janulewicz, Licong Xu, Boris Bolliet"
__email__ = "pj407@cam.ac.uk"


from .download import download_emulators

download_emulators(models=["lcdm", "ede-v2"], skip_existing=True)


#from .halos import HaloModel
#from .emulator_load import EmulatorLoader, EmulatorLoaderPCA
from .cosmology import Cosmology
from . import tracers


#__all__ = ["HaloModel", "Cosmology"]