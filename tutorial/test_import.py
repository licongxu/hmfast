#!/usr/bin/env python
"""Test that hmfast + tszpower can coexist in cobaya."""
import os
import sys

# Step 1: Hide GPUs from TF, import tszpower first
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/scratch/scratch-lxu/tszsbi/tszpower")

print("Step 1: Importing tszpower (TF on CPU)...")
import tszpower.tszpower_cobaya_likelihood_masked  # noqa: F401
print("  OK")

# Step 2: Show GPU to JAX
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
xf = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_persistent_cache_dir" in xf:
    os.environ.pop("XLA_FLAGS")

print("Step 2: Importing hmfast (JAX on GPU)...")
import hmfast  # noqa: F401
from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
print("  OK")

print("Step 3: Testing cobaya import...")
import cobaya  # noqa: F401
print("  OK")

print("Step 4: JAX device check...")
import jax
print(f"  Devices: {jax.devices()}")

print("\nAll imports successful!")
