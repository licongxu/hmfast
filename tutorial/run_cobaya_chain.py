#!/usr/bin/env python
"""
Run Cobaya MCMC chain with hmfast theory + tszpower likelihood.

Handles TF/JAX GPU coexistence by importing tszpower (TF) first with
GPUs hidden, then hmfast (JAX) with GPU visible.

Usage:
    python tutorial/run_cobaya_chain.py [--max-samples N]
"""
from __future__ import annotations

import os
import sys
import argparse

# Step 1: Hide GPUs from TF, import tszpower likelihood
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, "/scratch/scratch-lxu/tszsbi/tszpower")

# Import tszpower modules that use TF (classy_sz)
import tszpower.tszpower_cobaya_likelihood_masked  # noqa: F401
import tszpower.utils  # noqa: F401

# Step 2: Make GPU visible to JAX and import hmfast
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
# Clear any XLA persistent cache flag
xf = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_persistent_cache_dir" in xf:
    os.environ.pop("XLA_FLAGS")

# Now add hmfast tutorial path and import it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# cobaya import after all environment is set up
import cobaya


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Maximum MCMC samples")
    parser.add_argument("--output-prefix", default="hmfast_scatter_masked_signal_only_chain",
                        help="Output prefix")
    args = parser.parse_args()

    # Construct the info dict matching the YAML
    info = {
        "theory": {
            "hmfast_cobaya_theory_masked_scatter.HMFastTSZMaskedScatter": {
                "q_cat": 5.0,
                "M_min": 6.766e13,
                "M_max": 6.766e15,
                "z_min": 0.005,
                "z_max": 3.0,
                "n_grid_mz": 100,
                "n_ell_internal": 50,
                "n_grid_scatter": 512,
                "nsig_scatter": 8.0,
            }
        },
        "likelihood": {
            "tszpower.tszpower_cobaya_likelihood_masked.tSZ_PS_Likelihood_SignalOnly": {
                "data_directory": "/scratch/scratch-lxu/tsz_cnc_scatter/synthetic_data",
                "data_file": "Dl_binned_masked_signal_only_real0.txt",
                "cov_file": "covmat_Dl_binned_masked_signal_only.txt",
            }
        },
        "params": {
            "H0": {
                "prior": {"dist": "norm", "loc": 73.8, "scale": 2.4},
                "ref": {"dist": "norm", "loc": 73.8, "scale": 2.4},
                "proposal": 0.6,
                "latex": "H_0",
            },
            "ln10_10A_s": {
                "prior": {"min": 2.5, "max": 3.5},
                "ref": {"dist": "norm", "loc": 2.9718, "scale": 0.13},
                "proposal": 0.13,
                "latex": r"\ln(10^{10}A_s)",
            },
            "n_s": {
                "prior": {"dist": "norm", "loc": 0.962, "scale": 0.014},
                "ref": {"dist": "norm", "loc": 0.962, "scale": 0.014},
                "proposal": 0.014,
                "latex": r"n_\mathrm{s}",
            },
            "omega_b": {
                "prior": {"dist": "norm", "loc": 0.022, "scale": 0.002},
                "ref": {"dist": "norm", "loc": 0.022, "scale": 0.002},
                "proposal": 0.002,
                "latex": r"\Omega_\mathrm{b} h^2",
            },
            "Omega_m": {
                "prior": {"min": 0.2, "max": 0.5},
                "ref": {"dist": "norm", "loc": 0.3096, "scale": 0.02},
                "proposal": 0.02,
                "latex": r"\Omega_m",
            },
            "omega_cdm": {
                "value": "lambda Omega_m, H0, omega_b: Omega_m * (H0 / 100.0)**2 - omega_b",
                "latex": r"\Omega_\mathrm{c} h^2",
            },
            "tau_reio": {
                "value": 0.0544,
                "latex": r"\tau_{reio}",
            },
            "one_minus_b": {
                "value": 0.709,
                "latex": "1-b",
            },
            "B": {
                "value": "lambda one_minus_b: 1.0 / one_minus_b",
                "latex": "B",
            },
            "A_SZ": {
                "prior": {"min": -4.41, "max": -4.21},
                "ref": {"dist": "norm", "loc": -4.31, "scale": 0.02},
                "proposal": 0.02,
                "latex": r"A_\mathrm{SZ}",
            },
            "alpha_SZ": {
                "prior": {"min": 1.0, "max": 1.24},
                "ref": {"dist": "norm", "loc": 1.12, "scale": 0.02},
                "proposal": 0.02,
                "latex": r"\alpha_\mathrm{SZ}",
            },
            "sigma_lnY": {
                "prior": {"dist": "norm", "loc": 0.173, "scale": 0.023},
                "ref": {"dist": "norm", "loc": 0.173, "scale": 0.023},
                "proposal": 0.01,
                "latex": r"\sigma_{\ln Y}",
            },
        },
        "sampler": {
            "mcmc": {
                "Rminus1_stop": 0.05,
                "drag": False,
                "proposal_scale": 1.2,
                "learn_every": 40,
                "learn_proposal": True,
                "learn_proposal_Rminus1_max": 100.0,
                "learn_proposal_Rminus1_max_early": 100.0,
                "max_tries": args.max_samples,
                "burn_in": 50,
            }
        },
        "output": args.output_prefix,
        "logging_level": 2,
        "timing": True,
    }

    print("Starting Cobaya MCMC with hmfast theory...")
    print(f"  Output prefix: {args.output_prefix}")
    print(f"  Max samples: {args.max_samples}")

    updated_info, sampler = cobaya.run(info)

    print("Chain finished.")
    print(f"  Output: {args.output_prefix}")


if __name__ == "__main__":
    main()
