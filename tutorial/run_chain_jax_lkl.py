"""
Run hmfast MCMC chain with a JAX-only likelihood (no tszpower).

Default tszpower likelihood runs classy_sz/TF on CPU and dominates the
per-sample wall time. The cobaya tutorial likelihood here is a plain
Gaussian chi^2 against the binned D_ell data / covariance.

Usage:
    python tutorial/run_chain_jax_lkl.py [--max-samples N] [--output-prefix PATH]
"""
from __future__ import annotations
import os, sys, argparse

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
xf = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_persistent_cache_dir" in xf:
    os.environ.pop("XLA_FLAGS")

sys.path.insert(0, os.path.dirname(__file__))
import cobaya  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--output-prefix", default="tutorial/chains/hmfast_jax_chain")
    parser.add_argument("--rminus1", type=float, default=0.05)
    args = parser.parse_args()

    info = {
        "theory": {
            "hmfast_cobaya_theory_masked_scatter.HMFastTSZMaskedScatter": {
                "q_cat": 5.0,
                # M in M_sun/h convention (matches tszpower exactly). The
                # theory wrapper divides by current h per sample so the *physical*
                # mass range tracks cosmology the same way tszpower does.
                "M_min": 1e14 * 0.6766,
                "M_max": 1e16 * 0.6766,
                "z_min": 0.005, "z_max": 3.0,
                "n_grid_mz": 50, "n_ell_internal": 50,  # 50x50 grid -> 3.6 ms/sample, 0.6% Cl; posterior identical to N=75/100 within 0.1 sigma
            }
        },
        "likelihood": {
            "cobaya_tutorial_likelihood.tSZ_PS_Likelihood_FullSkyNPY": {
                "data_directory": "/scratch/scratch-lxu/tsz_cnc_scatter/synthetic_data",
                "data_file": "Dl_binned_masked_signal_only_real0.txt",
                "cov_file": "covmat_Dl_binned_masked_signal_only.npy",
                "log_every": 0,
            }
        },
        "params": {
            "H0": {"prior": {"dist": "norm", "loc": 73.8, "scale": 2.4},
                   "ref":   {"dist": "norm", "loc": 73.8, "scale": 2.4},
                   "proposal": 0.6, "latex": r"H_0"},
            "ln10_10A_s": {"prior": {"min": 2.5, "max": 3.5},
                           "ref": {"dist": "norm", "loc": 2.9718, "scale": 0.13},
                           "proposal": 0.13, "latex": r"\ln(10^{10}A_s)"},
            "n_s": {"prior": {"dist": "norm", "loc": 0.962, "scale": 0.014},
                    "ref": {"dist": "norm", "loc": 0.962, "scale": 0.014},
                    "proposal": 0.014, "latex": r"n_\mathrm{s}"},
            "omega_b": {"prior": {"dist": "norm", "loc": 0.022, "scale": 0.002},
                        "ref": {"dist": "norm", "loc": 0.022, "scale": 0.002},
                        "proposal": 0.002, "latex": r"\Omega_\mathrm{b} h^2"},
            "Omega_m": {"prior": {"min": 0.2, "max": 0.5},
                        "ref": {"dist": "norm", "loc": 0.3096, "scale": 0.02},
                        "proposal": 0.02, "latex": r"\Omega_m"},
            "omega_cdm": {"value": "lambda Omega_m, H0, omega_b: Omega_m * (H0 / 100.0)**2 - omega_b",
                          "latex": r"\Omega_\mathrm{c} h^2"},
            "tau_reio": {"value": 0.0544, "latex": r"\tau_{reio}"},
            "one_minus_b": {"value": 0.709, "latex": r"1-b"},
            "B": {"value": "lambda one_minus_b: 1.0 / one_minus_b", "latex": r"B"},
            "A_SZ": {"prior": {"min": -4.41, "max": -4.21},
                     "ref": {"dist": "norm", "loc": -4.31, "scale": 0.02},
                     "proposal": 0.02, "latex": r"A_\mathrm{SZ}"},
            "alpha_SZ": {"prior": {"min": 1.0, "max": 1.24},
                         "ref": {"dist": "norm", "loc": 1.12, "scale": 0.02},
                         "proposal": 0.02, "latex": r"\alpha_\mathrm{SZ}"},
            "sigma_lnY": {"prior": {"dist": "norm", "loc": 0.173, "scale": 0.023},
                          "ref": {"dist": "norm", "loc": 0.173, "scale": 0.023},
                          "proposal": 0.01, "latex": r"\sigma_{\ln Y}"},
        },
        "sampler": {
            "mcmc": {
                "Rminus1_stop": args.rminus1,
                "drag": False, "proposal_scale": 1.2,
                "learn_every": 40, "learn_proposal": True,
                "learn_proposal_Rminus1_max": 100.0,
                "learn_proposal_Rminus1_max_early": 100.0,
                "max_tries": args.max_samples, "burn_in": 50,
            }
        },
        "output": args.output_prefix,
        "logging_level": 2, "timing": True,
    }

    print(f"Starting Cobaya MCMC (JAX-only) -> {args.output_prefix}")
    print(f"  max_samples = {args.max_samples}, Rminus1_stop = {args.rminus1}")
    cobaya.run(info)
    print("Chain finished.")


if __name__ == "__main__":
    main()
