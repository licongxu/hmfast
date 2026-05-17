"""
Triangle plot in derived parameters:
  Omega_m, sigma_8, S_8, F, A_SZ, alpha_SZ, sigma_lnY
matching the reference tszpower paper plot
(/scratch/scratch-lxu/tsz_cnc_paper_plots/masked_signal_only_posterior_contour.png).

Derives sigma_8 sample-by-sample via hmfast.Cosmology (machine-equivalent to
tszpower's classy_sz.get_sigma8_and_der). Adds:
    S8 = sigma8 * sqrt(Omega_m / 0.3)
    F  = sigma8 * (Omega_m * (1-b))^0.40 * (H0/100)^(-0.21),
                                              with (1-b) = 0.709.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("XLA_FLAGS", "")
xf = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_persistent_cache_dir" in xf:
    os.environ.pop("XLA_FLAGS")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
})
import matplotlib.pyplot as plt
from getdist import plots, loadMCSamples

from hmfast.cosmology import Cosmology

HMFAST_PREFIX = "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/tutorial/chains/hmfast_jax_chain_Mh"
TSZP_PREFIX = "/scratch/scratch-lxu/tsz_cnc_scatter/chains/chains_tszpower_scatter_masked_signal_only/tszpower_scatter_masked_signal_only_chain"
OUT_DIR = "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/benchmarks/results"

ONE_MINUS_B = 0.709


def add_derived(samples, label=""):
    """Add sigma8, S8, F derived parameters to a getdist MCSamples object."""
    p = samples.getParams()
    H0 = np.asarray(p.H0)
    omega_cdm = np.asarray(p.omega_cdm)
    omega_b = np.asarray(p.omega_b)
    n_s = np.asarray(p.n_s)
    ln10_10A_s = np.asarray(p.ln10_10A_s)
    Omega_m = np.asarray(p.Omega_m)

    # Build a Cosmology that we'll update sample-by-sample for sigma8(z=0).
    print(f"[{label}] computing sigma8 for {len(H0)} samples ...")
    cosmo0 = Cosmology(emulator_set="lcdm:v1").update(
        H0=float(H0[0]), omega_cdm=float(omega_cdm[0]),
        omega_b=float(omega_b[0]), n_s=float(n_s[0]),
        ln1e10A_s=float(ln10_10A_s[0]), tau_reio=0.0544,
    )

    sigma8 = np.empty_like(H0)
    for i in range(len(H0)):
        cosmo = cosmo0.update(
            H0=float(H0[i]), omega_cdm=float(omega_cdm[i]),
            omega_b=float(omega_b[i]), n_s=float(n_s[i]),
            ln1e10A_s=float(ln10_10A_s[i]),
        )
        sigma8[i] = float(cosmo.sigma8(0.0))

    samples.addDerived(sigma8, name="sigma8", label=r"\sigma_8")
    p = samples.getParams()
    S8 = np.asarray(p.sigma8) * np.sqrt(Omega_m / 0.3)
    samples.addDerived(S8, name="S8", label=r"S_8")
    F = (np.asarray(p.sigma8)
         * (Omega_m * ONE_MINUS_B) ** 0.40
         * (H0 / 100.0) ** (-0.21))
    samples.addDerived(F, name="F", label=r"F")
    print(f"[{label}] derived params added: sigma8, S8, F")


def main():
    hmf = loadMCSamples(HMFAST_PREFIX, settings={"ignore_rows": 0.3})
    tsz = loadMCSamples(TSZP_PREFIX, settings={"ignore_rows": 0.3})
    print(f"hmfast: {hmf.numrows} samples; tszpower: {tsz.numrows} samples")

    add_derived(hmf, "hmfast")
    add_derived(tsz, "tszpower")

    params = ["Omega_m", "sigma8", "S8", "F", "A_SZ", "alpha_SZ", "sigma_lnY"]

    # Fiducial truth values, per tszpower paper-plot notebook
    truth = {
        "Omega_m":   0.309576,
        "sigma8":    0.78,
        "S8":        0.78 * np.sqrt(0.309576 / 0.3),
        "F":         0.78 * (0.309576 * ONE_MINUS_B) ** 0.40 * (0.6766) ** (-0.21),
        "A_SZ":     -4.2373,
        "alpha_SZ":  1.12,
        "sigma_lnY": 0.173,
    }
    print("Fiducial truth:", truth)

    g = plots.get_subplot_plotter(width_inch=14)
    g.settings.lab_fontsize = 18
    g.settings.axes_fontsize = 14
    g.settings.legend_fontsize = 18
    g.settings.linewidth_contour = 1.6
    g.settings.alpha_filled_add = 0.5

    g.triangle_plot(
        [tsz, hmf], params,
        legend_labels=[r"\texttt{tszpower}", r"\texttt{hmfast}"],
        contour_colors=["tab:orange", "tab:blue"],
        filled=True,
    )

    # Red dashed truth lines
    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            ax = g.subplots[i, j]
            if ax is None:
                continue
            if i == j:
                ax.axvline(truth[p1], ls="--", color="red", lw=1.3, zorder=4)
            else:
                ax.axvline(truth[p2], ls="--", color="red", lw=1.3, zorder=4)
                ax.axhline(truth[p1], ls="--", color="red", lw=1.3, zorder=4)

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = os.path.join(OUT_DIR, "hmfast_vs_tszpower_contours_derived.pdf")
    png = os.path.join(OUT_DIR, "hmfast_vs_tszpower_contours_derived.png")
    g.export(pdf)
    g.export(png, dpi=300)
    print(f"Saved {pdf}")
    print(f"Saved {png}")

    print("\nMean +/- std (hmfast | tszpower) | delta-mean / sigma_tszp | truth shift / sigma_tszp:")
    for p in params:
        h_mean = hmf.mean(p);   h_std = hmf.std(p)
        t_mean = tsz.mean(p);   t_std = tsz.std(p)
        d_h_t = (h_mean - t_mean) / t_std
        d_t_truth = (t_mean - truth[p]) / t_std
        d_h_truth = (h_mean - truth[p]) / h_std
        print(f"  {p:>10s}: hmfast {h_mean:8.4f} +/- {h_std:.4f}  |  "
              f"tszp {t_mean:8.4f} +/- {t_std:.4f}  |  "
              f"d(h,t) = {d_h_t:+.2f} | t-truth = {d_t_truth:+.2f} | h-truth = {d_h_truth:+.2f}")


if __name__ == "__main__":
    main()
