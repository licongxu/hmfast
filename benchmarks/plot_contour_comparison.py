"""
Compare hmfast MCMC contour to tszpower's reference chain.

tszpower chain: /scratch/scratch-lxu/tsz_cnc_scatter/chains/chains_tszpower_scatter_masked_signal_only/tszpower_scatter_masked_signal_only_chain.*.txt
hmfast chain:   tutorial/chains/hmfast_jax_chain.*.txt

Generates a getdist triangle plot of the shared parameter subset.
"""
from __future__ import annotations
import os
import sys
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
})
import getdist
from getdist import plots, MCSamples
import numpy as np

HMFAST_PREFIX = "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/tutorial/chains/hmfast_jax_chain_Mh"
TSZP_PREFIX = "/scratch/scratch-lxu/tsz_cnc_scatter/chains/chains_tszpower_scatter_masked_signal_only/tszpower_scatter_masked_signal_only_chain"

OUT_DIR = "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/benchmarks/results"
os.makedirs(OUT_DIR, exist_ok=True)


def load_chain(prefix):
    samples = getdist.loadMCSamples(prefix, settings={"ignore_rows": 0.3})
    print(f"Loaded {prefix}: {samples.numrows} samples, params: "
          f"{[p.name for p in samples.getParamNames().names]}")
    return samples


def main():
    hmf = load_chain(HMFAST_PREFIX)
    tsz = load_chain(TSZP_PREFIX)

    # Common parameters to plot (must exist in both chains)
    common = [p.name for p in hmf.getParamNames().names
              if p.name in [q.name for q in tsz.getParamNames().names]
              and p.name not in ("weight", "minuslogprior", "minuslogpost",
                                 "chi2", "minuslogprior__0", "chi2__tszpower_scatter_masked_signal_only")]
    print("Common parameters:", common)

    # Filter to the cosmology + scaling-relation subset most likely to be shared
    plot_pars = [p for p in
                 ["H0", "Omega_m", "ln10_10A_s", "n_s", "A_SZ", "alpha_SZ", "sigma_lnY"]
                 if p in common]
    print("Plotting:", plot_pars)
    if len(plot_pars) < 2:
        sys.exit("Need at least 2 common parameters to plot.")

    g = plots.get_subplot_plotter(width_inch=12)
    g.settings.figure_legend_loc = "upper right"
    g.settings.legend_fontsize = 16
    g.settings.lab_fontsize = 16
    g.settings.axes_fontsize = 12
    g.settings.linewidth_contour = 1.6
    g.settings.alpha_filled_add = 0.5
    g.triangle_plot(
        [tsz, hmf], plot_pars,
        legend_labels=[r"\texttt{tszpower}", r"\texttt{hmfast}"],
        contour_colors=["tab:orange", "tab:blue"],
        filled=True,
    )

    pdf_path = os.path.join(OUT_DIR, "hmfast_vs_tszpower_contours.pdf")
    png_path = os.path.join(OUT_DIR, "hmfast_vs_tszpower_contours.png")
    g.export(pdf_path)
    g.export(png_path, dpi=300)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")

    # Numerical mean/std comparison
    print("\nMean +/- std (hmfast | tszpower):")
    for p in plot_pars:
        h_mean = hmf.mean(p);   h_std = hmf.std(p)
        t_mean = tsz.mean(p);   t_std = tsz.std(p)
        d_mean = (h_mean - t_mean) / t_std
        print(f"  {p:>15s}: {h_mean:8.4f} +/- {h_std:.4f}  |  "
              f"{t_mean:8.4f} +/- {t_std:.4f}  |  delta_mean = {d_mean:+.2f} sigma_tszp")


if __name__ == "__main__":
    main()
