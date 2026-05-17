"""
Triangle plot overlaying three chains:
  - tszpower (reference, 22565 samples, R-1<0.05)
  - hmfast at R-1=0.05 (the previous "converged" chain)
  - hmfast at R-1=0.01 (this experiment's chain)

To determine whether the residual ~0.5-0.8 sigma shifts on S_8 / H_0 are
from chain non-convergence or from a real theory bias.
"""
from __future__ import annotations
import os
import numpy as np

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

import jax
jax.config.update("jax_enable_x64", True)
from getdist import plots, loadMCSamples
from hmfast.cosmology import Cosmology

HMF_OLD = "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/tutorial/chains/archive/hmfast_jax_chain_R01"
HMF_MH  = "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/tutorial/chains/hmfast_jax_chain_Mh"
TSZP    = "/scratch/scratch-lxu/tsz_cnc_scatter/chains/chains_tszpower_scatter_masked_signal_only/tszpower_scatter_masked_signal_only_chain"

OUT_DIR = "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/benchmarks/results"
ONE_MINUS_B = 0.709


def add_derived(samples, label):
    p = samples.getParams()
    H0 = np.asarray(p.H0); omega_cdm = np.asarray(p.omega_cdm); omega_b = np.asarray(p.omega_b)
    n_s = np.asarray(p.n_s); ln10A = np.asarray(p.ln10_10A_s); Omega_m = np.asarray(p.Omega_m)
    cosmo0 = Cosmology(emulator_set="lcdm:v1").update(
        H0=float(H0[0]), omega_cdm=float(omega_cdm[0]), omega_b=float(omega_b[0]),
        n_s=float(n_s[0]), ln1e10A_s=float(ln10A[0]), tau_reio=0.0544)
    sigma8 = np.empty_like(H0)
    print(f"[{label}] computing sigma8 for {len(H0)} samples ...")
    for i in range(len(H0)):
        c = cosmo0.update(H0=float(H0[i]), omega_cdm=float(omega_cdm[i]),
                          omega_b=float(omega_b[i]), n_s=float(n_s[i]),
                          ln1e10A_s=float(ln10A[i]))
        sigma8[i] = float(c.sigma8(0.0))
    samples.addDerived(sigma8, name="sigma8", label=r"\sigma_8")
    p = samples.getParams()
    samples.addDerived(np.asarray(p.sigma8) * np.sqrt(Omega_m / 0.3), name="S8", label=r"S_8")
    samples.addDerived(np.asarray(p.sigma8) * (Omega_m * ONE_MINUS_B)**0.40 * (H0/100.0)**(-0.21),
                       name="F", label=r"F")


def main():
    tsz = loadMCSamples(TSZP, settings={"ignore_rows": 0.3})
    r05 = loadMCSamples(HMF_OLD, settings={"ignore_rows": 0.3})
    rmh = loadMCSamples(HMF_MH, settings={"ignore_rows": 0.3})
    print(f"tszpower : {tsz.numrows} samples")
    print(f"hmfast OLD (fixed phys M) : {r05.numrows} samples")
    print(f"hmfast M_sun/h fix : {rmh.numrows} samples")

    add_derived(tsz, "tszpower")
    add_derived(r05, "hmf R05")
    add_derived(rmh, "hmf M_sun/h fix")

    params = ["Omega_m", "sigma8", "S8", "F", "A_SZ", "alpha_SZ", "sigma_lnY"]
    truth = {
        "Omega_m":   0.309576,
        "sigma8":    0.78,
        "S8":        0.78 * np.sqrt(0.309576 / 0.3),
        "F":         0.78 * (0.309576 * ONE_MINUS_B) ** 0.40 * (0.6766) ** (-0.21),
        "A_SZ":     -4.2373, "alpha_SZ":  1.12, "sigma_lnY": 0.173,
    }

    g = plots.get_subplot_plotter(width_inch=14)
    g.settings.lab_fontsize = 18
    g.settings.axes_fontsize = 14
    g.settings.legend_fontsize = 16
    g.settings.linewidth_contour = 1.5
    g.settings.alpha_filled_add = 0.45

    g.triangle_plot(
        [tsz, r05, rmh], params,
        legend_labels=[r"\texttt{tszpower}",
                       r"\texttt{hmfast} (M fixed phys.)",
                       r"\texttt{hmfast} (M$_\odot$/h fix, tracks h)"],
        contour_colors=["tab:orange", "tab:blue", "tab:green"],
        filled=True,
    )
    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            ax = g.subplots[i, j]
            if ax is None: continue
            if i == j:
                ax.axvline(truth[p1], ls="--", color="red", lw=1.2, zorder=4)
            else:
                ax.axvline(truth[p2], ls="--", color="red", lw=1.2, zorder=4)
                ax.axhline(truth[p1], ls="--", color="red", lw=1.2, zorder=4)

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = os.path.join(OUT_DIR, "hmfast_Msunh_fix_contours.pdf")
    png = os.path.join(OUT_DIR, "hmfast_Msunh_fix_contours.png")
    g.export(pdf); g.export(png, dpi=300)
    print(f"Saved {pdf}\nSaved {png}")

    print("\n  param      tszp                OLD (fixed)         M_sun/h FIX")
    for p in params:
        tm, ts = tsz.mean(p), tsz.std(p)
        a, b = r05.mean(p), r05.std(p)
        c, d = rmh.mean(p), rmh.std(p)
        print(f"  {p:>9s}: {tm:7.4f}+/-{ts:.4f}  {a:7.4f}+/-{b:.4f}  {c:7.4f}+/-{d:.4f}  "
              f"d(OLD-tszp)={(a-tm)/ts:+.2f}s d(Mh-tszp)={(c-tm)/ts:+.2f}s")


if __name__ == "__main__":
    main()
