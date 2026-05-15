"""
Triangle plot comparing hmfast + tszpower masked tSZ posteriors.
Matches the style of plot_triangle_signal_only_3contours.py with LaTeX.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from getdist.mcsamples import loadMCSamples
from getdist import plots

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 28,
    "axes.labelsize": 28,
    "axes.titlesize": 28,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 32,
    "text.latex.preamble": (
        r"\usepackage[T1]{fontenc}\usepackage{type1cm}"
        r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}"
    ),
})

# ---- Load chains ----
chains_dir = "/scratch/scratch-lxu/tsz_cnc_scatter/chains"

# tszpower masked signal-only chain (reference)
tszpower_root = os.path.join(
    chains_dir,
    "chains_tszpower_scatter_masked_signal_only",
    "tszpower_scatter_masked_signal_only_chain",
)
s_tszpower = loadMCSamples(tszpower_root, settings={"ignore_rows": 0.3})
print(f"tszpower chain: {s_tszpower.numrows} samples")

# hmfast masked signal-only chain
hmfast_root = (
    "benchmarks/results/hmfast_scatter_masked_signal_only_chain"
)
s_hmfast = loadMCSamples(hmfast_root, settings={"ignore_rows": 0.3})
print(f"hmfast chain: {s_hmfast.numrows} samples")

# ---- Add derived parameters (sigma8, S8, F) ----
# Use classy_sz for sigma8 derivation (same as reference)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ["OMP_NUM_THREADS"] = "1"

from classy_sz import Class as Class_sz  # noqa: E402

cosmo_params = {
    "omega_b": 0.02242,
    "omega_cdm": 0.11933,
    "H0": 67.66,
    "tau_reio": 0.0544,
    "ln10^{10}A_s": 2.9718,
    "n_s": 0.9665,
    "B": 1.41,
}
classy_sz = Class_sz()
classy_sz.set(cosmo_params)
classy_sz.set({"jax": 1})
classy_sz.compute_class_szfast()

one_minus_b_const = 0.709


def add_derived(s):
    """Add sigma8, S8, F to a chain sample."""
    p = s.getParams()
    sigma8_vals = np.empty(len(p.H0), dtype=float)
    for i in range(len(p.H0)):
        pdict = {
            "omega_b": float(p.omega_b[i]),
            "omega_cdm": float(p.omega_cdm[i]),
            "H0": float(p.H0[i]),
            "n_s": float(p.n_s[i]),
            "ln10^{10}A_s": float(p.ln10_10A_s[i]),
        }
        sigma8_vals[i] = classy_sz.get_sigma8_and_der(params_values_dict=pdict)[1]
    s.addDerived(sigma8_vals, name="sigma8", label=r"\sigma_8")
    p = s.getParams()
    S8 = p.sigma8 * np.sqrt(p.Omega_m / 0.3)
    s.addDerived(S8, name="S8", label=r"S_8")
    p = s.getParams()
    F = p.sigma8 * (p.Omega_m * one_minus_b_const) ** 0.40 * (p.H0 / 100.0) ** (-0.21)
    s.addDerived(F, name="F", label=r"F")
    return s


s_tszpower = add_derived(s_tszpower)
s_hmfast = add_derived(s_hmfast)

# ---- Triangle plot ----
params_to_plot = [
    "Omega_m", "sigma8", "S8", "F", "A_SZ", "alpha_SZ", "sigma_lnY",
]

samples_list = [s_tszpower, s_hmfast]
legend_labels = [
    r"masked tSZ (tszpower, 27k samples)",
    r"masked tSZ (hmfast, 3.6k samples)",
]
contour_colors = ["darkorange", "royalblue"]

g = plots.get_subplot_plotter(width_inch=18)
g.settings.lab_fontsize = 26
g.settings.axes_fontsize = 20
g.settings.legend_fontsize = 32
g.settings.alpha_filled_add = 0.5

g.triangle_plot(
    samples_list,
    params_to_plot,
    filled=True,
    legend_labels=legend_labels,
    contour_colors=contour_colors,
    legend_loc="upper right",
)

# Move default legend, replace with custom
if g.legend is not None:
    try:
        g.legend.remove()
    except Exception:
        pass

handles = []
for label, color in zip(legend_labels, contour_colors):
    handles.append(
        mpl.patches.Patch(facecolor=color, edgecolor=color, alpha=0.7, label=label)
    )
g.fig.legend(
    handles=handles,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    frameon=True,
    fontsize=32,
    framealpha=0.9,
)

# Fiducial truth lines
truth = {
    "Omega_m": 0.309576,
    "sigma8": 0.78,
    "S8": 0.78 * np.sqrt(0.309576 / 0.3),
    "F": 0.78 * (0.309576 / 1.41) ** 0.40 * (0.6766) ** (-0.21),
    "A_SZ": -4.2373,
    "alpha_SZ": 1.12,
    "sigma_lnY": 0.173,
}
vline_style = dict(ls="--", color="red", lw=1.2, zorder=4)
hline_style = dict(ls="--", color="red", lw=1.2, zorder=4)

for i, p1 in enumerate(params_to_plot):
    for j, p2 in enumerate(params_to_plot):
        ax = g.subplots[i, j]
        if ax is None:
            continue
        if i == j:
            ax.axvline(truth[p1], **vline_style)
        else:
            ax.axvline(truth[p2], **vline_style)
            ax.axhline(truth[p1], **hline_style)

os.makedirs("benchmarks/results", exist_ok=True)
out_png = "benchmarks/results/hmfast_tszpower_posterior_comparison.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved {out_png}")

# Print parameter means
print("\nParameter comparison (mean ± std):")
for p in params_to_plot:
    d_tsz = getattr(s_tszpower.getParams(), p)
    d_hmf = getattr(s_hmfast.getParams(), p)
    print(f"  {p:12s}: tszpower {np.mean(d_tsz):.4f} ± {np.std(d_tsz):.4f}  "
          f"hmfast {np.mean(d_hmf):.4f} ± {np.std(d_hmf):.4f}")
