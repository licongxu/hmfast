"""Mass-redshift kernels of 1-halo angular power spectra.

Plots :math:`d^2 C_\\ell / (d\\ln z\\, d\\ln M)` for the tSZ Compton-y,
CMB-lensing :math:`\\kappa`, galaxy-lensing :math:`\\gamma`, galaxy-density
:math:`g` autospectra and their crosses. This is the same per-halo weight
that :meth:`hmfast.halos.HaloModel.cl_1h_integrand` integrates to
:meth:`~hmfast.halos.HaloModel.cl_1h_masked` (counterterm omitted, as in the
masked :math:`yy` pipeline).

The overlaid contours are the spherical integrated Compton parameter
:math:`Y_{500}` of the same Arnaud GNFW profile used for :math:`yy`.

Run from the repository root::

    PYTHONPATH=src python tutorial/plot_mz_contributions.py

Figures are written to ``tutorial/figures/`` at 300 dpi with ``text.usetex``.
"""

from __future__ import annotations

import os

os.environ.setdefault("TEXINPUTS", "/home/ubuntu/texmf/tex/latex/type1cm//:")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition, convert_m_delta
from hmfast.halos.profiles import GNFWPressureProfile, NFWMatterProfile, Z07GalaxyHODProfile
from hmfast.tracers import CMBLensingTracer, GalaxyHODTracer, GalaxyLensingTracer, tSZTracer
from hmfast.utils import Const


ELL = np.array([100.0, 1000.0, 5000.0])
# Cluster and group scales. Low-z floor avoids the chi->0 Limber pole.
# Low-z floor has to sit below the ell=100 yy peak; 0.03 clips that blob.
Z = np.geomspace(0.008, 3.0, 48)
M = np.geomspace(1.0e12, 1.0e16, 42)

# Figure groups. Keys are stable ids; labels are the LaTeX row tags.
LENSING_ROWS = [
    ("yy", r"$C_\ell^{yy}$"),
    ("yk", r"$C_\ell^{y\kappa}$"),
    ("ygam", r"$C_\ell^{y\gamma}$"),
    ("kk", r"$C_\ell^{\kappa\kappa}$"),
    ("kgam", r"$C_\ell^{\kappa\gamma}$"),
    ("gamgam", r"$C_\ell^{\gamma\gamma}$"),
]
GALAXY_ROWS = [
    ("yg", r"$C_\ell^{yg}$"),
    ("kg", r"$C_\ell^{\kappa g}$"),
    ("gamg", r"$C_\ell^{\gamma g}$"),
    ("gg", r"$C_\ell^{gg}$"),
]


def _setup_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def build_model():
    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)
    nfw = NFWMatterProfile()
    tracers = {
        "y": tSZTracer(profile=GNFWPressureProfile()),
        "k": CMBLensingTracer(profile=nfw),
        "gam": GalaxyLensingTracer(profile=NFWMatterProfile()),
        "g": GalaxyHODTracer(profile=Z07GalaxyHODProfile()),
    }
    return hm, tracers


def spherical_Y500(hm, m, z, profile):
    """Spherical integrated :math:`Y_{500}` in Mpc^2, shape (Nm, Nz).

    Uses the same GNFW electron pressure as :meth:`GNFWPressureProfile.u_r`
    and integrates :math:`(\\sigma_T/m_e c^2)\\int_{r<R_{500}} P_e\\,4\\pi r^2 dr`.
    ``P_e`` is in eV cm^{-3} (Arnaud's :math:`1.65\\times 10^{-3}` keV cm^{-3}).
    """
    m = jnp.asarray(m)
    z = jnp.asarray(z)
    mass_def_500c = MassDefinition(500, "critical")
    c_old = hm.concentration.c_delta(hm, m, z)
    m500c = convert_m_delta(
        hm.cosmology, m, z, hm.mass_definition, mass_def_500c, c_old=c_old
    )
    r500 = mass_def_500c.r_delta(hm.cosmology, m500c, z)  # physical Mpc, (Nm, Nz)

    r = jnp.geomspace(1.0e-4, 40.0, 256)  # comoving Mpc
    pe = profile.u_r(hm, r, m, z)  # (Nr, Nm, Nz), eV/cm^3

    r_com = r[:, None, None]
    r_phys = r_com / (1.0 + z)[None, None, :]
    inside = r_phys <= r500[None, :, :]
    # dV_phys = 4 pi r_phys^2 dr_phys, dr_phys = dr_com / (1+z)
    integrand = jnp.where(
        inside,
        pe * 4.0 * jnp.pi * r_phys**2 / (1.0 + z)[None, None, :],
        0.0,
    )
    # eV/cm^3 * Mpc^3
    integral = jnp.trapezoid(integrand, x=r, axis=0)

    mpc_cm = Const._Mpc_over_m_ * 100.0
    sigma_t = 6.6524587e-25  # cm^2
    m_e_ev = 510998.95
    y_cm2 = (sigma_t / m_e_ev) * integral * mpc_cm**3
    y_mpc2 = y_cm2 / mpc_cm**2
    return np.asarray(y_mpc2)


def d2_dlnz_dlnm(integrand_dz, z):
    """Convert d^2 C / (dz dlnM) to d^2 C / (dlnz dlnM) = z * d^2 C / dz dlnM.

    ``integrand_dz`` has shape (Nz, Nell, Nm).
    """
    return np.asarray(integrand_dz) * np.asarray(z)[:, None, None]


def compute_kernels(hm, tracers, z, m, ell):
    pairs = {
        "yy": ("y", "y"),
        "yk": ("y", "k"),
        "ygam": ("y", "gam"),
        "yg": ("y", "g"),
        "kk": ("k", "k"),
        "kgam": ("k", "gam"),
        "kg": ("k", "g"),
        "gamgam": ("gam", "gam"),
        "gamg": ("gam", "g"),
        "gg": ("g", "g"),
    }
    z_j = jnp.asarray(z)
    m_j = jnp.asarray(m)
    ell_j = jnp.asarray(ell)
    out = {}
    for key, (a, b) in pairs.items():
        print(f"computing {key}", flush=True)
        t1 = tracers[a]
        t2 = t1 if a == b else tracers[b]
        raw = hm.cl_1h_integrand(t1, t2, ell_j, m_j, z_j, k_damp=0.0)
        out[key] = d2_dlnz_dlnm(raw, z)
        finite = np.isfinite(out[key])
        print(
            f"  finite {finite.mean():.3f}  max {np.nanmax(out[key]):.3e}",
            flush=True,
        )
    return out


def compute_masked_yy(hm, tracers, z, m, ell, y500, log10_cut=-5.0):
    """Unresolved-yy kernel: drop halos with log10(Y_500/Mpc^2) above the cut.

    This is a sharp Compton-Y selection in the same ``mask_mz`` slot used by
    the masked :math:`C_\\ell^{yy}` pipeline. It is not the Planck SNR mask
    (that needs the SZiFi noise curves).
    """
    # mask keeps the undetected population, shape (Nm, Nz)
    logy = np.log10(np.clip(y500, 1e-30, None))
    mask = (logy < log10_cut).astype(np.float64)
    raw = hm.cl_1h_integrand(
        tracers["y"],
        tracers["y"],
        jnp.asarray(ell),
        jnp.asarray(m),
        jnp.asarray(z),
        mask_mz=jnp.asarray(mask),
        k_damp=0.0,
    )
    return d2_dlnz_dlnm(raw, z), mask


def _scale(vmax):
    if not np.isfinite(vmax) or vmax <= 0:
        return 0, 1.0
    exp = int(np.floor(np.log10(vmax)))
    return exp, 10.0**exp


def _draw_panels(fig, axes, rows, kernels, z, m_plot, ell, y500, vmax_override=None):
    logy = np.log10(np.clip(y500, 1e-30, None))
    images = []
    for i, (key, label) in enumerate(rows):
        field = kernels[key]
        for j, ell_j in enumerate(ell):
            ax = axes[i, j]
            data = np.array(field[:, j, :], dtype=np.float64)
            data = np.where(np.isfinite(data), data, 0.0)
            positive = data[data > 0]
            vmax = float(np.max(data)) if positive.size else 1.0
            if vmax_override is not None:
                vmax = vmax_override[i, j]
            exp, scale = _scale(vmax)
            shown = np.clip(data, 0.0, None) / scale
            im = ax.pcolormesh(
                z,
                m_plot,
                shown.T,
                shading="nearest",
                cmap="YlOrRd",
                vmin=0.0,
                vmax=max(vmax / scale, 1e-30),
                rasterized=True,
            )
            # Half-peak contour so the sensitive region is visible, not only the core.
            if positive.size and np.max(data) > 0:
                ax.contour(
                    z,
                    m_plot,
                    (data / np.max(data)).T,
                    levels=[0.5],
                    colors="white",
                    linewidths=0.6,
                )
            ax.contour(
                z,
                m_plot,
                logy,
                levels=[-6.0, -5.0, -4.0],
                colors="k",
                linewidths=0.7,
                linestyles=["--", "-", "-."],
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(z[0], z[-1])
            ax.set_ylim(m_plot[0], m_plot[-1])
            if i == 0:
                ax.set_title(rf"$\ell = {int(ell_j)}$")
            if i == len(rows) - 1:
                ax.set_xlabel(r"$z$")
            if j == 0:
                ax.set_ylabel(r"$M_{200\mathrm{c}}\,[h^{-1}M_\odot]$")
                ax.text(
                    -0.42,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label(rf"$\times 10^{{{exp}}}$")
            cbar.ax.tick_params(labelsize=7)
            images.append(im)
    return images


def _contour_legend(fig):
    handles = [
        Line2D([0], [0], color="k", ls="--", lw=0.9, label=r"$\log_{10}(Y_{500}/\mathrm{Mpc}^{2})=-6$"),
        Line2D([0], [0], color="k", ls="-", lw=0.9, label=r"$\log_{10}(Y_{500}/\mathrm{Mpc}^{2})=-5$"),
        Line2D([0], [0], color="k", ls="-.", lw=0.9, label=r"$\log_{10}(Y_{500}/\mathrm{Mpc}^{2})=-4$"),
        Line2D([0], [0], color="white", ls="-", lw=1.2, label=r"half-peak of $\mathrm{d}^{2}C_\ell$"),
    ]
    # White half-peak line is invisible on the page; draw it dark gray in the legend only.
    handles[-1] = Line2D(
        [0], [0], color="0.45", ls="-", lw=1.0, label=r"half of the panel peak"
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.48, -0.01),
        fontsize=8,
    )


def plot_group(path, rows, kernels, z, m_plot, ell, y500, caption):
    _setup_style()
    nrows = len(rows)
    fig, axes = plt.subplots(
        nrows,
        len(ell),
        sharex=True,
        sharey=True,
        figsize=(10.2, 2.15 * nrows + 0.6),
        constrained_layout=False,
    )
    if nrows == 1:
        axes = np.array([axes])
    _draw_panels(fig, axes, rows, kernels, z, m_plot, ell, y500)
    fig.suptitle(caption, fontsize=11, y=0.995)
    _contour_legend(fig)
    fig.tight_layout(rect=(0.05, 0.055, 0.995, 0.965))
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def plot_masked(path, kernels_full, kernels_masked, z, m_plot, ell, y500):
    _setup_style()
    rows_full = [("yy", r"$C_\ell^{yy}$")]
    rows_masked = [("yy_masked", r"$C_\ell^{yy,\mathrm{unres.}}$")]
    both = {"yy": kernels_full["yy"], "yy_masked": kernels_masked}
    # Share the color scale of each multipole so the masked map shows what was removed.
    vmax = np.zeros((2, len(ell)))
    for j in range(len(ell)):
        peak = float(np.nanmax(kernels_full["yy"][:, j, :]))
        vmax[:, j] = peak
    fig, axes = plt.subplots(
        2, len(ell), sharex=True, sharey=True, figsize=(10.2, 5.2), constrained_layout=False
    )
    _draw_panels(
        fig,
        axes,
        rows_full + rows_masked,
        both,
        z,
        m_plot,
        ell,
        y500,
        vmax_override=vmax,
    )
    fig.suptitle(
        r"Full versus unresolved $yy$: halos with "
        r"$\log_{10}(Y_{500}/\mathrm{Mpc}^{2})>-5$ removed",
        fontsize=11,
        y=0.995,
    )
    _contour_legend(fig)
    fig.tight_layout(rect=(0.06, 0.07, 0.995, 0.95))
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def report_peaks(kernels, z, m_plot, ell):
    print("\nPeak of d^2 C / (d ln z d ln M)")
    print(f"{'spectrum':10s} {'ell':8s} {'z':8s} {'M_200c/h':12s}")
    for key, field in kernels.items():
        for j, ell_j in enumerate(ell):
            data = field[:, j, :]
            data = np.where(np.isfinite(data), data, -np.inf)
            iz, im = np.unravel_index(np.argmax(data), data.shape)
            print(f"{key:10s} {ell_j:8.0f} {z[iz]:8.3f} {m_plot[im]:12.3e}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    art_dir = "/opt/cursor/artifacts"
    os.makedirs(art_dir, exist_ok=True)

    hm, tracers = build_model()
    h = float(hm.cosmology.H0 / 100.0)
    m_plot = M / h

    print("computing Y_500", flush=True)
    y500 = spherical_Y500(hm, M, Z, tracers["y"].profile)
    # Spot-check a cluster near the middle of the grid.
    im = np.argmin(np.abs(M - 3.0e14))
    iz = np.argmin(np.abs(Z - 0.5))
    print(
        f"  Y_500(M={M[im]:.2e}, z={Z[iz]:.2f}) = {y500[im, iz]:.3e} Mpc^2"
        f"  log10 = {np.log10(y500[im, iz]):.2f}",
        flush=True,
    )

    kernels = compute_kernels(hm, tracers, Z, M, ELL)
    kernels["yy_masked"], _mask = compute_masked_yy(hm, tracers, Z, M, ELL, y500)

    np.savez(
        os.path.join(out_dir, "mz_kernels.npz"),
        z=Z,
        m=M,
        m_over_h=m_plot,
        ell=ELL,
        y500=y500,
        **kernels,
    )

    caption = (
        r"1-halo Limber weight $\mathrm{d}^{2}C_\ell/(\mathrm{d}\ln z\,\mathrm{d}\ln M_{200\mathrm{c}})$"
        r" on the default $\Lambda$CDM emulator ($h=%.2f$)" % h
    )
    paths = []
    for name, rows in (
        ("mz_kernels_tsz_lensing.png", LENSING_ROWS),
        ("mz_kernels_galaxies.png", GALAXY_ROWS),
    ):
        path = os.path.join(out_dir, name)
        plot_group(path, rows, kernels, Z, m_plot, ELL, y500, caption)
        paths.append(path)
    masked_path = os.path.join(out_dir, "mz_kernels_yy_masked.png")
    plot_masked(masked_path, kernels, kernels["yy_masked"], Z, m_plot, ELL, y500)
    paths.append(masked_path)

    for path in paths:
        dest = os.path.join(art_dir, os.path.basename(path))
        # copy bytes
        with open(path, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())
        print(f"copied {dest}", flush=True)

    report_peaks(kernels, Z, m_plot, ELL)


if __name__ == "__main__":
    main()
