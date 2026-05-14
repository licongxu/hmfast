"""
Selection-function utilities for the unresolved (masked) tSZ angular
power spectrum.

This module mirrors the SNR construction used by ``tszpower`` so that the
hmfast masked tSZ power-spectrum is numerically consistent with
``tszpower.maskedpower.compute_integral_snr_simple_uRC`` for the same
catalogue threshold :math:`q_{\\rm cat}`, scaling-relation parameters
(:math:`A_{\\rm SZ}, \\alpha_{\\rm SZ}, B`), and Planck-collaboration
SZiFi noise curves.

The mask is built on a :math:`(M, z)` grid in three steps:

1. **Compton-:math:`y_0` amplitude.** The parametric form

   .. math::

       y_0^{\\rm param}(M, z) = 10^{A_{\\rm SZ}}
           \\left(\\frac{M_{500c} h / B}{0.7 \\times 3 \\times 10^{14}}\\right)^{\\alpha_{\\rm SZ}}
           E(z)^2 \\left(\\frac{h}{0.7}\\right)^{-1/2}

   is used. Halo masses are first converted from the halo-model's native
   mass definition to :math:`M_{500c}`. When the input mass is already
   :math:`M_{500c}` (e.g. ``MassDefinition(500, "critical")``), the
   conversion is a no-op.

2. **Cluster angular size.** The Planck convention

   .. math::

       \\theta_{500}(M, z) = 6.997'
            \\left(\\frac{h}{0.7}\\right)^{-2/3}
            \\left(\\frac{M_{500c}/B}{h \\cdot 3 \\times 10^{14} M_\\odot}\\right)^{1/3}
            E(z)^{-2/3}
            \\left(\\frac{D_A}{500\\,\\mathrm{Mpc}}\\right)^{-1}

   reproduces ``tszpower.maskedpower.compute_theta500_arcmin``.

3. **Noise curve.** The Planck-collaboration SZiFi
   :math:`\\sigma_{y_0}(\\theta_{500})` curve is loaded from disk
   (sky-fraction-weighted, log-log polynomial fit, default ``deg=3`` to
   match ``cosmocnc``). The SNR is :math:`q = y_0 / \\sigma_{y_0}` and
   the mask is :math:`\\mathcal{M}(M, z) = \\Theta(q_{\\rm cat} - q)`,
   i.e. it keeps undetected (un-masked, "unresolved") halos.

Public API
----------
``compute_y0_parametric(halo_model, m, z, A_SZ, alpha_SZ, B)``
    Parametric :math:`y_0(M, z)`.
``compute_theta500_arcmin(halo_model, m, z, B)``
    Angular cluster size in arcmin.
``load_sigma_y0_curve(sigma_obj_file, skyfr_file, ...)``
    Load and return the sky-averaged :math:`\\log\\sigma_{y_0}` polynomial
    fit coefficients and :math:`\\log\\theta` range.
``sigma_y0_from_theta(theta_arcmin, coeff)``
    Evaluate the polynomial fit.
``build_snr_grid(halo_model, m, z, A_SZ, alpha_SZ, B, ...)``
    Build the SNR field on a :math:`(M, z)` grid.
``snr_mask(snr_grid, q_cat)``
    Heaviside selection :math:`\\Theta(q_{\\rm cat} - q)`.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp

from hmfast.halos.mass_definition import MassDefinition, convert_m_delta


# ----------------------------------------------------------------------
# Path defaults — the Planck/cosmocnc-style SZiFi noise files live under
# the tszpower scratch directory by default; users can override per call.
# ----------------------------------------------------------------------
DEFAULT_SIGMA_OBJ_FILE = "/scratch/scratch-lxu/tszsbi/noise_files/sigma_dict_szifi.npy"
DEFAULT_SKYFR_FILE = "/scratch/scratch-lxu/tszsbi/noise_files/skyfracs_szifi_cosmology.npy"


# ----------------------------------------------------------------------
# Parametric y0 (matches tszpower.parametric_profile.compute_y0_parametric)
# ----------------------------------------------------------------------
def compute_y0_parametric(halo_model, m, z, A_SZ, alpha_SZ, B):
    """Parametric Compton-:math:`y_0` amplitude.

    Parameters
    ----------
    halo_model : HaloModel
        Halo model providing the cosmology and mass-definition conversion.
    m : array
        Halo mass grid in physical :math:`M_\\odot` (halo-model native).
    z : array
        Redshift grid.
    A_SZ : float
        :math:`\\log_{10}` amplitude.
    alpha_SZ : float
        Mass-scaling exponent.
    B : float
        Hydrostatic mass-bias factor (Planck :math:`B = 1/(1-b)`).

    Returns
    -------
    y0 : array
        Compton-:math:`y_0` with shape :math:`(N_m, N_z)`.
    """
    m = jnp.atleast_1d(m)
    z = jnp.atleast_1d(z)

    H0 = halo_model.cosmology.H0
    h = H0 / 100.0

    # Convert mass from native definition to M_500c (no-op when already 500c).
    mass_def_500c = MassDefinition(500, "critical")
    c_old = halo_model.concentration.c_delta(halo_model, m, z)
    m500c = convert_m_delta(
        halo_model.cosmology, m, z,
        halo_model.mass_definition, mass_def_500c, c_old=c_old,
    )  # (Nm, Nz)

    H_z = jnp.atleast_1d(halo_model.cosmology.hubble_parameter(z))[None, :]  # (1, Nz)
    E_z = H_z / H0  # dimensionless

    m_tilde = (m500c * h / B) / (0.7 * 3.0e14)  # (Nm, Nz), in 0.7*3e14 M_sun units
    y0 = (10.0 ** A_SZ) * (m_tilde ** alpha_SZ) * (E_z ** 2) * ((h / 0.7) ** (-0.5))
    return y0


def compute_theta500_arcmin(halo_model, m, z, B):
    r"""Cluster angular size in arcmin.

    Uses the Planck/tszpower convention with
    :math:`\theta_\star = 6.997'`, :math:`M_{\rm true} = M_{500c} / B`.
    """
    m = jnp.atleast_1d(m)
    z = jnp.atleast_1d(z)

    H0 = halo_model.cosmology.H0
    h = H0 / 100.0

    mass_def_500c = MassDefinition(500, "critical")
    c_old = halo_model.concentration.c_delta(halo_model, m, z)
    m500c = convert_m_delta(
        halo_model.cosmology, m, z,
        halo_model.mass_definition, mass_def_500c, c_old=c_old,
    )

    M_true = m500c / B  # (Nm, Nz), in M_sun
    H_z = jnp.atleast_1d(halo_model.cosmology.hubble_parameter(z))[None, :]
    E_z = H_z / H0
    D_A = jnp.atleast_1d(halo_model.cosmology.angular_diameter_distance(z))[None, :]  # Mpc

    theta_star = 6.997  # arcmin
    theta_500 = (
        theta_star
        * (h / 0.7) ** (-2.0 / 3.0)
        * (M_true / (h * 3.0e14)) ** (1.0 / 3.0)
        * E_z ** (-2.0 / 3.0)
        * (D_A / 500.0) ** (-1.0)
    )
    return theta_500


# ----------------------------------------------------------------------
# Noise curve: log-log polynomial fit of the sky-averaged sigma_y0(theta)
# ----------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_sigma_skyavg_cached(
    sigma_obj_file: str,
    skyfr_file: str,
    filter_name: str,
    theta_min: float,
    theta_max: float,
):
    """Load and average the Planck SZiFi noise curve over sky fractions.

    Returns a numpy ``(theta_grid_arcmin, sigma_skyavg)`` tuple. Cached so
    repeated calls (e.g. inside a Cobaya theory) hit the disk once.
    """
    sigma_obj = np.load(sigma_obj_file, allow_pickle=True).item()
    skyfr = np.asarray(np.load(skyfr_file)).ravel()

    data = sigma_obj[filter_name]
    first = next(iter(data.values()))
    n_theta = len(first)

    theta_grid = np.exp(np.linspace(np.log(theta_min), np.log(theta_max), n_theta))

    num = np.zeros(n_theta, dtype=np.float64)
    den = 0.0
    for tile_idx, arr in data.items():
        w = float(skyfr[int(tile_idx)])
        num += w * np.asarray(arr, dtype=np.float64)
        den += w
    sigma_skyavg = num / max(den, 1e-300)
    return theta_grid, sigma_skyavg


def load_sigma_y0_curve(
    sigma_obj_file: str = DEFAULT_SIGMA_OBJ_FILE,
    skyfr_file: str = DEFAULT_SKYFR_FILE,
    filter_name: str = "immf6",
    theta_min: float = 0.5,
    theta_max: float = 32.0,
    poly_deg: int = 3,
):
    r"""Return polynomial coefficients of :math:`\log\sigma_{y_0}(\log\theta_{500})`.

    The polynomial is fit on the sky-fraction-weighted average across all
    SZiFi tiles. ``poly_deg=3`` matches the ``cosmocnc`` / ``tszpower``
    default (``compute_sigma_y0`` polynomial fit).

    Returns
    -------
    coeff : np.ndarray
        Coefficients (highest power first), suitable for
        :func:`sigma_y0_from_theta`.
    log_theta_range : tuple of float
        Min and max of :math:`\log\theta_{500}` covered by the input
        sigma curve (in arcmin). Useful for diagnostics; not enforced.
    """
    theta_grid, sigma_skyavg = _load_sigma_skyavg_cached(
        sigma_obj_file, skyfr_file, filter_name, theta_min, theta_max,
    )

    x = np.log(theta_grid)
    eps = 1e-20
    y = np.log(np.clip(sigma_skyavg, eps, None))
    coeff = np.polyfit(x, y, deg=poly_deg)
    return coeff, (float(x.min()), float(x.max()))


def sigma_y0_from_theta(theta_arcmin, coeff):
    """Evaluate the log-log polynomial fit for :math:`\\sigma_{y_0}`.

    Parameters
    ----------
    theta_arcmin : array
        Cluster angular size :math:`\\theta_{500}` in arcmin.
    coeff : array
        Polynomial coefficients (highest power first), as returned by
        :func:`load_sigma_y0_curve`.
    """
    log_theta = jnp.log(jnp.asarray(theta_arcmin))
    coeff = jnp.asarray(coeff)

    out = jnp.zeros_like(log_theta)
    for ck in coeff:
        out = out * log_theta + ck
    return jnp.exp(out)


# ----------------------------------------------------------------------
# SNR grid + Heaviside selection
# ----------------------------------------------------------------------
def build_snr_grid(
    halo_model,
    m,
    z,
    A_SZ: float,
    alpha_SZ: float,
    B: float,
    *,
    sigma_obj_file: str = DEFAULT_SIGMA_OBJ_FILE,
    skyfr_file: str = DEFAULT_SKYFR_FILE,
    filter_name: str = "immf6",
    theta_min: float = 0.5,
    theta_max: float = 32.0,
    poly_deg: int = 3,
    coeff=None,
):
    r"""Construct the cluster SNR grid :math:`q(M, z) = y_0 / \sigma_{y_0}`.

    Parameters
    ----------
    halo_model : HaloModel
    m, z : array
        Halo-mass and redshift grids.
    A_SZ, alpha_SZ, B : float
        Pressure-profile / scaling-relation parameters; see
        :func:`compute_y0_parametric`.
    coeff : array or None
        Precomputed polynomial coefficients for
        :math:`\log\sigma_{y_0}(\log\theta)`. If ``None`` they are
        (re)loaded from disk via :func:`load_sigma_y0_curve`.

    Returns
    -------
    snr_grid : array
        SNR with shape :math:`(N_m, N_z)`.
    """
    if coeff is None:
        coeff, _ = load_sigma_y0_curve(
            sigma_obj_file=sigma_obj_file,
            skyfr_file=skyfr_file,
            filter_name=filter_name,
            theta_min=theta_min, theta_max=theta_max,
            poly_deg=poly_deg,
        )

    y0 = compute_y0_parametric(halo_model, m, z, A_SZ, alpha_SZ, B)        # (Nm, Nz)
    theta500 = compute_theta500_arcmin(halo_model, m, z, B)                # (Nm, Nz)
    sigma = sigma_y0_from_theta(theta500, coeff)                            # (Nm, Nz)
    return y0 / sigma


def snr_mask(snr_grid, q_cat: float, at: float = 0.0):
    r"""Heaviside selection :math:`\Theta(q_{\rm cat} - \mathrm{SNR})`.

    Keeps halos *below* the catalogue threshold, i.e. the unresolved
    component that survives in a masked tSZ map.
    """
    q = jnp.asarray(q_cat, dtype=snr_grid.dtype)
    return jnp.heaviside(q - snr_grid, at)


# ----------------------------------------------------------------------
# Log-normal intrinsic scatter completeness
# ----------------------------------------------------------------------
from functools import partial


@partial(jax.jit, static_argnames=("n_grid", "nsig", "n_power"))
def conditional_An_undetected(snr_grid, sigma_lnY: float, q_cat: float,
                              *, n_power: int = 2, n_grid: int = 1024,
                              nsig: float = 8.0):
    r"""Conditional moment :math:`\langle A^n \mathbf{1}(q_{\rm obs} < q_{\rm cat}) \rangle`.

    When the y-profile amplitude has lognormal scatter
    :math:`\ln A \sim \mathcal{N}(0, \sigma_{\ln Y}^2)`, the masked 1-halo
    integrand that goes as :math:`A^n` must be weighted by this
    conditional moment rather than :math:`\Theta(q_{\rm cat} - q)`.

    Use ``n_power=2`` for the masked power spectrum and ``n_power=4`` for
    the masked trispectrum.

    .. math::

        \langle A^n \mathbf{1}(q_{\rm obs} < q_{\rm cat}) \rangle
        = \frac{1}{\sqrt{2\pi}} \int du\, e^{-u^2/2}\, e^{n \sigma_{\ln Y} u}
          \, \Phi(q_{\rm cat} - e^{\sigma_{\ln Y} u} \bar q)

    where :math:`u = \ln A / \sigma_{\ln Y}` and :math:`\Phi` is the
    standard-normal CDF. This matches
    ``tszpower.parametric_profile._conditional_An_undetected_grid``.

    Parameters
    ----------
    snr_grid : array
        Theory SNR :math:`\bar q(M, z)`, shape :math:`(N_m, N_z)`.
    sigma_lnY : float
        Intrinsic log-normal scatter width.
    q_cat : float
        Catalogue threshold.
    n_power : int
        Moment order (2 for PS, 4 for trispectrum).
    n_grid : int
        Number of quadrature points for the Gauss-Hermite integral.
    nsig : float
        Integration half-width in units of sigma_lnY.

    Returns
    -------
    cond_An : array
        Same shape as ``snr_grid``.
    """
    import jax.scipy.special as jsp

    u = jnp.linspace(-nsig, nsig, n_grid)
    du = u[1] - u[0]
    gauss = jnp.exp(-0.5 * u * u)
    A_n = jnp.exp(n_power * sigma_lnY * u)

    flat = snr_grid.reshape(-1)
    mu = jnp.log(jnp.maximum(flat, 1e-30))                # (N,)
    q_m = jnp.exp(mu[:, None] + sigma_lnY * u[None, :])   # (N, n_grid)
    Phi = 0.5 * (1.0 + jsp.erf((q_cat - q_m) / jnp.sqrt(2.0)))

    integrand = gauss[None, :] * A_n[None, :] * Phi       # (N, n_grid)
    raw = jnp.trapezoid(integrand, dx=du, axis=1)
    cond_flat = raw / jnp.sqrt(2.0 * jnp.pi)
    return cond_flat.reshape(snr_grid.shape)


__all__ = [
    "DEFAULT_SIGMA_OBJ_FILE",
    "DEFAULT_SKYFR_FILE",
    "compute_y0_parametric",
    "compute_theta500_arcmin",
    "load_sigma_y0_curve",
    "sigma_y0_from_theta",
    "build_snr_grid",
    "snr_mask",
    "conditional_An_undetected",
]
