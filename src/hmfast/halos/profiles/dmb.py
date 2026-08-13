"""
DMB (Dark Matter + Baryon) halo profiles.

Two formula conventions (``convention=`` on the profile / in the param dict):

- ``"godmax"`` (default): ``BCM_18_wP`` in
  ``ref_packages/GODMAX/src/get_BCMP_profile_jit.py`` — gas outer factor
  ``(1+(r/r_ej)^γ)^{(δ-β)/γ}``, stellar amplitude ``A=0.09``, ``P_nt`` on
  ``R_200c`` with slope ``n_nt``, HSE to ``6 R_200c``.
- ``"literature"``: To et al. (2024) eqs. 3.1–3.13 / Dalal et al. (2026)
  eqs. 2.1–2.11 — gas outer ``(1+r/r_ej)^{(δ-β)/γ}``, ``A=0.055``,
  ``M1=2.5×10^{11} h^{-1} M_⊙``, ``P_nt`` To 3.12 on ``R_500c`` with index
  ``0.8``, HSE to the work-grid edge (paper ``∞``).

``n_nt_zcap`` overlays Dalal (2026) eq. 2.12 on either convention.
``ρ_clm`` always includes the ``1/ζ^3`` Jacobian (GODMAX eq. 13); the
printed To 3.7 / Dalal 2.7 omit it.

Internal units: GODMAX (``Msun/h``, comoving ``Mpc/h``) at the kernel,
physical ``Msun`` / comoving ``Mpc`` / ``eV/cm^3`` at the hmfast boundary.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node

from hmfast.halos.mass_definition import MassDefinition, convert_m_delta
from hmfast.halos.profiles.base_profile import HankelTransform
from hmfast.halos.profiles.density import DensityProfile
from hmfast.halos.profiles.matter import MatterProfile
from hmfast.halos.profiles.pressure import PressureProfile
from hmfast.utils import Const

# G in (Msun/h, Mpc/h) → keV/cm^3 (GODMAX / astropy).
_G_KEV = 1.8167909997572036e-30
_PE_KEV_TO_EV = 1000.0
_MU_E = 1.14
_PTH_TO_PE = 1.932
_RHO_CRIT_0_H = 2.77536627245708e11

_PARAM_KEYS = (
    "theta_ej_0",
    "log10_Mstar0_theta_ej",
    "nu_theta_ej_M",
    "nu_theta_ej_z",
    "nu_theta_ej_c",
    "theta_co_0",
    "log10_Mstar0_theta_co",
    "nu_theta_co_M",
    "nu_theta_co_z",
    "nu_theta_co_c",
    "mu_beta",
    "eta_star",
    "eta_cga",
    "A_starcga",
    "log10_M1_starcga",
    "epsilon_rt",
    "log10_Mc0",
    "nu_z",
    "nu_M",
    "log10_Mstar0",
    "a_zeta",
    "n_zeta",
    "alpha_nt",
    "beta_nt",
    "n_nt",
    "n_nt_zcap",
    "gamma_rhogas",
    "delta_rhogas",
    "c200c",
)

_STATIC_KEYS = ("nfw_trunc", "num_points_trapz_int", "convention")

# To 3.2 / Dalal 2.2: A=0.055, M1=2.5e11 h^{-1} M_⊙ (vs GODMAX A=0.09, log10 M1=11.4).
_LITERATURE_DEFAULTS = dict(
    A_starcga=0.055,
    log10_M1_starcga=float(np.log10(2.5e11)),
)

_DEFAULTS = dict(
    theta_ej_0=4.0,
    log10_Mstar0_theta_ej=14.0,
    nu_theta_ej_M=0.0,
    nu_theta_ej_z=0.0,
    nu_theta_ej_c=0.0,
    theta_co_0=0.1,
    log10_Mstar0_theta_co=14.0,
    nu_theta_co_M=0.0,
    nu_theta_co_z=0.0,
    nu_theta_co_c=0.0,
    mu_beta=0.21,
    eta_star=0.3,
    eta_cga=0.6,
    A_starcga=0.09,
    log10_M1_starcga=11.4,
    epsilon_rt=4.0,
    log10_Mc0=14.83,
    nu_z=0.0,
    nu_M=0.0,
    log10_Mstar0=13.0,
    a_zeta=0.3,
    n_zeta=2.0,
    alpha_nt=0.18,
    beta_nt=0.5,
    n_nt=0.3,
    n_nt_zcap=None,
    gamma_rhogas=2.0,
    delta_rhogas=7.0,
    nfw_trunc=True,
    num_points_trapz_int=64,
    c200c=None,
    convention="godmax",
)


def _trapz_mass(f_r, log_r):
    r = jnp.exp(log_r)
    return jnp.trapezoid(f_r * 4.0 * jnp.pi * r**2 * r, x=log_r)


def _cum_mass(rho, r):
    """
    Cumulative enclosed mass ``M(<r)`` via trapezoid on a log-spaced radial grid.

    Replaces repeated ``∫_0^{r_i}`` calls with one ``O(N_r)`` sweep (GPU-friendly).
    The mass interior to the first node uses ``4π/3 r_0^3 ρ_0`` so ``M(r_0)>0``
    (needed for a stable adiabatic-contraction root near the origin).
    """
    log_r = jnp.log(r)
    # dM/dln r = 4π r^3 ρ
    integ = 4.0 * jnp.pi * rho * r**3
    dln = jnp.diff(log_r)
    dM = 0.5 * (integ[:-1] + integ[1:]) * dln
    m_inner = (4.0 / 3.0) * jnp.pi * r[0] ** 3 * rho[0]
    return jnp.concatenate(
        [jnp.array([m_inner], dtype=rho.dtype), m_inner + jnp.cumsum(dM)]
    )


def _reverse_cumtrapz(f, r):
    """``I(r_i) = ∫_{r_i}^{r_max} f(r) dr`` from a single reverse sweep."""
    dr = jnp.diff(r)
    dI = 0.5 * (f[:-1] + f[1:]) * dr
    from_out = jnp.cumsum(dI[::-1])[::-1]
    return jnp.concatenate([from_out, jnp.zeros((1,), dtype=f.dtype)])


def _param_dict(obj):
    d = {k: getattr(obj, k) for k in _PARAM_KEYS}
    d["nfw_trunc"] = obj.nfw_trunc
    d["num_points_trapz_int"] = obj.num_points_trapz_int
    d["convention"] = obj.convention
    return d


def _param_leaves(obj):
    return tuple(getattr(obj, k) for k in _PARAM_KEYS)


def _set_params(obj, leaves_or_dict, static=None):
    if isinstance(leaves_or_dict, dict):
        for k in _PARAM_KEYS:
            setattr(obj, k, leaves_or_dict[k])
        for k in _STATIC_KEYS:
            if k in leaves_or_dict:
                setattr(obj, k, leaves_or_dict[k])
    else:
        for k, v in zip(_PARAM_KEYS, leaves_or_dict):
            setattr(obj, k, v)
        if static is not None:
            obj.nfw_trunc, obj.num_points_trapz_int, obj.convention = static


class _DMBParamsMixin:
    """Shared DMB baryon-feedback parameters (GODMAX ``BCM_18_wP`` defaults)."""

    def _init_dmb_params(self, **kwargs):
        convention = kwargs.get("convention", "godmax")
        if convention not in ("godmax", "literature"):
            raise ValueError(
                f"convention must be 'godmax' or 'literature', got {convention!r}"
            )
        merged = dict(_DEFAULTS)
        if convention == "literature":
            merged.update(_LITERATURE_DEFAULTS)
        for k, v in kwargs.items():
            if k in merged:
                merged[k] = v
        merged["convention"] = convention
        _set_params(self, merged)

    def _merged_update(self, **kwargs):
        out = _param_dict(self)
        for k, v in kwargs.items():
            if k in out:
                out[k] = v
        return out

    def _tree_flatten(self):
        aux = (
            tuple(self._x.tolist()),
            self._hankel,
            bool(self.nfw_trunc),
            int(self.num_points_trapz_int),
            str(self.convention),
        )
        return (_param_leaves(self), aux)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        obj = cls.__new__(cls)
        x_tuple, hankel, nfw_trunc, n_int, convention = aux_data
        _set_params(obj, leaves, static=(nfw_trunc, n_int, convention))
        obj._x = np.array(x_tuple)
        obj._hankel = hankel
        return obj

    def update(self, **kwargs):
        merged = self._merged_update(**kwargs)
        leaves = tuple(merged[k] for k in _PARAM_KEYS)
        _, aux = self._tree_flatten()
        x_tuple, hankel, nfw_trunc, n_int, convention = aux
        nfw_trunc = bool(merged.get("nfw_trunc", nfw_trunc))
        n_int = int(merged.get("num_points_trapz_int", n_int))
        convention = str(merged.get("convention", convention))
        return type(self)._tree_unflatten(
            (x_tuple, hankel, nfw_trunc, n_int, convention), leaves
        )


def _find_r_delta_from_mass(r_h, cum_mass, delta, rho_c_z_h, z=0.0):
    """Comoving ``R_Δ`` from enclosed mass on the work grid.

    Spherical-overdensity uses *physical* radius:
    ``M = Δ ρ_c (4π/3) [r_comov/(1+z)]³``. ``r_h`` / ``cum_mass`` are
    GODMAX internal (Mpc/h, Msun/h). Mean density falls with ``r``, so
    the interpolant is reversed (JAX ``interp`` needs increasing ``xp``).
    """
    a = 1.0 + z
    target = delta * rho_c_z_h * (4.0 * jnp.pi / 3.0) / a**3
    ratio = cum_mass / jnp.maximum(r_h**3, 1e-30)
    log_r = jnp.log(r_h)
    log_ratio = jnp.log(jnp.maximum(ratio, 1e-30))
    log_target = jnp.log(jnp.maximum(target, 1e-30))
    return jnp.exp(jnp.interp(log_target, log_ratio[::-1], log_r[::-1]))


def dmb_halo_quantities(
    m_phys,
    z,
    c200c,
    omega_b,
    omega_m,
    h,
    params,
    r_work_over_r200=None,
):
    """
    DMB densities and electron pressure on a work radial grid for one halo.

    **Correctness.** Physics matches GODMAX ``BCM_18_wP`` (same β, θ_ej/θ_co,
    component fractions, adiabatic-contraction root, HSE → Pe). Enclosed mass
    and HSE use cumulative trapezoid sweeps on the work grid instead of
    GODMAX's per-radius re-integration; on the same nodes this is the same
    trapezoid rule. Residual vs GODMAX is dominated by quadrature-grid
    differences and is checked in ``tests/test_dmb_profiles.py``
    (median |Δ| / |ref| ≲ 2% on ``0.05–3 R200c``).

    **GPU path.** ζ is solved on a dense ``(N_ζ, N_r)`` grid with precomputed
    cumulative masses (no nested ∫ per trial ζ); HSE is one reverse sweep.
    Callers evaluate many halos with ``vmap`` (see ``_eval_field``).

    Parameters
    ----------
    m_phys : float
        ``M_200c`` in physical ``M_sun``.
    z, c200c, omega_b, omega_m, h : float
        Redshift, concentration, and cosmology.
    params : dict
        DMB parameter dictionary.
    r_work_over_r200 : array_like, optional
        Dimensionless comoving radii ``r / r_200c``.

    Returns
    -------
    dict
        ``r_comoving``, ``rho_gas``, ``rho_dmb``, ``Pe``, ``zeta``, …
    """
    n_int = int(params["num_points_trapz_int"])
    convention = params.get("convention", "godmax")
    if r_work_over_r200 is None:
        # Extend slightly past 6 R200c so the HSE outer boundary matches GODMAX.
        r_work_over_r200 = jnp.logspace(-3.0, jnp.log10(16.0), 96)

    m_h = m_phys * h
    ez2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
    rho_c_z_h = _RHO_CRIT_0_H * ez2
    r200c_h = (m_h * 3.0 / (4.0 * jnp.pi * 200.0 * rho_c_z_h)) ** (1.0 / 3.0)
    r200c_h = r200c_h * (1.0 + z)  # comoving Mpc/h
    r200c_comoving = r200c_h / h

    r_h = r_work_over_r200 * r200c_h
    rt = params["epsilon_rt"] * r200c_h
    log_r_h = jnp.log(r_h)

    Mc0 = 10.0 ** params["log10_Mc0"]
    Mstar0 = 10.0 ** params["log10_Mstar0"]
    Mc = Mc0 * (m_h / Mstar0) ** params["nu_M"] * (1.0 + z) ** params["nu_z"]
    beta = (
        3.0
        * (m_h / Mc) ** params["mu_beta"]
        / (1.0 + (m_h / Mc) ** params["mu_beta"])
    )

    theta_ej = (
        params["theta_ej_0"]
        * (m_h / 10.0 ** params["log10_Mstar0_theta_ej"]) ** params["nu_theta_ej_M"]
        * (1.0 + z) ** params["nu_theta_ej_z"]
        * (1.0 / c200c) ** params["nu_theta_ej_c"]
    )
    theta_co = (
        params["theta_co_0"]
        * (m_h / 10.0 ** params["log10_Mstar0_theta_co"]) ** params["nu_theta_co_M"]
        * (1.0 + z) ** params["nu_theta_co_z"]
        * (1.0 / c200c) ** params["nu_theta_co_c"]
    )
    r_co = theta_co * r200c_h
    r_ej = theta_ej * r200c_h

    M1 = 10.0 ** params["log10_M1_starcga"]
    fstar = params["A_starcga"] * (M1 / m_h) ** params["eta_star"]
    fcga = params["A_starcga"] * (M1 / m_h) ** params["eta_cga"]
    fgas = (omega_b / omega_m) - fstar
    fclm = (1.0 - omega_b / omega_m) + fstar - fcga
    Rh = 0.015 * r200c_h

    nfw_trunc = params["nfw_trunc"]
    gamma_g = params["gamma_rhogas"]
    delta_g = params["delta_rhogas"]

    def nfw_unnorm(r):
        rs = r200c_h / c200c
        x = r / rs
        y = r / rt
        rho = 1.0 / (x * (1.0 + x) ** 2)
        if nfw_trunc:
            rho = rho / (1.0 + y**2) ** 2
        return rho

    def gas_unnorm(r):
        u = r / r_co
        v = r / r_ej
        # To 3.4 / Dalal 2.4 print (1+v)^{(δ-β)/γ}; GODMAX uses (1+v^γ)^{(δ-β)/γ}.
        if convention == "literature":
            return 1.0 / (
                (1.0 + u) ** beta * (1.0 + v) ** ((delta_g - beta) / gamma_g)
            )
        return 1.0 / (
            (1.0 + u) ** beta
            * (1.0 + v**gamma_g) ** ((delta_g - beta) / gamma_g)
        )

    # Normalizations on a dedicated quadrature grid (matches GODMAX limits).
    log_norm = jnp.linspace(jnp.log(0.01 * r200c_h), jnp.log(r200c_h), n_int)
    rho_nfw_0 = m_h / _trapz_mass(nfw_unnorm(jnp.exp(log_norm)), log_norm)

    def rho_nfw(r):
        return rho_nfw_0 * nfw_unnorm(r)

    log_tot = jnp.linspace(jnp.log(0.01 * r200c_h), jnp.log(16.0 * r200c_h), n_int)
    Mtot = _trapz_mass(rho_nfw(jnp.exp(log_tot)), log_tot)
    rho_gas_0 = fgas * Mtot / _trapz_mass(gas_unnorm(jnp.exp(log_tot)), log_tot)

    def rho_gas(r):
        return rho_gas_0 * gas_unnorm(r)

    def rho_cga(r):
        return (
            (fcga * Mtot)
            / (4.0 * (jnp.pi**1.5) * Rh * r**2)
            * jnp.exp(-((0.5 * r / Rh) ** 2))
        )

    rho_nfw_arr = rho_nfw(r_h)
    rho_gas_arr = rho_gas(r_h)
    rho_cga_arr = rho_cga(r_h)

    Mnfw = _cum_mass(rho_nfw_arr, r_h)
    Mgas = _cum_mass(rho_gas_arr, r_h)
    Mcga = _cum_mass(rho_cga_arr, r_h)

    # Adiabatic contraction ζ(r): evaluate the root equation on a (Nζ, Nr)
    # grid in one shot using precomputed cumulative masses (no per-radius
    # re-integration). This is algebraically the same root as GODMAX's
    # 32-point interp, just vectorized for GPU.
    n_zeta_grid = 32
    zeta_grid = jnp.linspace(0.5, 1.5, n_zeta_grid)
    a_zeta = params["a_zeta"]
    n_zeta = params["n_zeta"]
    rf = zeta_grid[:, None] * r_h[None, :]  # (Nζ, Nr)
    log_rf = jnp.log(rf)

    def _interp_rows(log_q):
        return jnp.interp(log_q, log_r_h, Mcga), jnp.interp(
            log_q, log_r_h, Mgas
        )

    Mcga_rf, Mgas_rf = jax.vmap(_interp_rows)(log_rf)
    Mi = Mnfw[None, :]
    Mf = fclm * Mi + Mcga_rf + Mgas_rf
    Mf = jnp.maximum(Mf, 1e-30 * m_h)
    eq = (zeta_grid[:, None] - 1.0) - a_zeta * ((Mi / Mf) ** n_zeta - 1.0)
    zeta_arr = jax.vmap(
        lambda eq_col: jnp.interp(0.0, eq_col, zeta_grid), in_axes=1
    )(eq)

    rho_clm = (fclm / zeta_arr**3) * rho_nfw(r_h / zeta_arr)
    rho_dmb_arr = rho_gas_arr + rho_cga_arr + rho_clm
    mdmb_arr = _cum_mass(rho_dmb_arr, r_h)

    # HSE: GODMAX cuts at 6 R200c; papers integrate to ∞ → work-grid edge.
    r_out = r_h[-1] if convention == "literature" else (6.0 * r200c_h)
    w_out = jnp.where(r_h <= r_out, 1.0, 0.0)
    f_hse = w_out * (_G_KEV * rho_gas_arr * mdmb_arr / (r_h**2))
    ptot_comoving = jnp.clip(_reverse_cumtrapz(f_hse, r_h), 1e-30) * h**2

    a = 1.0 / (1.0 + z)
    ptot_phys = ptot_comoving / a**4
    n_nt_zcap = params["n_nt_zcap"]
    r500c_h = _find_r_delta_from_mass(r_h, Mnfw, 500.0, rho_c_z_h, z)
    r_over_r500c = r_h / r500c_h
    if n_nt_zcap is not None:
        # Dalal et al. 2026 eq. 2.12.
        alpha_nt = params["alpha_nt"]
        fz_act = jnp.minimum(
            (1.0 + z) ** 0.5,
            (4.0 ** (-n_nt_zcap / alpha_nt) - 1.0) * jnp.tanh(0.5 * z) + 1.0,
        )
        pnt_fac = alpha_nt * fz_act * (r_over_r500c**0.8)
    elif convention == "literature":
        # To et al. 2024 eq. 3.12.
        alpha_nt = params["alpha_nt"]
        fz_to = jnp.minimum(
            (1.0 + z) ** 0.5,
            (6.0 ** (-0.8) / alpha_nt - 1.0) * jnp.tanh(0.5 * z) + 1.0,
        )
        pnt_fac = alpha_nt * fz_to * (r_over_r500c**0.8)
    else:
        # GODMAX: n_nt is radial slope on R_200c, beta_nt is redshift power.
        fmax = 6.0 ** (-params["n_nt"]) / params["alpha_nt"]
        fz = jnp.minimum(
            (1.0 + z) ** params["beta_nt"],
            (fmax - 1.0) * jnp.tanh(params["beta_nt"] * z) + 1.0,
        )
        pnt_fac = params["alpha_nt"] * fz * (r_work_over_r200**params["n_nt"])
    pe_ev = (ptot_phys * jnp.maximum(0.0, 1.0 - pnt_fac) / _PTH_TO_PE) * _PE_KEV_TO_EV

    rho_to_phys = h**2
    return dict(
        r_comoving=r_h / h,
        r_over_r200=r_work_over_r200,
        r200c_comoving=r200c_comoving,
        rho_gas=rho_gas_arr * rho_to_phys,
        rho_cga=rho_cga_arr * rho_to_phys,
        rho_clm=rho_clm * rho_to_phys,
        rho_nfw=rho_nfw_arr * rho_to_phys,
        rho_dmb=rho_dmb_arr * rho_to_phys,
        Pe=pe_ev,
        zeta=zeta_arr,
        pnt_fac=pnt_fac,
        r500c_comoving=r500c_h / h,
    )


def _interp_log(r_query, r_grid, y_grid):
    r_query = jnp.atleast_1d(r_query)
    return jnp.exp(
        jnp.interp(
            jnp.log(r_query),
            jnp.log(r_grid),
            jnp.log(jnp.maximum(y_grid, 1e-30)),
        )
    )


def _m200c_c200c(halo_model, m, z, c200c_override):
    """Return ``m200c`` (Nm,) broadcastable and ``c200c`` with shape (Nm, Nz)."""
    m = jnp.atleast_1d(m)
    z = jnp.atleast_1d(z)
    mass_def_200c = MassDefinition(200, "critical")
    c_native = halo_model.concentration.c_delta(halo_model, m, z)

    same_def = (halo_model.mass_definition.delta == 200) & (
        halo_model.mass_definition.reference == "critical"
    )
    if same_def:
        m200c = m
        c200c = c_native
    else:
        m200c_mz = convert_m_delta(
            halo_model.cosmology,
            m,
            z,
            halo_model.mass_definition,
            mass_def_200c,
            c_native,
        )
        # Use z=0 column masses for the 1D m API; c scales with radius ratio.
        m200c = m200c_mz[:, 0]
        r_old = halo_model.mass_definition.r_delta(halo_model.cosmology, m, z)
        r_new = mass_def_200c.r_delta(halo_model.cosmology, m200c, z)
        c200c = c_native * (r_old / r_new)

    if c200c_override is not None:
        c_ov = jnp.asarray(c200c_override, dtype=c200c.dtype)
        c200c = jnp.ones_like(c200c) * c_ov if c_ov.ndim == 0 else c_ov
    return m200c, c200c


def _eval_field(halo_model, r, m, z, params, field, c200c_override=None):
    """
    Evaluate a DMB scalar field on shape ``(Nr, Nm, Nz)``.

    Halos are evaluated in parallel with ``vmap`` over the flattened ``(M, z)``
    grid (one JIT kernel; GPU-friendly).
    """
    r = jnp.atleast_1d(r)
    m = jnp.atleast_1d(m)
    z = jnp.atleast_1d(z)
    cparams = halo_model.cosmology._cosmo_params()
    h = cparams["h"]
    omega_b = cparams["Omega_b"]
    omega_m = cparams["Omega0_m"]
    m200c, c200c = _m200c_c200c(halo_model, m, z, c200c_override)

    nm, nz = m200c.shape[0], z.shape[0]
    m_b = jnp.broadcast_to(m200c[:, None], (nm, nz)).reshape(-1)
    z_b = jnp.broadcast_to(z[None, :], (nm, nz)).reshape(-1)
    c_b = c200c.reshape(-1)

    def one(m_i, z_i, c_i):
        q = dmb_halo_quantities(
            m_i, z_i, c_i, omega_b, omega_m, h, params
        )
        return _interp_log(r, q["r_comoving"], q[field])

    # (Nm*Nz, Nr_query)
    vals = jax.vmap(one)(m_b, z_b, c_b)
    return jnp.moveaxis(vals.reshape(nm, nz, -1), -1, 0)


def _r200c_grid(halo_model, m, z, c200c_override):
    m200c, _ = _m200c_c200c(halo_model, m, z, c200c_override)
    mass_def_200c = MassDefinition(200, "critical")
    # r_delta(m200c, z) with m200c 1D → (Nm, Nz)
    return mass_def_200c.r_delta(halo_model.cosmology, m200c, z)


def _hankel_interp_ell(profile, halo_model, k, m, z, r_delta, prefactor_fn):
    """Shared Limber/Hankel path used by pressure and gas density profiles."""
    k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
    d_A = jnp.atleast_1d(halo_model.cosmology.angular_diameter_distance(z))
    ell_delta = d_A[None, :] / r_delta
    chi = d_A * (1.0 + z)
    ell_target = k[:, None] * chi[None, :] - 0.5
    prefactor = prefactor_fn(r_delta, d_A, chi, z)
    r = profile.x[:, None, None] * r_delta[None, :, :] * (1.0 + z[None, None, :])
    k_native, u_k_native = profile._u_k_hankel(halo_model, profile.x, r, m, z)
    u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2.0 * k_native[:, None, None]))
    ell_native = k_native[:, None, None] * ell_delta[None, :, :]
    u_ell_val = prefactor[None, :, :] * u_ell_native

    def interp_col(ell_t, ell_n, u_n):
        return jnp.interp(ell_t, ell_n, u_n)

    return jax.vmap(
        jax.vmap(interp_col, in_axes=(None, 1, 1), out_axes=1),
        in_axes=(1, 2, 2),
        out_axes=2,
    )(ell_target, ell_native, u_ell_val)


class DMBPressureProfile(_DMBParamsMixin, PressureProfile):
    """Electron pressure from DMB HSE.

    ``convention="godmax"`` (default) or ``"literature"`` — see module docstring.
    """

    def __init__(self, x=None, **kwargs):
        self._init_dmb_params(**kwargs)
        self.x = x if x is not None else np.logspace(-4, 1.5, 256)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.asarray(value)
        self._hankel = HankelTransform(jnp.asarray(self._x), nu=0.5)

    @partial(jax.jit, static_argnums=(0,))
    def u_r(self, halo_model, r, m, z):
        return _eval_field(
            halo_model, r, m, z, _param_dict(self), "Pe", self.c200c
        )

    @partial(jax.jit, static_argnums=(0,))
    def u_k(self, halo_model, k, m, z):
        r_delta = _r200c_grid(halo_model, m, z, self.c200c)
        mpc_to_m = Const._Mpc_over_m_

        def prefactor_fn(r_delta, d_A, chi, z):
            ell_delta = d_A[None, :] / r_delta
            return (1.0 + z)[None, :] * 4.0 * jnp.pi * r_delta * mpc_to_m / (
                ell_delta**2
            )

        return _hankel_interp_ell(
            self, halo_model, k, m, z, r_delta, prefactor_fn
        )


class DMBMatterProfile(_DMBParamsMixin, MatterProfile):
    """Total DMB matter profile ``rho_dmb / rho_m0`` for lensing / ``P(k)``.

    GODMAX normalizes both DMB and the gravity-only NFW by the same
    ``Mtot = ∫_0^{16 R200} 4π r² ρ_nfw dr`` (not ``M200c``). Use
    :class:`DMBNFWMatterProfile` as the NFW counterpart so ``u(k→0)`` matches.
    """

    _density_field = "rho_dmb"

    def __init__(self, x=None, **kwargs):
        self._init_dmb_params(**kwargs)
        self.x = x if x is not None else np.logspace(-4, 1.5, 256)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.asarray(value)
        self._hankel = HankelTransform(jnp.asarray(self._x), nu=0.5)

    @partial(jax.jit, static_argnums=(0,))
    def u_r(self, halo_model, r, m, z):
        rho = _eval_field(
            halo_model, r, m, z, _param_dict(self), self._density_field, self.c200c
        )
        cparams = halo_model.cosmology._cosmo_params()
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"]
        return rho / rho_mean_0

    @partial(jax.jit, static_argnums=(0,))
    def u_k(self, halo_model, k, m, z):
        """
        3D Fourier transform of ``rho_dmb / rho_m0``.

        The Hankel grid is dimensionless ``x = r / R`` with
        ``R = r_200c (1+z)`` (comoving). Converting the ``x``-space transform
        to physical ``k`` therefore needs ``k_phys = k_x / R`` and amplitude
        ``4π R³`` (same spherical-Bessel convention as the analytic NFW ``u_k``).
        """
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        r_delta = _r200c_grid(halo_model, m, z, self.c200c)
        R = r_delta * (1.0 + z[None, :])
        r = self.x[:, None, None] * R[None, :, :]
        k_x, u_k_x = self._u_k_hankel(halo_model, self.x, r, m, z)
        u_x = u_k_x * jnp.sqrt(jnp.pi / (2.0 * k_x[:, None, None]))
        u_3d = (4.0 * jnp.pi * R[None, :, :] ** 3) * u_x
        k_phys = k_x[:, None, None] / R[None, :, :]

        def interp_col(k_n, u_n):
            return jnp.interp(k, k_n, u_n)

        return jax.vmap(
            jax.vmap(interp_col, in_axes=(1, 1), out_axes=1),
            in_axes=(2, 2),
            out_axes=2,
        )(k_phys, u_3d)


class DMBGasDensityProfile(_DMBParamsMixin, DensityProfile):
    """DMB gas density for kSZ tracers (shape relative to ``M / R^3``)."""

    def __init__(self, x=None, **kwargs):
        self._init_dmb_params(**kwargs)
        self.x = x if x is not None else np.logspace(-4, 1.5, 256)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.asarray(value)
        self._hankel = HankelTransform(jnp.asarray(self._x), nu=0.5)

    @partial(jax.jit, static_argnums=(0,))
    def u_r(self, halo_model, r, m, z):
        rho = _eval_field(
            halo_model, r, m, z, _param_dict(self), "rho_gas", self.c200c
        )
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        r_delta = _r200c_grid(halo_model, m, z, self.c200c)
        return rho / (m[None, :, None] / (r_delta[None, :, :] ** 3))

    @partial(jax.jit, static_argnums=(0,))
    def u_k(self, halo_model, k, m, z):
        r_delta = _r200c_grid(halo_model, m, z, self.c200c)

        def prefactor_fn(r_delta, d_A, chi, z):
            velocity_dispersion = jnp.sqrt(
                halo_model.cosmology.velocity_dispersion(z)
            )
            return (
                4.0
                * jnp.pi
                * r_delta**3
                / _MU_E
                * (1.0 + z)[None, :] ** 3
                / chi[None, :] ** 2
                * velocity_dispersion[None, :]
            )

        return _hankel_interp_ell(
            self, halo_model, k, m, z, r_delta, prefactor_fn
        )


class DMBNFWMatterProfile(DMBMatterProfile):
    """GODMAX truncated NFW counterpart to :class:`DMBMatterProfile`.

    Same ``Mtot`` and Hankel path as DMB (``setup_power_spectra_jit.uk_nfw``).
    Do not use analytic :class:`~hmfast.halos.profiles.matter.NFWMatterProfile`
    for ``P_DMB/P_NFW`` — that is ``M200c``, not GODMAX ``Mtot``.
    """

    _density_field = "rho_nfw"


register_pytree_node(
    DMBPressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux, children: DMBPressureProfile._tree_unflatten(aux, children),
)
register_pytree_node(
    DMBMatterProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux, children: DMBMatterProfile._tree_unflatten(aux, children),
)
register_pytree_node(
    DMBNFWMatterProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux, children: DMBNFWMatterProfile._tree_unflatten(aux, children),
)
register_pytree_node(
    DMBGasDensityProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux, children: DMBGasDensityProfile._tree_unflatten(aux, children),
)
