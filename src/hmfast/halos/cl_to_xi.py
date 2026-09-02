"""
Angular power spectrum → correlation function via Hankel transforms.

Implements the flat-sky / Limber Bessel integrals used by GODMAX
(``get_corrfunc_BCMP``) and equivalent to CCL's FFTLog / Bessel correlators:

.. math::

    \\xi_+(\\theta) = \\int_0^\\infty \\frac{\\ell\\,d\\ell}{2\\pi}\\,
    C_\\ell^{\\kappa\\kappa}\\, J_0(\\ell\\theta)

    \\xi_-(\\theta) = \\int_0^\\infty \\frac{\\ell\\,d\\ell}{2\\pi}\\,
    C_\\ell^{\\kappa\\kappa}\\, J_4(\\ell\\theta)

    \\mathrm{gty}(\\theta) = \\int_0^\\infty \\frac{\\ell\\,d\\ell}{2\\pi}\\,
    C_\\ell^{\\kappa y}\\, J_2(\\ell\\theta)

``mcfit.Hankel`` evaluates ``∫ f(ℓ) J_ν(ℓθ) ℓ dℓ``; dividing by ``2π`` recovers
the expressions above (same convention as GODMAX: ``nu∈{0,2,4}``, ``q=1``,
``/ (2π)``). CCL ``correlation(..., type='GG+'/'GG-'/'NG', method='fftlog')``
targets the same transforms (CCL's public API also offers a full-sky Wigner-d
sum; the Bessel/FFTLog path is the small-angle limit used here).

Uses ``mcfit`` (already a hmfast dependency) with the JAX backend so the
transform runs on GPU when JAX does.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import mcfit
import numpy as np


def _hankel_cl(ell, cl, nu, q=1.0):
    """
    Hankel-transform ``C_ℓ`` to a correlation function.

    Parameters
    ----------
    ell : array_like
        Multipoles (strictly increasing, typically log-spaced).
    cl : array_like
        Angular power spectrum sampled on ``ell``. Extra leading axes are
        allowed; the transform runs along the last axis.
    nu : float
        Bessel order (``0`` → ξ₊, ``2`` → gty / γ_t, ``4`` → ξ₋).
    q : float
        mcfit bias parameter (GODMAX uses ``q=1``).

    Returns
    -------
    theta_rad : jnp.ndarray
        Native angular grid in radians.
    xi : jnp.ndarray
        Correlation function ``Hankel(C_ℓ) / (2π)`` with the same leading
        shape as ``cl``.
    """
    ell = jnp.asarray(ell)
    cl = jnp.asarray(cl)
    hankel = mcfit.Hankel(
        np.asarray(ell), nu=nu, q=q, lowring=True, backend="jax"
    )
    hankel_jit = jax.jit(functools.partial(hankel, extrap=False))

    if cl.ndim == 1:
        theta, xi = hankel_jit(cl)
        return jnp.asarray(theta), jnp.asarray(xi) / (2.0 * jnp.pi)

    leading = cl.shape[:-1]
    cl_flat = cl.reshape((-1, cl.shape[-1]))

    def one(row):
        theta, xi = hankel_jit(row)
        return theta, xi / (2.0 * jnp.pi)

    theta0, _ = one(cl_flat[0])
    xi_rows = jax.vmap(lambda row: one(row)[1])(cl_flat)
    return theta0, xi_rows.reshape(leading + (xi_rows.shape[-1],))


def cl_ky_to_gty(ell, cl_ky):
    """Convert ``C_ℓ^{κy}`` to the shear–y correlation ``γ_t y (θ)`` (``J_2``)."""
    return _hankel_cl(ell, cl_ky, nu=2.0)


def cl_kk_to_xip(ell, cl_kk):
    """Convert ``C_ℓ^{κκ}`` to ``ξ₊(θ)`` (``J_0``; CCL ``GG+``)."""
    return _hankel_cl(ell, cl_kk, nu=0.0)


def cl_kk_to_xim(ell, cl_kk):
    """Convert ``C_ℓ^{κκ}`` to ``ξ₋(θ)`` (``J_4``; CCL ``GG-``)."""
    return _hankel_cl(ell, cl_kk, nu=4.0)


def theta_to_arcmin(theta_rad):
    """Convert radians to arcminutes."""
    return jnp.asarray(theta_rad) * (180.0 * 60.0 / jnp.pi)


def interp_xi_theta(theta_native_rad, xi_native, theta_out_arcmin):
    """
    Log-log interpolate a correlation function onto output angles in arcmin.
    """
    theta_native_arcmin = theta_to_arcmin(theta_native_rad)
    theta_out = jnp.atleast_1d(jnp.asarray(theta_out_arcmin))
    xi_native = jnp.asarray(xi_native)

    def interp_one(xi):
        log_xi = jnp.interp(
            jnp.log(theta_out),
            jnp.log(theta_native_arcmin),
            jnp.log(jnp.maximum(jnp.abs(xi), 1e-30)),
        )
        sgn = jnp.sign(jnp.interp(theta_out, theta_native_arcmin, xi))
        return jnp.exp(log_xi) * sgn

    if xi_native.ndim == 1:
        return interp_one(xi_native)
    flat = xi_native.reshape((-1, xi_native.shape[-1]))
    out = jax.vmap(interp_one)(flat)
    return out.reshape(xi_native.shape[:-1] + (theta_out.shape[0],))
