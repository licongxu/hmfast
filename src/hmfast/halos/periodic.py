"""Cubic reciprocal-lattice helpers for the periodic-universe 2-halo term.

The allowed wavevectors on a cubic 3-torus of side :math:`L` are
:math:`k_p = 2\\pi p / L` with :math:`p \\in \\mathbb{Z}^3 \\setminus \\{0\\}`.
Isotropic profiles and :math:`P_{\\mathrm{lin}}` depend only on
:math:`k_s = (2\\pi/L)\\sqrt{s}` where :math:`s = p_x^2+p_y^2+p_z^2`, so
modes can be grouped by shell multiplicity :math:`g_s`.

See ``ref_derivation/Periodic_Universe_Halo_Model_tSZ.pdf`` §§4.4, 6.1, 10.
"""

import numpy as np
from scipy.integrate import trapezoid


def _resolve_lattice_cut(s_max=None, n_max=None):
    """Return integer ``(s_max, n_max)`` from whichever cutoff was supplied."""
    if n_max is None and s_max is None:
        raise ValueError("Provide n_max or s_max (or both).")
    if n_max is None:
        s_max = int(s_max)
        if s_max < 1:
            raise ValueError("s_max must be >= 1 to include any nonzero mode.")
        n_max = int(np.floor(np.sqrt(s_max)))
    else:
        n_max = int(n_max)
        if n_max < 1:
            raise ValueError("n_max must be >= 1 to include any nonzero mode.")
        if s_max is None:
            s_max = 3 * n_max * n_max
        else:
            s_max = int(s_max)
            if s_max < 1:
                raise ValueError("s_max must be >= 1 to include any nonzero mode.")
    return s_max, n_max


def cubic_lattice_vectors(s_max=None, n_max=None):
    """Integer reciprocal-lattice vectors with :math:`p \\neq 0`.

    Parameters
    ----------
    s_max : int, optional
        Keep vectors with :math:`s = p_x^2+p_y^2+p_z^2 \\le s_{\\max}`.
    n_max : int, optional
        Restrict each component to :math:`|p_i| \\le n_{\\max}`.
        If only ``n_max`` is given, ``s_max`` defaults to :math:`3 n_{\\max}^2`.
        If only ``s_max`` is given, ``n_max`` defaults to
        :math:`\\lfloor\\sqrt{s_{\\max}}\\rfloor`.

    Returns
    -------
    p : ndarray, shape ``(N, 3)``
        Integer vectors, DC mode excluded.
    """
    s_max, n_max = _resolve_lattice_cut(s_max=s_max, n_max=n_max)
    rng = np.arange(-n_max, n_max + 1, dtype=np.int64)
    px, py, pz = np.meshgrid(rng, rng, rng, indexing="ij")
    p = np.stack((px.ravel(), py.ravel(), pz.ravel()), axis=1)
    s = np.sum(p * p, axis=1)
    return p[(s > 0) & (s <= s_max)]


def cubic_lattice_shells(s_max=None, n_max=None):
    """Unique shells :math:`s` and multiplicities :math:`g_s`, DC excluded.

    Parameters
    ----------
    s_max, n_max : int, optional
        Same cutoffs as :func:`cubic_lattice_vectors`.

    Returns
    -------
    s : ndarray of int, shape ``(N_s,)``
        Sorted unique :math:`s = p_x^2+p_y^2+p_z^2 \\ge 1`.
    g : ndarray of int, shape ``(N_s,)``
        Multiplicity :math:`g_s = |\\{p : p^2 = s\\}|`.
    """
    p = cubic_lattice_vectors(s_max=s_max, n_max=n_max)
    s_all = np.sum(p * p, axis=1)
    s, g = np.unique(s_all, return_counts=True)
    return s.astype(np.int64), g.astype(np.int64)


def lattice_wavenumbers(L, s):
    """Shell wavenumbers :math:`k_s = 2\\pi \\sqrt{s} / L` in :math:`\\mathrm{Mpc}^{-1}`.

    Parameters
    ----------
    L : float
        Periodic box side length in physical :math:`\\mathrm{Mpc}`.
    s : array-like
        Integer shell indices :math:`s = p_x^2+p_y^2+p_z^2`.

    Returns
    -------
    k_s : ndarray
        Wavenumbers in physical :math:`\\mathrm{Mpc}^{-1}`.
    """
    L = float(L)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("L must be a positive finite box side in Mpc.")
    s = np.asarray(s, dtype=float)
    return (2.0 * np.pi) * np.sqrt(s) / L


def n_max_for_kmax(L, k_max):
    """Largest ``n_max`` whose fundamental cube still lies at :math:`k\\le k_{\\max}`."""
    L = float(L)
    k_max = float(k_max)
    if L <= 0.0 or k_max <= 0.0:
        raise ValueError("L and k_max must be positive.")
    return max(int(np.floor(k_max * L / (2.0 * np.pi))), 1)


def interpolate_kz(k_src, arr, k_tgt):
    """Linear interpolation of an ``(N_k, N_z)`` array onto a new k-grid."""
    k_src = np.asarray(k_src, dtype=float)
    k_tgt = np.asarray(k_tgt, dtype=float)
    arr = np.asarray(arr, dtype=float)
    out = np.empty((k_tgt.size, arr.shape[1]), dtype=float)
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(k_tgt, k_src, arr[:, j])
    return out


def integrate_radial_bessel(ell, k, z, chi, weight, n_chi=None):
    """Non-Limber line-of-sight integral :math:`\\int dz\\, w(k,z)\\, j_\\ell(k\\chi)`.

    ``weight`` is the slow kernel (already includes :math:`dV/dz\\,d\\Omega`,
    :math:`W`, :math:`I_1`, and :math:`\\sqrt{P}`). It is interpolated from the
    native :math:`z` grid onto a finer radial grid so that :math:`j_\\ell(k\\chi)`
    is resolved. Returns shape ``(N_\\ell, N_k)``.
    """
    from scipy.special import spherical_jn

    ell = np.atleast_1d(np.asarray(ell, dtype=float))
    k = np.atleast_1d(np.asarray(k, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    chi = np.atleast_1d(np.asarray(chi, dtype=float))
    weight = np.atleast_2d(np.asarray(weight, dtype=float))

    if n_chi is None:
        span = float(chi[-1] - chi[0])
        k_max = float(np.max(k)) if k.size else 0.0
        need = int(np.ceil(span * max(k_max, 1.0e-6) / 0.5)) + 1
        n_chi = int(np.clip(need, max(int(z.size), 32), 4096))
    n_chi = int(n_chi)
    if n_chi < 2:
        raise ValueError("n_chi must be >= 2")

    if n_chi == z.size and np.allclose(np.linspace(z[0], z[-1], n_chi), z):
        z_f = z
        chi_f = chi
        w_f = weight
    else:
        z_f = np.linspace(z[0], z[-1], n_chi)
        chi_f = np.interp(z_f, z, chi)
        w_f = np.empty((k.size, n_chi), dtype=float)
        for i in range(k.size):
            w_f[i] = np.interp(z_f, z, weight[i])

    x = k[:, None] * chi_f[None, :]
    R = np.empty((ell.size, k.size), dtype=float)
    for i, ell_i in enumerate(ell):
        jl = spherical_jn(ell_i, x)
        R[i] = trapezoid(w_f * jl, x=z_f, axis=-1)
    return R


def gaussian_cl_variance(cl, l):
    """Isotropic full-sky Gaussian variance :math:`2 C_\\ell^2 / (2\\ell+1)`."""
    cl = np.atleast_1d(np.asarray(cl, dtype=float))
    l = np.atleast_1d(np.asarray(l, dtype=float))
    return 2.0 * cl * cl / (2.0 * l + 1.0)


def lattice_Q_ell(ell, p, A, A_iso=0.0, chunk=256):
    """Periodic anisotropy factor :math:`Q_\\ell=\\sum_{p,q} w_{\\ell p}w_{\\ell q}P_\\ell^2(\\mu_{pq})`.

    Weights are :math:`w_{\\ell p}=A_{\\ell p}/\\sum_q A_{\\ell q}` with
    :math:`A_{\\ell p}=|R_\\ell(k_p)|^2` (PDF eqs. 61–64). In the isotropic
    continuum :math:`Q_\\ell\\to 1/(2\\ell+1)`.

    ``A_iso`` is the total amplitude of additional high-:math:`k` modes
    treated as isotropic (many directions). Then
    :math:`Q=W_{\\mathrm{ex}}^2 Q_{\\mathrm{ex}}+(1-W_{\\mathrm{ex}}^2)/(2\\ell+1)`
    with :math:`W_{\\mathrm{ex}}=\\sum A/(\\sum A+A_{\\mathrm{iso}})`.

    Parameters
    ----------
    ell : array-like, shape ``(N_\\ell,)``
        Multipoles (integers expected).
    p : array-like, shape ``(N, 3)``
        Reciprocal-lattice vectors, DC excluded.
    A : array-like, shape ``(N_\\ell, N)``
        Per-mode amplitudes :math:`|R_\\ell(k_p)|^2`.
    A_iso : float or array, shape ``(N_\\ell,)``
        Isotropic remainder of :math:`\\sum_p |R|^2`.
    chunk : int
        Rows of the :math:`p,q` pair sum computed at once.

    Returns
    -------
    Q : ndarray, shape ``(N_\\ell,)``
    """
    from scipy.special import eval_legendre

    ell = np.atleast_1d(np.asarray(ell, dtype=float))
    p = np.asarray(p, dtype=float)
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("p must have shape (N, 3)")
    if A.shape[1] != p.shape[0]:
        raise ValueError("A must have shape (N_ell, N_modes)")

    nrm = np.linalg.norm(p, axis=1)
    uhat = p / nrm[:, None]
    tot = np.sum(A, axis=1)
    w = np.divide(A, tot[:, None], out=np.zeros_like(A), where=tot[:, None] > 0.0)
    A_iso = np.broadcast_to(np.asarray(A_iso, dtype=float), tot.shape)
    w_ex = np.divide(
        tot, tot + A_iso, out=np.zeros_like(tot), where=(tot + A_iso) > 0.0
    )

    n = p.shape[0]
    Q = np.empty(ell.size, dtype=float)
    for i, li in enumerate(ell):
        floor = 1.0 / (2.0 * li + 1.0)
        if tot[i] <= 0.0:
            Q[i] = floor
            continue
        acc = 0.0
        wi = w[i]
        for i0 in range(0, n, int(chunk)):
            i1 = min(i0 + int(chunk), n)
            mu = np.clip(uhat[i0:i1] @ uhat.T, -1.0, 1.0)
            Pl = eval_legendre(li, mu)
            acc += float(np.sum(wi[i0:i1, None] * wi[None, :] * (Pl * Pl)))
        Q[i] = (w_ex[i] ** 2) * acc + (1.0 - w_ex[i] ** 2) * floor
    return Q


def multipole_bin_weights(ell, ell_edges):
    """Uniform bandpower weights :math:`W_{b\\ell}=1/N_b` (PDF §7).

    Bins are :math:`[\\ell_{\\mathrm{lo}},\\ell_{\\mathrm{hi}})` except the
    last, which is closed on the right. Empty bins raise.

    Returns
    -------
    W : ndarray, shape ``(N_b, N_\\ell)``
    ell_eff : ndarray, shape ``(N_b,)``
        Mean :math:`\\ell` in each bin.
    n_ell : ndarray of int, shape ``(N_b,)``
    """
    ell = np.atleast_1d(np.asarray(ell, dtype=float))
    edges = np.atleast_1d(np.asarray(ell_edges, dtype=float))
    if edges.size < 2:
        raise ValueError("ell_edges must contain at least two values.")
    n_bin = edges.size - 1
    W = np.zeros((n_bin, ell.size), dtype=float)
    ell_eff = np.empty(n_bin, dtype=float)
    n_ell = np.empty(n_bin, dtype=np.int64)
    for b in range(n_bin):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bin - 1:
            mask = (ell >= lo) & (ell <= hi)
        else:
            mask = (ell >= lo) & (ell < hi)
        n = int(np.count_nonzero(mask))
        if n == 0:
            raise ValueError(f"multipole bin [{lo}, {hi}] contains no ell samples.")
        W[b, mask] = 1.0 / n
        ell_eff[b] = float(np.mean(ell[mask]))
        n_ell[b] = n
    return W, ell_eff, n_ell


def lattice_gaussian_cl_cov(ell, p, R, L, R2=None):
    """Periodic 2-halo Gaussian covariance (PDF eq. 107).

    Both Wick pairings are summed with no extra factor of two:

    .. math::

        \\mathrm{Cov}^G_L(\\hat C_\\ell^{12},\\hat C_{\\ell'}^{12})
            = \\frac{(4\\pi)^2}{L^6}
              \\sum_{p,q\\neq0}
              \\Big[
              R^1_\\ell(k_p)R^1_{\\ell'}(k_p)
              R^2_\\ell(k_q)R^2_{\\ell'}(k_q)
              +
              R^1_\\ell(k_p)R^2_{\\ell'}(k_p)
              R^2_\\ell(k_q)R^1_{\\ell'}(k_q)
              \\Big]
              P_\\ell(\\mu_{pq})P_{\\ell'}(\\mu_{pq})

    ``R`` and optional ``R2`` have shape ``(N_\\ell, N_{\\mathrm{modes}})``.
    Omitting ``R2`` computes an auto-spectrum covariance, for which the two
    pairings coincide and the diagonal is :math:`2(C_\\ell^{2h})^2 Q_\\ell`.
    """
    from scipy.special import eval_legendre

    ell = np.atleast_1d(np.asarray(ell, dtype=float))
    p = np.asarray(p, dtype=float)
    R = np.atleast_2d(np.asarray(R, dtype=float))
    R2 = R if R2 is None else np.atleast_2d(np.asarray(R2, dtype=float))
    L = float(L)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("p must have shape (N, 3)")
    if R.shape[1] != p.shape[0]:
        raise ValueError("R must have shape (N_ell, N_modes)")
    if R2.shape != R.shape:
        raise ValueError("R2 must have the same shape as R")

    pref = (4.0 * np.pi / L**3) ** 2
    nrm = np.linalg.norm(p, axis=1)
    mu = (p @ p.T) / (nrm[:, None] * nrm[None, :])
    mu = np.clip(mu, -1.0, 1.0)
    Pl = [eval_legendre(li, mu) for li in ell]

    n_ell = ell.size
    cov = np.empty((n_ell, n_ell), dtype=float)
    for i in range(n_ell):
        for j in range(i, n_ell):
            Pl_ij = Pl[i] * Pl[j]
            v11 = R[i] * R[j]
            v22 = R2[i] * R2[j]
            v12 = R[i] * R2[j]
            v21 = R2[i] * R[j]
            val = pref * float(
                np.einsum("p,q,pq->", v11, v22, Pl_ij)
                + np.einsum("p,q,pq->", v12, v21, Pl_ij)
            )
            cov[i, j] = val
            cov[j, i] = val
    return cov


def project_bandpowers(W, vector=None, cov=None):
    """Project a vector and/or covariance with bandpower weights ``W``."""
    W = np.atleast_2d(np.asarray(W, dtype=float))
    out = {}
    if vector is not None:
        out["vector"] = W @ np.atleast_1d(np.asarray(vector, dtype=float))
    if cov is not None:
        C = np.atleast_2d(np.asarray(cov, dtype=float))
        out["cov"] = W @ C @ W.T
    return out
