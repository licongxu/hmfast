import warnings
from functools import partial

import jax
import jax.numpy as jnp
import mcfit
import numpy as np
from jax import lax


# This file is primarily to implement functions/units that are needed for this package but are not implemented in JAX

# Create a central location for all constants
class Const:
    # SI 2019 defining constants (exact) and constants derived from them
    c = 2.99792458e8            # speed of light in m/s
    h_P = 6.62607015e-34        # Planck constant in J s
    k_B = 1.380649e-23          # Boltzmann constant in J/K
    eV = 1.602176634e-19        # 1 eV expressed in J
    sigma_B = 5.670374419e-08   # Stefan-Boltzmann constant in W/m^2/K^4

    # CODATA 2018 measured constants
    G = 6.67430e-11             # Newton constant in m^3/kg/s^2
    sigma_T = 6.6524587321e-29  # Thomson cross-section in m^2
    m_e = 9.1093837015e-31      # electron mass in kg
    m_p = 1.67262192369e-27     # proton mass in kg

    # IAU 2015 astronomical constants
    Mpc_over_m = 3.085677581491367e22  # megaparsec in m, exact from the au
    GM_sun = 1.3271244e20       # nominal solar mass parameter in m^3/s^2
    L_sun = 3.828e26            # nominal solar luminosity in W
    M_sun = GM_sun / G      # solar mass in kg

    # Derived units and model conventions
    Gyr_over_Mpc = 3.06601394e2  # conversion factor from megaparsecs to gigayears
    C1_IA = 5e-14                # h^-2 Msun^-1 Mpc^3 (Hirata & Seljak 2004 / Joachimi et al. 2011 Eq. 6 normalization)




# Newton's method root finder
def _newton_solver(f, x0, tol=1e-8, max_iter=25):
    df = jax.grad(f)
    def cond_fn(state):
        x, i = state
        return (jnp.abs(f(x)) > tol) & (i < max_iter)
    def body_fn(state):
        x, i = state
        fx = f(x)
        dfx = df(x)
        x_new = x - fx / dfx
        return (x_new, i + 1)
    x_final, _ = lax.while_loop(cond_fn, body_fn, (x0, 0))
    return x_final

def newton_root(f, x0, tol=1e-8, max_iter=25):
    def solve(f, x0):
        return _newton_solver(f, x0, tol=tol, max_iter=max_iter)
    # For scalar roots, use the recommended tangent_solve
    tangent_solve = lambda g, y: y / g(1.0)
    return lax.custom_root(f, x0, solve, tangent_solve=tangent_solve)


# Dormand-Prince (DOPRI5) Butcher tableau -- order 5, embedded order-4 error estimate.
_DOPRI5_C = (1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0)
_DOPRI5_A = (
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
)
_DOPRI5_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)


def dopri5_integrate(rhs, coeffs, y0, x_start, x_end, n_steps):
    """
    Fixed-step Dormand-Prince (order 5) integration of dy/dx = rhs(coeffs(x), y), y(x_start) = y0,
    from x_start to x_end on ``n_steps`` uniform steps.

    The x-dependence enters only through ``coeffs``, evaluated once, vectorised, at every stage point
    of every step; the sequential ``lax.scan`` then only does arithmetic on the state, so it is cheap
    on GPU and in reverse mode.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``rhs(c, y) -> dy/dx``, with ``c`` the coefficients at one x. ``y`` and its
        return value may be arbitrary JAX pytrees of matching structure.
    coeffs : callable
        ``coeffs(x) -> c`` for a 1D array of x, a pytree whose leaves have a leading axis of ``len(x)``.
    y0 : pytree
        Initial state at ``x_start``.
    x_start, x_end : float
        Integration bounds (``x_end`` may be less than ``x_start`` to integrate backward).
    n_steps : int
        Number of uniform steps.

    Returns
    -------
    x_traj : jnp.ndarray
        The ``n_steps + 1`` uniform nodes from ``x_start`` to ``x_end``.
    y_traj : pytree
        State trajectory at each node in ``x_traj``, same pytree structure
        as ``y0`` with a leading axis of length ``n_steps + 1`` on every leaf.
    """
    h = (x_end - x_start) / n_steps
    x_traj = x_start + h * jnp.arange(n_steps + 1)
    x_stage = x_traj[:-1, None] + h * jnp.asarray((0.0,) + _DOPRI5_C)[None, :]  # (n_steps, 6)
    c_all = jax.tree_util.tree_map(lambda a: a.reshape(x_stage.shape), coeffs(x_stage.ravel()))

    def combine(y, weights, ks):
        return jax.tree_util.tree_map(
            lambda y_leaf, *k_leaves: y_leaf + h * sum(w * k for w, k in zip(weights, k_leaves)), y, *ks
        )

    def step(y, c_step):
        stage = lambda j: jax.tree_util.tree_map(lambda a: a[j], c_step)
        ks = [rhs(stage(0), y)]
        for j, a_row in enumerate(_DOPRI5_A):
            ks.append(rhs(stage(j + 1), combine(y, a_row, ks)))
        y_new = combine(y, _DOPRI5_B5, ks)
        return y_new, y_new

    _, y_traj = lax.scan(step, y0, c_all)
    y_traj = jax.tree_util.tree_map(
        lambda leaf0, traj: jnp.concatenate([jnp.asarray(leaf0)[None], traj]), y0, y_traj
    )
    return x_traj, y_traj



# Lambert W function. As of the writing of this comment, it is not yet implemented in JAX
def _real_lambertw_recursion(w: jax.Array, x: jax.Array) -> jax.Array:
    return w / (1+w) * (1+jnp.log(x / w))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _lambertwk0(x, max_steps=5):
    # See https://en.wikipedia.org/wiki/Lambert_W_function#Numerical_evaluation
    w_0 = jax.lax.select(
        x > jnp.e,
        jnp.log(x) - jnp.log(jnp.log(x)),
        x / jnp.e
    )
    w_0 = jax.lax.select(
        x > 0,
        w_0,
        jnp.e * x / (1 + jnp.e * x + jnp.sqrt(1 + jnp.e * x)) * jnp.log(
            1 + jnp.sqrt(1 + jnp.e * x))
    )
    
    w, _ = jax.lax.scan(
        lambda carry, _: (_real_lambertw_recursion(carry, x),)*2,
        w_0,
        xs=None, length=max_steps
    )
    
    w = jax.lax.select(
        jnp.isclose(x, 0.0),
        0.0,
        w
    )
        
    return w


@_lambertwk0.defjvp
def _lambertw_jvp(max_steps, primals, tangents):
    # Note: All branches for lambert W satisfy this JVP.
    x, = primals
    t, = tangents

    y = _lambertwk0(x, max_steps)
    dydx = 1 / (x + jnp.exp(y))

    jvp = jax.lax.select(
        jnp.isclose(x, -1/jnp.e),
        jnp.nan,
        dydx * t
    )

    return y, jvp


@jnp.vectorize
def lambertw(x, k=0, max_steps=5):
    if k != 0:
        raise NotImplementedError()

    return _lambertwk0(x, max_steps=max_steps)


@jax.jit
def log_interp1d_extrap(x, xp, fp):
    """
    Interpolate positive 1D data in log-log space and extrapolate using the
    endpoint log-slopes.

    Parameters
    ----------
    x : float or jnp.ndarray
        Evaluation points. Must be strictly positive.
    xp : jnp.ndarray
        Monotonic sample points. Must be strictly positive.
    fp : jnp.ndarray
        Positive function values sampled on ``xp``.

    Returns
    -------
    jnp.ndarray
        Interpolated or extrapolated values with the same shape as ``x``.
    """
    x = jnp.asarray(x)
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)

    log_x = jnp.log(x)
    log_xp = jnp.log(xp)
    log_fp = jnp.log(fp)

    left_slope = (log_fp[1] - log_fp[0]) / (log_xp[1] - log_xp[0])
    right_slope = (log_fp[-1] - log_fp[-2]) / (log_xp[-1] - log_xp[-2])

    log_f_interp = jnp.interp(log_x, log_xp, log_fp)
    log_f_left = log_fp[0] + left_slope * (log_x - log_xp[0])
    log_f_right = log_fp[-1] + right_slope * (log_x - log_xp[-1])

    log_f = jnp.where(
        x < xp[0],
        log_f_left,
        jnp.where(x > xp[-1], log_f_right, log_f_interp),
    )
    return jnp.exp(log_f)


def _gather_along0(values, index):
    """Gather ``values`` along axis 0.

    ``take_along_axis`` for a short table and a large index is about 10 ms on
    this TPU. Indexing the table directly is one gather and stays under 0.2 ms.
    A trailing shape of all ones is a single shared column.
    """
    trail = values.shape[1:]
    if not trail:
        return values[index]
    if all(size == 1 for size in trail):
        return values.reshape((values.shape[0],))[index]
    if index.shape[-len(trail):] != trail:
        index = jnp.broadcast_to(index, index.shape[:-len(trail)] + trail)
    n_columns = int(np.prod(trail, dtype=int))
    table = values.reshape((values.shape[0], n_columns))
    gathered = table[index.reshape((-1, n_columns)), jnp.arange(n_columns)]
    return gathered.reshape(index.shape)


def _cubic_interp_uniform(x, x0, dx, fp, left, right):
    """
    Cubic convolution interpolation (Keys 1981, a = -1/2) of data on a uniform grid, along its first axis.

    The grid is ``x0 + i * dx`` for ``i = 0, ..., n - 1``. The two end intervals are interpolated linearly, and
    points outside the grid take ``left`` or ``right``. Every trailing axis of ``fp`` is an independent column
    sharing the grid, so many columns are interpolated in one call.

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points, with as many axes as ``fp``; trailing axes broadcast against those of ``fp``.
    x0 : float
        First grid node.
    dx : float
        Grid spacing.
    fp : jnp.ndarray
        Values on the grid, shape ``(n, ...)`` with ``n >= 4``.
    left, right : float or jnp.ndarray
        Values below the first and above the last grid node, broadcastable to the output.

    Returns
    -------
    jnp.ndarray
        Interpolated values, with the broadcast shape of ``x`` and the trailing axes of ``fp``.
    """
    n = fp.shape[0]
    t = (x - x0) / dx
    at = lambda idx: _gather_along0(fp, idx)
    i = jnp.clip(jnp.floor(t).astype(int), 1, n - 3)
    d = t - i
    p0, p1, p2, p3 = at(i - 1), at(i), at(i + 1), at(i + 2)
    f = p1 + 0.5 * d * (p2 - p0 + d * (2 * p0 - 5 * p1 + 4 * p2 - p3 + d * (3 * (p1 - p2) + p3 - p0)))
    j = jnp.clip(jnp.floor(t).astype(int), 0, n - 2)
    linear = at(j) + (t - j) * (at(j + 1) - at(j))
    # lax.select, not jnp.where: where is an outlined call and breaks fusion on TPU.
    f = jax.lax.select((t < 1) | (t > n - 2), linear, f)
    f = jax.lax.select(t > n - 1, jnp.broadcast_to(right, f.shape), f)
    return jax.lax.select(t < 0, jnp.broadcast_to(left, f.shape), f)


def gauss_legendre_nodes_weights(a, b, n):
    """
    Gauss-Legendre quadrature nodes and weights on ``[a, b]``.

    The reference nodes/weights on ``[-1, 1]`` are a fixed, ``n``-dependent
    table (as with the Dormand-Prince tableau above), computed once via
    NumPy and then affinely mapped onto ``[a, b]``; this keeps the mapping
    differentiable with respect to ``a`` and ``b`` even though the reference
    table itself is not JAX-native.

    Parameters
    ----------
    a, b : float
        Integration bounds.
    n : int
        Number of quadrature nodes.

    Returns
    -------
    nodes : jnp.ndarray
        Quadrature nodes on ``[a, b]``, shape ``(n,)``.
    weights : jnp.ndarray
        Quadrature weights corresponding to ``nodes``, shape ``(n,)``.
    """
    x_ref, w_ref = np.polynomial.legendre.leggauss(n)
    x_ref, w_ref = jnp.asarray(x_ref), jnp.asarray(w_ref)
    scale = 0.5 * (b - a)
    return scale * x_ref + 0.5 * (a + b), scale * w_ref

def inside_truncation(x, x_out):
    """Whether a radius is inside ``x_out``, including the boundary.

    ``x = r / r_delta`` computed in float32 can land a few ulps above the
    truncation value when the sample was meant to sit on it. A relative pad of
    ``1e-5`` keeps that boundary sample. The pad is far smaller than the
    radial grid spacing, and a NaN radius stays NaN.
    """
    return jnp.isnan(x) | (x <= x_out * (1.0 + 1e-5))


def pairwise_sum(values, axis=0):
    """Sum ``values`` along ``axis`` by a balanced tree.

    A sequential sum in float32 drops digits when the terms cancel or span many
    decades. Pairing them keeps those digits. The length of ``axis`` is static.
    """
    values = jnp.moveaxis(values, axis, 0)

    def add(block):
        n = block.shape[0]
        if n == 1:
            return block[0]
        mid = n // 2
        return add(block[:mid]) + add(block[mid:])

    return add(values)


def pairwise_contract(matrix, samples):
    """Contract ``matrix`` (m, n) with ``samples`` (n, ...) by a pairwise sum on ``n``."""
    matrix = jnp.asarray(matrix, dtype=samples.dtype)

    def add(mat, samp):
        n = samp.shape[0]
        if n == 1:
            # Keep the matrix axis in front. A bare (m,) times a trailing
            # vector would broadcast from the right and move that axis.
            lead = mat[:, 0].reshape((mat.shape[0],) + (1,) * samp[0].ndim)
            return lead * samp[0]
        mid = n // 2
        return add(mat[:, :mid], samp[:mid]) + add(mat[:, mid:], samp[mid:])

    return add(matrix, samples)


_HANKEL_MATRICES = {}  # (x_min, x_max, n) -> (q nodes, transform matrix), built once per grid per process


def _truncation_steps(x64, h, x_out):
    """Number of log-steps from ``x64[0]`` to the truncation, snapped to an integer.

    A truncation that lands on a grid node is an integer step count. Float32
    ``log`` misses that integer by several ulps, which drops the boundary
    sample, so the count is formed in float64 whenever ``x_out`` is a concrete
    number. A traced ``x_out`` keeps the JAX path with a snap wide enough for
    float32.
    """
    limit = _concrete_float(x_out)
    if limit is None:
        s = jnp.log(jnp.minimum(x_out, x64[-1]) / x64[0]) / h
        return jnp.where(jnp.abs(s - jnp.round(s)) < 1e-4, jnp.round(s), s)
    if not np.isfinite(limit):
        s = np.log(x64[-1] / x64[0]) / h
    else:
        s = np.log(min(limit, x64[-1]) / x64[0]) / h
    if abs(s - round(s)) < 1e-4:
        s = round(s)
    return s


def _concrete_float(value):
    """Python float for a concrete number, or None when ``value`` is traced."""
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.shape != () or not np.issubdtype(array.dtype, np.number):
        return None
    return float(array)


def _cubic_basis(fraction):
    """Weights on the samples at ``i-1, i, i+1, i+2`` for fraction ``t - i``."""
    d2 = fraction * fraction
    d3 = d2 * fraction
    return (
        -0.5 * fraction + d2 - 0.5 * d3,
        1.0 - 2.5 * d2 + 1.5 * d3,
        0.5 * fraction + 2.0 * d2 - 1.5 * d3,
        -0.5 * d2 + 0.5 * d3,
    )


def _cubic_weight_matrix(index, fraction, n_source):
    """Rows are cubic stencils. Applying this matrix is the interpolation.

    A direct ``table[index]`` compiles to a multi-millisecond gather once it is
    inlined into the profile. The same stencil as a matrix product stays on the
    matrix unit.
    """
    offset = jnp.arange(n_source)[None, :] - index[:, None]
    below, at, above, high = _cubic_basis(fraction)
    return (
        (offset == -1) * below[:, None]
        + (offset == 0) * at[:, None]
        + (offset == 1) * above[:, None]
        + (offset == 2) * high[:, None]
    )


def _exact_cubic(table, log_q, lq0, dlq, left, right):
    """Cubic at each entry of ``log_q``.

    The stencil is a matrix product. Points off the left or right of the table
    take ``left`` or ``right``; those are the ``q -> 0`` integral and the last
    sample, and a cubic extrapolation there is wrong.
    """
    n_source = table.shape[0]
    leading = log_q.shape[0] if log_q.ndim else 1
    query = jnp.reshape(log_q, (leading, -1))
    left = jnp.reshape(left, (-1,))
    right = jnp.reshape(right, (-1,))
    rows = []
    for j in range(leading):
        coordinate = (query[j] - lq0) / dlq
        bin_index = jnp.clip(jnp.floor(coordinate).astype(jnp.int32), 1, n_source - 3)
        fraction = coordinate - bin_index.astype(coordinate.dtype)
        weights = _cubic_weight_matrix(bin_index, fraction, n_source)
        if table.shape[1] == 1:
            row = weights @ table[:, 0]
            row_left = jnp.broadcast_to(left, row.shape)
            row_right = jnp.broadcast_to(right, row.shape)
        else:
            row = jnp.einsum("cs,sc->c", weights, table)
            row_left, row_right = left, right
        row = jax.lax.select(coordinate < 0, row_left, row)
        row = jax.lax.select(coordinate > n_source - 1, row_right, row)
        rows.append(row)
    return jnp.reshape(jnp.stack(rows, 0), log_q.shape)


@jax.jit
def _hankel_device(flat, q, matrix_w, sinc_w, q_native, moment_coeff, lq0, dlq, x_s, scale):
    """Matrix product and interpolation.

    The end weights already sit in the matrices. The cubic is two further
    products: resample onto the query spacing, then a per-column shift.
    """
    u = jnp.matmul(matrix_w, flat) * jnp.sqrt(jnp.pi / (2.0 * q_native))[:, None]
    u_direct = jnp.matmul(sinc_w, flat)
    u = jnp.where(q_native[:, None] <= 1.0 / x_s, u_direct, u)
    log_q = jnp.log(q)
    moment_sum = jnp.matmul(moment_coeff[None, :], flat)[0]
    interpolated = _exact_cubic(u, log_q, lq0, dlq, moment_sum, u[-1])
    return 4.0 * jnp.pi * interpolated * scale


def hankel_transform(x, f, q, x_out=jnp.inf):
    """
    Spherical Hankel transform :math:`F(q) = 4\\pi \\int_0^{x_{\\rm out}} x^2 f(x)\\, j_0(q x)\\, dx` of a function
    sampled on a log-spaced grid, interpolated to ``q``.

    The integral stops at ``x_out`` or at the end of the grid, whichever comes first. At low :math:`q` it is done by
    direct quadrature of the same samples, and :math:`q` below the transform's own grid takes the :math:`q = 0` value.

    Parameters
    ----------
    x : np.ndarray
        Log-spaced sample points, shape ``(n,)``.
    f : jnp.ndarray
        Samples, shape ``(n, ...)``; every trailing axis is an independent function on the same grid.
    q : jnp.ndarray
        Points at which to evaluate the transform, with as many axes as ``f``; trailing axes broadcast against
        those of ``f``.
    x_out : float, optional
        Truncation point; ``f`` is taken to vanish beyond it.

    Returns
    -------
    jnp.ndarray
        Transform at ``q``, with the broadcast shape of ``q`` and the trailing axes of ``f``.
    """
    x64 = np.asarray(x, dtype=np.float64)
    h = np.log(x64[1] / x64[0])
    key = (x64[0], x64[-1], len(x64))
    if key not in _HANKEL_MATRICES:
        # Plan in float64. The FFTLog matrix entries span many decades, and a
        # float32 plan rounds away the digits the transform is a difference of.
        n_pad = int(np.ceil(2.0 * np.log(10.0) / h))
        with jax.ensure_compile_time_eval():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="use backend='jax' if desired")
                hankel = mcfit.Hankel(x64[0] * np.exp(h * np.arange(len(x64) + n_pad)), nu=0.5, lowring=True, backend="numpy")
                q_n, y = hankel(np.pad(np.eye(len(x64)), [(0, 0), (0, n_pad)]), extrap=False)
        q_native = np.asarray(q_n)
        matrix = np.asarray(y).T
        phase = q_native[:, None] * x64[None, :] / np.pi
        sinc = np.sinc(phase)
        _HANKEL_MATRICES[key] = (q_native, matrix, sinc, x64, h)
    q_native, matrix, sinc, x64, h = _HANKEL_MATRICES[key]
    # One scale per column. The integrand is a difference of large samples;
    # dividing that scale out keeps the difference inside the working dtype.
    scale = jnp.max(jnp.abs(f), axis=0)
    scale = jnp.where(scale > 0, scale, 1.0)
    f = f / scale
    # The end-weight is constant for a concrete truncation, so it is folded into
    # the matrix. A device-side mask next to the interpolation compiles to a
    # ~10 ms kernel on TPU; the multiply and the gather alone are ~0.15 ms.
    s = _truncation_steps(x64, h, x_out)
    if isinstance(s, (float, int, np.floating)):
        index = np.arange(len(x64))
        floor_s = np.floor(s)
        weight = np.where(index < floor_s, 1.0, np.where(index == floor_s, 0.5 + s - floor_s, 0.0))
        coeff = weight * np.sqrt(x64)
        moment_coeff = h * weight * x64 ** 3
        matrix_w = matrix * coeff[None, :]
        sinc_w = sinc * moment_coeff[None, :]
        x_s = float(np.max(np.where(weight > 0, x64, 0.0)))
        flat = f.reshape((f.shape[0], -1))
        dtype = flat.dtype
        lq = np.log(q_native)
        return _hankel_device(
            flat,
            q,
            jnp.asarray(matrix_w, dtype=dtype),
            jnp.asarray(sinc_w, dtype=dtype),
            jnp.asarray(q_native),
            jnp.asarray(moment_coeff, dtype=dtype),
            float(lq[0]),
            float(lq[1] - lq[0]),
            x_s,
            scale,
        )
    else:
        col = lambda a: jnp.reshape(a, (-1,) + (1,) * (f.ndim - 1))
        i = col(jnp.arange(len(x64)))
        w = jnp.where(i < jnp.floor(s), 1.0, jnp.where(i == jnp.floor(s), 0.5 + s - jnp.floor(s), 0.0))
        moment = h * w * col(jnp.asarray(x64)) ** 3 * f
        samples = w * col(jnp.asarray(x64)) ** 0.5 * f
        u = jnp.tensordot(jnp.asarray(matrix, dtype=samples.dtype), samples, axes=(1, 0)) * col(jnp.sqrt(jnp.pi / (2.0 * q_native)))
        x_s = jnp.max(jnp.where(w > 0, col(jnp.asarray(x64)), 0.0))
        u_direct = jnp.tensordot(jnp.asarray(sinc, dtype=moment.dtype), moment, axes=(1, 0))
        u = jnp.where(col(q_native) <= 1.0 / x_s, u_direct, u)
        flat_u = jnp.reshape(u, (u.shape[0], -1))
        lq = np.log(q_native)
        log_q = jnp.log(q)
        moment_sum = jnp.reshape(jnp.sum(moment, axis=0), (-1,))
        interpolated = _exact_cubic(
            flat_u, log_q, float(lq[0]), float(lq[1] - lq[0]), moment_sum, jnp.reshape(u[-1], (-1,))
        )
        return 4.0 * jnp.pi * interpolated * scale
