"""CosmoPower emulators of CLASS as a Cosmology."""
import os
import pickle
import zipfile
from dataclasses import dataclass, replace
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from hmfast.cosmology.cosmology import Cosmology
from hmfast import data_path
from hmfast.utils import Const, dopri5_integrate, log_interp1d_extrap


class _NoTFUnpickler(pickle.Unpickler):
    """Unpickler that reads TensorFlow's ListWrapper, pickled into cosmopower .npz files, as a plain list."""

    def find_class(self, module, name):
        if module.startswith("tensorflow") and name == "ListWrapper":
            return list
        return super().find_class(module, name)


class _EmulatorLoader:
    """
    Neural-network emulator from a cosmopower .npz file, with the PCA reconstruction applied when the file has one.

    Parameters
    ----------
    filename : str
        Path to the .npz file, without the extension.
    """

    def __init__(self, filename):
        # The .npz holds one pickled dict; reading it with _NoTFUnpickler avoids importing TensorFlow.
        with zipfile.ZipFile(filename + ".npz") as zf, zf.open("arr_0.npy") as arr:
            np.lib.format.read_magic(arr)
            np.lib.format.read_array_header_1_0(arr)
            fpz = _NoTFUnpickler(arr).load().item()
        as_jnp = lambda x: jnp.asarray(x, dtype=jnp.float64)
        self.parameters = list(fpz["parameters"])
        self.n_layers = fpz["n_layers"]
        self.parameters_mean, self.parameters_std = as_jnp(fpz["param_train_mean"]), as_jnp(fpz["param_train_std"])
        self.features_mean, self.features_std = as_jnp(fpz["feature_train_mean"]), as_jnp(fpz["feature_train_std"])
        self.W = [as_jnp(w) for w in fpz["weights_"]]
        self.b = [as_jnp(b) for b in fpz["biases_"]]
        self.alphas = [as_jnp(a) for a in fpz["alphas_"]]
        self.betas = [as_jnp(b) for b in fpz["betas_"]]
        # (mean, std, transform) of the PCA basis the network outputs coefficients in, or None.
        self.pca = None
        if "pca_transform_matrix" in fpz:
            self.pca = (as_jnp(fpz["pca_mean"]), as_jnp(fpz["pca_std"]), as_jnp(fpz["pca_transform_matrix"]))

    @partial(jax.jit, static_argnums=(0,))
    def forward_pass(self, parameters_arr):
        """Network output for an input array ordered as ``self.parameters``."""
        x = (parameters_arr - self.parameters_mean) / self.parameters_std
        for i in range(self.n_layers - 1):
            act = jnp.dot(x, self.W[i]) + self.b[i]
            x = (self.betas[i] + (1.0 - self.betas[i]) * 1.0 / (1.0 + jnp.exp(-self.alphas[i] * act))) * act
        out = jnp.dot(x, self.W[-1]) + self.b[-1]
        if self.pca is not None:
            pca_mean, pca_std, transform = self.pca
            out = jnp.dot(out * pca_std + pca_mean, transform)
        return out * self.features_std + self.features_mean

    def predictions(self, parameters_dict):
        """Network output for a dict of the emulator's input parameters."""
        return self.forward_pass(jnp.stack([parameters_dict[k] for k in self.parameters], axis=-1))

    def ten_to_predictions(self, parameters_dict):
        """10 to the network output, for emulators trained on log10 quantities."""
        return 10.0 ** self.predictions(parameters_dict)


# Parameter -> (emulator input name, transform into it, fiducial value, training range); N_ur's fiducial is the set's.
_PARAMS = {
    "H0": ("H0", None, 68.0, (39.99, 100.01)),
    "omega_cdm": ("omega_cdm", None, 0.12, (0.08, 0.20)),
    "omega_b": ("omega_b", None, 0.02246576, (0.01933, 0.02533)),
    "A_s": ("ln10^{10}A_s", lambda x: jnp.log(1e10 * x), 2.1053e-9, (np.exp(2.5) / 1e10, np.exp(3.5) / 1e10)),
    "n_s": ("n_s", None, 0.965, (0.8, 1.2)),
    "tau": ("tau_reio", None, 0.0544, (0.02, 0.12)),
    "m_ncdm": ("m_ncdm", None, 0.06, (0.0, 2.0)),
    "N_ur": ("N_ur", None, None, (0.49, 4.49)),
    "w0": ("w0_fld", None, -1.0, (-2.0, -0.33)),
    "f_ede": ("fEDE", None, 0.1, (0.001, 0.5)),
    "z_c": ("log10z_c", jnp.log10, 3162.278, (1e3, 10**4.3)),
    "theta_i": ("thetai_scf", None, 1.57, (0.1, 3.1)),
    "r": ("r", None, 0.01, (0.0, 0.3)),
}


@dataclass(frozen=True)
class _EmulatorSet:
    """One emulator set with its settings; it is the CosmoPowerCosmology's functions, compared by value in treedefs."""

    name: str
    subdir: str
    suffix: str
    free: tuple = ()
    deg_ncdm: float = 1.0
    N_ur: float = 2.0328
    bounds: tuple = (("n_s", (0.8, 1.2)),)  # training ranges that differ from _PARAMS
    z_max_pk: float = 5.0
    v2: bool = False  # log10 background outputs, and P(k) k^3 on 5e-4 < k < 10 Mpc^-1
    extrapolate_z: bool = False
    extrapolate_k: bool = True

    z_bg: ClassVar = np.linspace(0.0, 20.0, 5000)  # redshifts of the emulated background
    _networks: ClassVar = {}  # (subdir, suffix, key) -> network, loaded once per process

    # --- parameters ---

    @property
    def names(self):
        return ("H0", "omega_cdm", "omega_b", "A_s", "n_s", "tau", *self.free)

    def fiducial(self, name):
        return self.N_ur if name == "N_ur" else _PARAMS[name][2]

    def fixed(self):
        """Parameters the set fixes: flat, w0 = -1, wa = 0, and the fiducial neutrinos when they are not free."""
        fixed = {"Omega_k": 0.0, "m_ncdm": 0.06, "w0": -1.0, "wa": 0.0, "T_cmb": 2.7255, "T_ncdm": 0.71611,
                 "N_ur": self.N_ur}
        return {n: v for n, v in fixed.items() if n not in self.names}

    def inputs(self, p):
        """Parameters under the emulators' input names."""
        return {emu: (f(p[n]) if f else p[n]) for n, (emu, f, _, _) in _PARAMS.items() if n in p}

    def mask(self, p, x):
        """NaN wherever the parameters lie outside the training ranges."""
        bounds = {n: b for n, (_, _, _, b) in _PARAMS.items()} | dict(self.bounds)
        valid = True
        for n in self.names:
            lo, hi = bounds[n]
            valid = valid & (p[n] >= lo) & (p[n] <= hi)
        return jnp.where(valid, x, jnp.nan)

    # --- networks and their grids ---

    def network(self, key):
        folder = {"DER": "derived-parameters", "PKL": "PK", "PKNL": "PK", "TT": "TTTEEE", "EE": "TTTEEE",
                  "TE": "TTTEEE", "PP": "PP"}.get(key, "growth-and-distances")
        cache_key = (self.subdir, self.suffix, key)
        if cache_key not in self._networks:
            # Keep the weights concrete even when first reached under a trace.
            with jax.ensure_compile_time_eval():
                path = os.path.join(data_path(), self.subdir, folder, f"{key}_{self.suffix}")
                self._networks[cache_key] = _EmulatorLoader(path)
        return self._networks[cache_key]

    def k_grid(self):
        """Wavenumbers of the emulated spectra, and the factor that turns the network output into P(k)."""
        if self.v2:
            k = np.geomspace(5e-4, 10.0, 1000)
            return k, k ** -3
        ell = np.arange(2, 5002)[::10]
        return np.geomspace(1e-4, 50.0, 5000)[::10], (ell * (ell + 1.0) / (2.0 * np.pi)) ** -1

    def background(self, key, p):
        """Emulated H (km/s/Mpc), D_A (Mpc) or sigma_8 on z_bg."""
        y = self.network(key).predictions(self.inputs(p))
        if key == "HZ":
            return 10.0 ** y * (Const.c / 1e3)
        if key == "DAZ" and self.v2:
            return jnp.insert(10.0 ** y, 0, 0.0)
        return 10.0 ** y if self.v2 else y

    def on_z_bg(self, key, z, p):
        return self.mask(p, jnp.interp(z, self.z_bg, self.background(key, p), left=jnp.nan, right=jnp.nan))

    def emulated_pk(self, key, k, z, p):
        inputs = self.inputs(p)
        k_grid, pk_fac = self.k_grid()

        def one_z(z_i):
            pk = 10.0 ** self.network(key).predictions({**inputs, "z_pk_save_nonclass": z_i}) * pk_fac
            return log_interp1d_extrap(k, k_grid, pk)

        return self.mask(p, jax.vmap(one_z, out_axes=1)(z))

    # --- beyond the emulated redshifts ---

    def cosmology(self, p):
        """CosmoPowerCosmology with parameters ``p``, for its densities."""
        return CosmoPowerCosmology(self.name, **p, extrapolate_z=self.extrapolate_z, extrapolate_k=self.extrapolate_k)

    def hubble_flrw(self, z, p, z_cal):
        """FLRW H(z) from rho_x, rescaled to the emulated H at ``z_cal``."""
        cosmo = self.cosmology(p)
        hz = lambda zz: p["H0"] * jnp.sqrt(cosmo.rho_x(zz, "critical") / cosmo.rho_x(0.0, "critical"))
        return hz(z) * self.on_z_bg("HZ", z_cal, p) / hz(z_cal)

    def ln_growth(self, z, p):
        """ln D(z), D(0) = 1, from the linear growth equation for baryons and CDM."""
        H0, cosmo = p["H0"], self.cosmology(p)
        hz = lambda zz: self.hubble_flrw(zz, p, self.z_max_pk)
        rho_crit_0 = cosmo.rho_x(0.0, "critical")
        rho_cb = lambda zz: cosmo.rho_x(zz, "matter") - cosmo.rho_x(zz, "neutrinos_massive")

        # Seed well past any requested z so the solution has room to relax onto the growing mode.
        z_start = jax.lax.stop_gradient(4.0 * jnp.maximum(jnp.max(z), self.z_max_pk))
        x_start = -jnp.log1p(z_start)
        # Growing-mode rate with radiation and matter (Meszaros), f = 1.5 y / (1 + 1.5 y) for y = rho_cb / rho_r.
        y = rho_cb(z_start) / (cosmo.rho_x(z_start, "radiation") + cosmo.rho_x(z_start, "neutrinos_rel"))
        f_start = 1.5 * y / (1.0 + 1.5 * y)

        def growth_coeffs(x):
            dlnH_dx = jax.vmap(jax.grad(lambda xx: jnp.log(hz(jnp.expm1(-xx)))))(x)
            zz = jnp.expm1(-x)
            return 2.0 + dlnH_dx, 1.5 * rho_cb(zz) / rho_crit_0 * H0 ** 2 / hz(zz) ** 2

        def growth_rhs(c, y):
            (A, S), (_, f) = c, y
            return f, -f ** 2 - A * f + S

        x_traj, (lnD_traj, f_traj) = dopri5_integrate(growth_rhs, growth_coeffs, (jnp.array(0.0), f_start), x_start,
                                                      0.0, n_steps=64)
        lnD = self.hermite(-jnp.log1p(z), x_traj, lnD_traj, f_traj)
        return lnD - jnp.interp(0.0, x_traj, lnD_traj)

    @staticmethod
    def hermite(x, x_nodes, y, dydx):
        """Cubic Hermite interpolation in x <= 0, so the result is smooth and its derivative matches dydx at nodes."""
        # Clamps by jnp.where, not minimum/clip, so the derivative is not halved at the boundaries (z = 0 or a node).
        x = jnp.where(x > 0.0, 0.0, x)
        # side="left" keeps x = 0 in the last real interval, not in the zero-length ones after convergence.
        i = jnp.clip(jnp.searchsorted(x_nodes, x, side="left") - 1, 0, x_nodes.shape[0] - 2)
        h = x_nodes[i + 1] - x_nodes[i]
        h_safe = jnp.where(h > 0, h, 1.0)  # nodes after convergence repeat x = 0
        t = (x - x_nodes[i]) / h_safe
        t = jnp.where(t < 0.0, 0.0, jnp.where(t > 1.0, 1.0, t))
        return ((2 * t**3 - 3 * t**2 + 1) * y[i] + (t**3 - 2 * t**2 + t) * h * dydx[i]
                + (-2 * t**3 + 3 * t**2) * y[i + 1] + (t**3 - t**2) * h * dydx[i + 1])

    # --- the five functions every Cosmology needs ---

    def hubble_parameter(self, z, p):
        z_max = float(self.z_bg[-1])
        hz = self.on_z_bg("HZ", jnp.minimum(z, z_max), p)
        fill = self.hubble_flrw(z, p, z_max) if self.extrapolate_z else jnp.nan
        return jnp.where(z > z_max, fill, hz)

    def angular_diameter_distance(self, z, p):
        z_max = float(self.z_bg[-1])
        da = self.on_z_bg("DAZ", jnp.minimum(z, z_max), p)
        if not self.extrapolate_z:
            return jnp.where(z > z_max, jnp.nan, da)
        d_m_max = self.on_z_bg("DAZ", z_max, p) * (1.0 + z_max)
        c_kms = Const.c / 1e3
        curv = self.cosmology(p).Omega_k * (p["H0"] / c_kms) ** 2
        dchi_dx = lambda x: c_kms * (1.0 + jnp.expm1(x)) / self.hubble_flrw(jnp.expm1(x), p, z_max)
        # Transverse distance D_M and its slope dD_M/dchi, from D_M'' = Omega_k (H0/c)^2 D_M
        rhs = lambda c, y: (c * y[1], c * curv * y[0])
        y_max = (d_m_max, jnp.sqrt(1.0 + curv * d_m_max ** 2))

        # Integrate to each z independently -- a shared trajectory would be too sparse to interpolate safely.
        def d_m_at(z_target):
            _, (d_m_traj, _) = dopri5_integrate(rhs, dchi_dx, y_max, jnp.log1p(z_max), jnp.log1p(z_target), n_steps=16)
            return d_m_traj[-1]

        return jnp.where(z > z_max, jax.vmap(d_m_at)(z) / (1.0 + z), da)

    def growth_factor(self, z, p):
        D = self.mask(p, jnp.exp(self.ln_growth(z, p)))
        return D if self.extrapolate_z else jnp.where(z > self.z_max_pk, jnp.nan, D)

    def pk_linear(self, k, z, p):
        return self.pk("PKL", k, z, p)

    def pk_nonlinear(self, k, z, p):
        return self.pk("PKNL", k, z, p)

    def pk(self, key, k, z, p):
        pk = self.emulated_pk(key, k, jnp.minimum(z, self.z_max_pk), p)
        if self.extrapolate_z:
            lnD = self.ln_growth(jnp.append(z, self.z_max_pk), p)
            growth_ratio = jnp.where(z > self.z_max_pk, jnp.exp(lnD[:-1] - lnD[-1]), 1.0)
            pk = pk * growth_ratio[None, :] ** 2
        else:
            pk = jnp.where(z[None, :] > self.z_max_pk, jnp.nan, pk)
        if not self.extrapolate_k:
            k_grid = self.k_grid()[0]
            pk = jnp.where(((k >= k_grid[0]) & (k <= k_grid[-1]))[:, None], pk, jnp.nan)
        return pk


_EDE = ("m_ncdm", "N_ur", "f_ede", "z_c", "theta_i", "r")
_EDE_BOUNDS = (("n_s", (0.8, 1.2)), ("m_ncdm", (0.0, 0.33333)), ("N_ur", (-2.0, 2.5)))
_SETS = {s.name: s for s in (
    _EmulatorSet("lcdm:v1", "lcdm", "v1", bounds=(("n_s", (0.8812, 1.0492)),)),
    _EmulatorSet("mnu:v1", "mnu", "mnu_v1", free=("m_ncdm",)),
    _EmulatorSet("neff:v1", "neff", "neff_v1", free=("N_ur",)),
    _EmulatorSet("wcdm:v1", "wcdm", "w_v1", free=("w0",)),
    _EmulatorSet("ede:v1", "ede", "v1", free=_EDE, deg_ncdm=3.0, N_ur=0.00641, bounds=_EDE_BOUNDS),
    _EmulatorSet("mnu-3states:v1", "mnu-3states", "v1", free=("m_ncdm",), deg_ncdm=3.0, N_ur=0.00641),
    _EmulatorSet("ede:v2", "ede", "v2", free=_EDE, deg_ncdm=3.0, N_ur=0.00641, bounds=_EDE_BOUNDS, z_max_pk=20.0,
                 v2=True),
)}


class CosmoPowerCosmology(Cosmology):
    """
    Cosmology from the CosmoPower emulators of CLASS: background, linear and nonlinear (HMcode) :math:`P(k)`,
    :math:`\\sigma_8(z)`, CMB spectra and derived parameters. Outputs are NaN outside the training ranges.
    Inherits from :class:`~hmfast.cosmology.Cosmology`, with the emulators as its functions.

    Every set varies ``H0``, ``omega_cdm``, ``omega_b``, ``A_s``, ``n_s`` and ``tau``, plus:

    ==================  ===============================================  ==================  ==============
    Set                 Extra parameters                                 Massive states      ``z_max_pk``
    ==================  ===============================================  ==================  ==============
    ``lcdm:v1``         --                                               1 (0.06 eV)         5
    ``mnu:v1``          ``m_ncdm``                                       1                   5
    ``mnu-3states:v1``  ``m_ncdm`` (per state)                           3                   5
    ``neff:v1``         ``N_ur``                                         1 (0.06 eV)         5
    ``wcdm:v1``         ``w0``                                           1 (0.06 eV)         5
    ``ede:v1``          ``m_ncdm``, ``N_ur``, ``f_ede``, ``z_c``,        3                   5
                        ``theta_i``, ``r``
    ``ede:v2``          as ``ede:v1``                                    3                   20
    ==================  ===============================================  ==================  ==============

    The background is valid to :math:`z = 20`. The set's ``z_max_pk`` is also its
    :attr:`~hmfast.cosmology.Cosmology.z_max_pk`, above which projected statistics scale the halo model by the
    linear growth factor. The growth factor solves the linear growth equation for baryons and cold dark matter.

    Parameters
    ----------
    emulator_set : str
        Emulator set, one of the table above (default ``"lcdm:v1"``).
    H0, omega_cdm, omega_b, A_s, n_s, tau, m_ncdm, N_ur, w0, f_ede, z_c, theta_i, r : float or None
        As in :class:`~hmfast.cosmology.Cosmology`, for the parameters the set varies (table above); None takes the
        fiducial value (``tau`` 0.0544, ``m_ncdm`` 0.06, ``N_ur`` giving :math:`N_\\mathrm{eff} = 3.046`, ``w0`` -1,
        ``f_ede`` 0.1, ``z_c`` 3162.278, ``theta_i`` 1.57, ``r`` 0.01). A set that does not vary ``m_ncdm``, ``N_ur`` or ``w0`` fixes them at 0.06 eV,
        2.0328 (0.00641 for ``mnu-3states:v1``, so :math:`N_\\mathrm{eff} = 3.046`) and -1, and has no ``f_ede``,
        ``z_c``, ``theta_i`` or ``r``; setting any of these raises a ``TypeError``.
        Every set is flat (``Omega_k`` = 0) with ``wa`` = 0, ``T_cmb`` = 2.7255 K and ``T_ncdm`` = 0.71611.
    ncdm_mode : {"cb", "m"}
        As in :class:`~hmfast.cosmology.Cosmology`.
    extrapolate_k : bool
        If True (default), :math:`P(k)` is extrapolated as a power law in log-log beyond the emulators'
        wavenumber range; if False, it is NaN there.
    extrapolate_z : bool
        If True, redshifts above the emulators' range are extrapolated with an FLRW :math:`H(z)` from
        :meth:`~hmfast.cosmology.Cosmology.rho_x`, matched to the emulators at the top of the range, and
        :math:`P(k)` scaled by the linear growth factor; if False (default), they are NaN. This is less
        accurate for early dark energy models.

    Examples
    --------
    >>> cosmo = CosmoPowerCosmology("ede:v2", f_ede=0.08)
    """

    def __init__(self, emulator_set="lcdm:v1", *,
                 H0=68.0, omega_cdm=0.12, omega_b=0.02246576, A_s=2.1053e-9, n_s=0.965, tau=0.0544,
                 m_ncdm=None, N_ur=None, w0=None, f_ede=None, z_c=None, theta_i=None, r=None,
                 ncdm_mode="cb", extrapolate_k=True, extrapolate_z=False):
        if emulator_set not in _SETS:
            raise ValueError(f"Unknown emulator set {emulator_set!r}. Allowed: {', '.join(_SETS)}.")
        emu = replace(_SETS[emulator_set], extrapolate_z=extrapolate_z, extrapolate_k=extrapolate_k)
        named = dict(H0=H0, omega_cdm=omega_cdm, omega_b=omega_b, A_s=A_s, n_s=n_s, tau=tau,
                     m_ncdm=m_ncdm, N_ur=N_ur, w0=w0, f_ede=f_ede, z_c=z_c, theta_i=theta_i, r=r)
        given = {n: v for n, v in named.items() if v is not None}
        k = emu.k_grid()[0]
        self._setup({n: given.get(n, emu.fiducial(n)) for n in emu.names}, emu, (k[0], k[-1]), len(k), ncdm_mode,
                    emu.z_max_pk)
        unknown = [n for n in given if n not in emu.names]
        if unknown:
            raise TypeError(self._not_parameters(unknown))
        if os.environ.get("READTHEDOCS") != "True":
            for key in ("HZ", "DAZ", "S8Z", "PKL", "PKNL", "DER"):
                emu.network(key)

    @property
    def emulator_set(self):
        return self._functions.name

    @property
    def extrapolate_z(self):
        return self._functions.extrapolate_z

    @property
    def extrapolate_k(self):
        return self._functions.extrapolate_k

    def __repr__(self):
        return f"CosmoPowerCosmology({self.emulator_set!r}, {self._repr_values()})"

    def update(self, *, extrapolate_z=None, extrapolate_k=None, **changes):
        """
        Return a new cosmology with some parameters or settings changed.

        Parameters
        ----------
        extrapolate_z, extrapolate_k : bool or None
            New values of these settings; None leaves a value unchanged.
        **changes
            New values of the set's free parameters, the differentiable leaves: ``H0``, ``omega_cdm``, ``omega_b``,
            ``A_s``, ``n_s`` and ``tau`` in every set, plus ``m_ncdm`` (``mnu:v1``, ``mnu-3states:v1``, ``ede:*``),
            ``N_ur`` (``neff:v1``, ``ede:*``), ``w0`` (``wcdm:v1``) and ``f_ede``, ``z_c``, ``theta_i``, ``r``
            (``ede:*``). None leaves a value unchanged.

        Returns
        -------
        CosmoPowerCosmology
            New instance with the same emulator set.

        Examples
        --------
        >>> cosmo = CosmoPowerCosmology("ede:v2").update(f_ede=0.08, extrapolate_z=True)
        """
        obj = super().update(**changes)
        settings = {"extrapolate_z": extrapolate_z, "extrapolate_k": extrapolate_k}
        obj._functions = replace(obj._functions, **{n: v for n, v in settings.items() if v is not None})
        return obj

    def _fixed_params(self):
        return self._functions.fixed()

    def _label(self):
        return repr(self.emulator_set)

    @property
    def _deg_ncdm(self):
        return self._functions.deg_ncdm

    def _sigma8(self, z):
        return self._squeeze_single(self._functions.on_z_bg("S8Z", jnp.atleast_1d(z), self.params))

    # ------------------------------------------------------------------
    # CMB angular power spectra and derived parameters
    # ------------------------------------------------------------------

    def cl_cmb(self, type, l):
        """
        Evaluate the CMB power spectrum of the specified type at requested multipoles `l`.
        This method can be used to evaluate :math:`C_\\ell^{TT}`, :math:`C_\\ell^{EE}`, :math:`C_\\ell^{TE}`, and :math:`C_\\ell^{\\phi\\phi}` by passing the appropriate `type` argument.

        Parameters
        ----------
        type : str
            Power-spectrum specifier, e.g. 'TT', 'EE', 'TE', or 'PP'. Case-insensitive.
        l : int or array-like
            Multipole(s) at which to evaluate C_ell.

        Returns
        -------
        jnp.ndarray
            C_ell for the requested type evaluated at `l`. Out-of-range `l` return NaN.
        """
        s = str(type).upper()
        if s not in ("TT", "EE", "TE", "PP"):
            raise ValueError(f"Unsupported spectrum type: {type}")

        return self._cl_jit(s, l)

    @partial(jax.jit, static_argnums=(1,))
    def _cl_jit(self, s, l):
        emu = self._functions
        net = emu.network(s)
        inputs = emu.inputs(self.params)
        cl = net.predictions(inputs) if s == "TE" else net.ten_to_predictions(inputs)
        if s == "PP":
            cl = cl / (2.0 * jnp.pi)
        ell = jnp.arange(2, cl.shape[0] + 2)
        cl = jnp.interp(jnp.atleast_1d(l), ell, cl, left=jnp.nan, right=jnp.nan)
        return jnp.squeeze(emu.mask(self.params, cl))

    @jax.jit
    def derived_parameters(self):
        """
        Get derived cosmological parameters from the emulator.

        Returns
        -------
        dict
            Dictionary of derived parameters with the following keys:

            - '100*theta_s' : Sound horizon angle (in units of 1/100 radians)
            - 'sigma8' : Dimensionless RMS linear matter fluctuation in 8 Mpc/h spheres
            - 'YHe' : Primordial helium fraction
            - 'z_reio' : Redshift of reionization
            - 'Neff' : Effective number of relativistic species
            - 'tau_rec' : Conformal time at recombination (maximum visibility)
            - 'z_rec' : Redshift at recombination (maximum visibility)
            - 'rs_rec' : Comoving sound horizon at recombination [Mpc]
            - 'chi_rec' : Comoving distance to recombination [Mpc]
            - 'tau_star' : Conformal time at last scattering (optical depth = 1)
            - 'z_star' : Redshift at last scattering (optical depth = 1)
            - 'rs_star' : Comoving sound horizon at last scattering [Mpc]
            - 'chi_star' : Comoving distance to last scattering [Mpc]
            - 'rs_drag' : Comoving sound horizon at baryon drag [Mpc]
        """
        names = ("100*theta_s", "sigma8", "YHe", "z_reio", "Neff", "tau_rec", "z_rec", "rs_rec", "chi_rec",
                 "tau_star", "z_star", "rs_star", "chi_star", "rs_drag")
        emu = self._functions
        values = emu.network("DER").ten_to_predictions(emu.inputs(self.params))
        return {n: emu.mask(self.params, v) for n, v in zip(names, values)}
