"""
Cobaya theory module: hmfast masked (unresolved) tSZ D_ell with the
parametric pressure-profile amplitude and the lognormal-scatter
completeness (matching tszpower's ``tSZ_PS_Theory_Scatter`` semantics).

This sets up the same forward model as
``tszpower.tszpower_cobaya_theory_masked_scatter.tSZ_PS_Theory_Scatter`` so
the existing chain YAML
(``tszpower_scatter_masked_signal_only_chain.input.yaml``) can be
re-pointed at this theory and the resulting posterior compared.

Implementation notes
--------------------
- The mask is the conditional second moment
  :math:`\\langle A^2 \\mathbf{1}(q_{\\rm obs} < q_{\\rm cat}) \\rangle`,
  computed by ``hmfast.tracers.tsz_completeness.conditional_An_undetected``.
- SNR uses the same Planck SZiFi noise files and the same polynomial-fit
  noise curve as tszpower (default ``poly_deg=3``).
- The internal :math:`(M, z)` grid is geomspace (M_min..M_max) × geomspace
  (z_min..z_max) at ``n_grid_mz=100`` to match tszpower.

Outputs ``D_\\ell \\times 10^{12}`` in 18 Planck bands, matching the
``tszpower.maskedpower.ELL_EFF`` convention exactly.
"""
from __future__ import annotations

import os
import time

# JAX device selection — same conventions as hmfast_cobaya_theory_fullsky.py
if os.environ.get("HMFAST_COBAYA_USE_GPU", "1").strip() not in ("1", "true", "True", "yes", "YES"):
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Clear any persistent-cache XLA flag that may have been inherited
# (some users set XLA_FLAGS for non-JAX TF builds; it breaks JAX).
xf = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_persistent_cache_dir" in xf:
    os.environ.pop("XLA_FLAGS")

import numpy as np
import jax
import jax.numpy as jnp
from cobaya.theory import Theory

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import ParametricGNFWPressureProfile
from hmfast.halos.mass_definition import MassDefinition
from hmfast.tracers import tSZTracer
from hmfast.tracers.tsz_completeness import (
    build_snr_grid,
    conditional_An_undetected,
    load_sigma_y0_curve,
    DEFAULT_SIGMA_OBJ_FILE,
    DEFAULT_SKYFR_FILE,
)


# Planck-style band edges (must match tszpower's ELL_MIN/ELL_MAX exactly)
_ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78,
     102, 133, 173, 224, 292, 380, 494, 642, 835], dtype=int
)
_ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102,
     133, 173, 224, 292, 380, 494, 642, 835, 1085], dtype=int
)
_ELL_EFF = np.array(
    [10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5,
     117.0, 152.5, 198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5]
)
# Padded integer-ell grid for the "interpolate then average" binner.
_L_MAX = int(np.max(_ELL_MAX - _ELL_MIN + 1))
_ELL_INT = _ELL_MIN[:, None] + np.arange(_L_MAX)[None, :]
_ELL_MASK = (_ELL_INT <= _ELL_MAX[:, None]).astype(np.float64)


def _bin_to_18(ell_in, Cl_in):
    """tszpower-compatible binning: interpolate Cl onto integer ell in each
    band, convert to D_ell, average uniformly within each band."""
    log_ell = np.log(np.asarray(ell_in))
    Cl_q = np.interp(np.log(_ELL_INT.astype(float)), log_ell, np.asarray(Cl_in))
    Dl_q = _ELL_INT * (_ELL_INT + 1.0) * Cl_q / (2.0 * np.pi) * 1e12
    num = np.sum(Dl_q * _ELL_MASK, axis=1)
    den = np.sum(_ELL_MASK, axis=1)
    return num / den


# Fiducial point for one-time HaloModel build + JIT warmup. Pulled from the
# chain YAML so the JIT specialisation is reasonable.
_FIDUCIAL = dict(
    H0=73.8, omega_cdm=0.119, omega_b=0.022,
    ln10_10A_s=2.9718, n_s=0.962, tau_reio=0.0544,
)
_FID_B = 1.0 / 0.709    # 1 / (1-b) per chain YAML
_FID_A_SZ = -4.31
_FID_ALPHA_SZ = 1.12
_FID_SIGMA_LNY = 0.173


class HMFastTSZMaskedScatter(Theory):
    """
    Masked (unresolved) tSZ D_ell with parametric profile + lognormal scatter.

    Outputs ``Cl_sz["1h"]``: (18,) array of :math:`D_\\ell \\times 10^{12}`
    binned to ``_ELL_EFF`` exactly like
    ``tszpower.maskedpower.bin_Dl_to_18``.
    """

    output = ["Cl_sz"]

    params = {
        "omega_b": 0, "omega_cdm": 0, "H0": 0, "tau_reio": 0,
        "ln10_10A_s": 0, "n_s": 0, "B": 0,
        "A_SZ": 0, "alpha_SZ": 0, "sigma_lnY": 0,
    }

    # Fixed selection / catalogue params, overridden in YAML
    q_cat: float = 5.0

    # Noise curves
    sigma_obj_file: str = DEFAULT_SIGMA_OBJ_FILE
    skyfr_file: str = DEFAULT_SKYFR_FILE
    filter_name: str = "immf6"
    theta_min: float = 0.5
    theta_max: float = 32.0
    poly_deg: int = 3

    # Internal grids — match tszpower's N_GRID_MZ=100
    n_grid_mz: int = 100
    n_grid_scatter: int = 1024
    nsig_scatter: float = 8.0

    # Mass / redshift ranges — match the chain YAML
    M_min: float = 1e14 * 0.6766
    M_max: float = 1e16 * 0.6766
    z_min: float = 5e-3
    z_max: float = 3.0

    # Internal ell grid used before binning to 18 bands
    n_ell_internal: int = 50

    def get_requirements(self):
        return {k: None for k in self.params}

    def initialize(self):
        # Load the noise curve once (cached at the module level)
        coeff, _ = load_sigma_y0_curve(
            sigma_obj_file=self.sigma_obj_file,
            skyfr_file=self.skyfr_file,
            filter_name=self.filter_name,
            theta_min=self.theta_min, theta_max=self.theta_max,
            poly_deg=self.poly_deg,
        )
        self._sigma_coeff = jnp.asarray(coeff)

        # Geometry-pinned grids
        self._ell_int = jnp.geomspace(
            float(_ELL_MIN[0]), float(_ELL_MAX[-1]), int(self.n_ell_internal)
        )
        self._m = jnp.geomspace(self.M_min, self.M_max, self.n_grid_mz)
        self._z = jnp.geomspace(self.z_min, self.z_max, self.n_grid_mz)

        # Seed cosmology + halo model
        self._cosmo_seed = Cosmology(emulator_set="lcdm:v1")
        cosmo_fid = self._cosmo_seed.update(
            H0=_FIDUCIAL["H0"], omega_cdm=_FIDUCIAL["omega_cdm"],
            omega_b=_FIDUCIAL["omega_b"], ln1e10A_s=_FIDUCIAL["ln10_10A_s"],
            n_s=_FIDUCIAL["n_s"], tau_reio=_FIDUCIAL["tau_reio"],
        )
        # M_500c definition matches tszpower's M grid convention
        self._hm_seed = HaloModel(
            cosmology=cosmo_fid,
            mass_definition=MassDefinition(500, "critical"),
            convert_masses=True,
        )
        self._prof_seed = ParametricGNFWPressureProfile(
            A_SZ=_FID_A_SZ, alpha_SZ=_FID_ALPHA_SZ, B=_FID_B,
        )
        self._tsz = tSZTracer(profile=self._prof_seed)

        # JIT warmup at the fiducial point and at a small offset
        block = getattr(jax, "block_until_ready", lambda x: x)
        t0 = time.perf_counter()
        block(self._eval_masked_cl(
            _FIDUCIAL["H0"], _FIDUCIAL["omega_cdm"], _FIDUCIAL["omega_b"],
            _FIDUCIAL["ln10_10A_s"], _FIDUCIAL["n_s"], _FIDUCIAL["tau_reio"],
            _FID_A_SZ, _FID_ALPHA_SZ, _FID_SIGMA_LNY, _FID_B,
        ))
        block(self._eval_masked_cl(
            _FIDUCIAL["H0"] + 0.1, _FIDUCIAL["omega_cdm"], _FIDUCIAL["omega_b"],
            _FIDUCIAL["ln10_10A_s"], _FIDUCIAL["n_s"], _FIDUCIAL["tau_reio"],
            _FID_A_SZ + 0.01, _FID_ALPHA_SZ + 0.01, _FID_SIGMA_LNY, _FID_B,
        ))
        self.log.info(
            "HMFastTSZMaskedScatter: init + JIT warmup in %.4fs",
            time.perf_counter() - t0,
        )
        super().initialize()

    def _eval_masked_cl(self, H0, omega_cdm, omega_b, ln10_10A_s, n_s,
                        tau_reio, A_SZ, alpha_SZ, sigma_lnY, B):
        cosmo = self._cosmo_seed.update(
            H0=H0, omega_cdm=omega_cdm, omega_b=omega_b,
            ln1e10A_s=ln10_10A_s, n_s=n_s, tau_reio=tau_reio,
        )
        hm = self._hm_seed.update(cosmology=cosmo)
        prof = self._prof_seed.update(A_SZ=A_SZ, alpha_SZ=alpha_SZ, B=B)
        tsz = self._tsz.update(profile=prof)

        snr = build_snr_grid(
            hm, self._m, self._z,
            A_SZ=A_SZ, alpha_SZ=alpha_SZ, B=B,
            coeff=self._sigma_coeff,
        )
        mask = conditional_An_undetected(
            snr, sigma_lnY=sigma_lnY, q_cat=self.q_cat,
            n_power=2, n_grid=self.n_grid_scatter, nsig=self.nsig_scatter,
        )
        cl_int = hm.cl_1h_masked(
            tsz, None, self._ell_int, self._m, self._z, mask, k_damp=0.0,
        )
        return cl_int

    def calculate(self, state, want_derived=True, **params_values):
        t0 = time.perf_counter()
        cl_int = self._eval_masked_cl(
            float(params_values["H0"]),
            float(params_values["omega_cdm"]),
            float(params_values["omega_b"]),
            float(params_values["ln10_10A_s"]),
            float(params_values["n_s"]),
            float(params_values["tau_reio"]),
            float(params_values["A_SZ"]),
            float(params_values["alpha_SZ"]),
            float(params_values["sigma_lnY"]),
            float(params_values["B"]),
        )
        block = getattr(jax, "block_until_ready", lambda x: x)
        block(cl_int)

        Dl_binned = _bin_to_18(np.asarray(self._ell_int), np.asarray(cl_int))
        state["Cl_sz"] = {"1h": Dl_binned, "2h": np.zeros_like(Dl_binned)}
        self._current_state = state
        self.log.info(
            "HMFastTSZMaskedScatter: masked PS computed in %.4fs",
            time.perf_counter() - t0,
        )

    def get_Cl_sz(self):
        return self._current_state.get("Cl_sz", None)
