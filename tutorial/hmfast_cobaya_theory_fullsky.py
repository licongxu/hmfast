"""
Cobaya theory module: hmfast full-sky tSZ binned D_ell (tutorial).

Maps LCDM parameters to :class:`hmfast.cosmology.Cosmology` (emulator_set lcdm:v1),
builds :class:`hmfast.halos.HaloModel` and :class:`hmfast.tracers.tSZTracer`, and
returns binned :math:`10^{12}\,D_\ell` at the ell centers of the data vector file
(same units as the scatter ``synthetic_data`` binned tSZ spectra; native hmfast
:math:`D_\ell` is multiplied by :data:`_DL_ELL_DATA_SCALE` before the likelihood).

Set environment variable HMFAST_COBAYA_USE_GPU=0 before importing this module
(e.g. in the shell that runs ``cobaya-run``) to force JAX on CPU.
Default is to allow GPU (HMFAST_COBAYA_USE_GPU=1 or unset).

**GPU selection:** ``JAX_PLATFORMS`` defaults to ``cuda``. ``CUDA_VISIBLE_DEVICES``:
non-MPI or ``mpirun -np 1`` defaults to ``0``. With **MPI** (``world_size > 1``), each
rank is mapped to a physical GPU by ``rank % HMFAST_COBAYA_NUM_GPUS`` (default **2**),
so ``mpirun -np 2`` uses GPU 0 and GPU 1 when two cards exist. Set
``HMFAST_COBAYA_NUM_GPUS=1`` to pin every rank to GPU 0 (shared card). Override
``CUDA_VISIBLE_DEVICES`` in the shell **before** Python if you need a custom layout.

**JAX GPU memory:** By default each process uses ``MEM_FRACTION=0.5`` (50% of the
**visible** device’s memory). With MPI, if several ranks share one GPU, preallocation is
off and each rank’s fraction is ``0.5 / ranks_per_gpu`` (so total JAX use on that card
stays near 50% when possible, with a small floor to avoid pathological values).

**Low VRAM in nvtop** with moderate GPU utilization is normal for this pipeline.


**MPI vs single-process timing:** ``mpirun -np N`` runs **N independent MCMC
chains** (one JAX/Cobaya process per rank). Each rank evaluates ``calculate`` on
its **own** proposals; work is **not** divided across ranks for one likelihood call.

**Why ``-np 4`` on *one* GPU does not give 4× wall-clock sampling vs ``-np 1``:**
a single GPU runs JAX/XLA work largely **serially** through one execution queue.
Four ranks **share** that device, so each rank’s step is **slower** than exclusive
use (~0.02 s vs ~0.005 s is typical). **Aggregate samples per second** across all
chains is often **similar** to one chain on one GPU—you mainly gain **four parallel
chains** (mixing, Gelman–Rubin), not four times faster throughput per GPU.

**To keep ~single-rank step time *per chain* while running four chains:** give each
MPI rank **its own GPU** (e.g. ``CUDA_VISIBLE_DEVICES`` per rank, or job scheduler
GPU binding). Then four chains advance at ~the same **per-step** wall time as
``-np 1``, and total sampling throughput scales ~with GPU count.

Grids in :meth:`initialize` stay on **NumPy** so JAX does not touch the GPU
until :meth:`calculate`.

**Performance / lifecycle:** All heavy objects are created **once** in
:meth:`initialize`:

- ``Cosmology(emulator_set="lcdm:v1")`` — seed; emulators live in ``_emu`` and
  persist across :meth:`~hmfast.cosmology.Cosmology.update`.
- One :class:`~hmfast.halos.HaloModel` via :meth:`~hmfast.halos.HaloModel.update`
  (not a new ``HaloModel(...)`` each step).
- One :class:`~hmfast.halos.profiles.GNFWPressureProfile` seed and
  :class:`~hmfast.tracers.tSZTracer`.

Each :meth:`calculate` uses ``Cosmology.update`` / ``HaloModel.update`` and
``tSZTracer.update(profile=GNFWPressureProfile.update(B=...))``. Halo-model JIT in
``hmfast`` traces tracers and pressure profiles as PyTrees so ``B`` is a dynamic
parameter (no full XLA recompile every step from static specialization).

**Note:** ``HaloModel.update`` keeps the **TophatVar** instance from the fiducial
build (hmfast design). For a full rebuild per point, call
``HaloModel(cosmology=...)`` instead (slower).
"""

from __future__ import annotations

import math
import os
import time


def _mpi_world_size() -> int:
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE"):
        v = os.environ.get(key)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                continue
    return 1


def _mpi_rank() -> int | None:
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"):
        v = os.environ.get(key)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                continue
    return None


_w_mpi = _mpi_world_size()
_rk_mpi = _mpi_rank()
# Physical GPUs on this node for MPI round-robin (user has two cards → default 2).
_NUM_GPUS = max(1, int(os.environ.get("HMFAST_COBAYA_NUM_GPUS", "2")))

# GPU/CPU for JAX: must run before jax / hmfast import in a fresh process.
if os.environ.get("HMFAST_COBAYA_USE_GPU", "1").strip() not in (
    "1",
    "true",
    "True",
    "yes",
    "YES",
):
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    if _w_mpi > 1 and _rk_mpi is not None:
        # One device per process view: rank r uses physical GPU (r % num_gpus).
        _phys = _rk_mpi % _NUM_GPUS
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_phys)
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_gpu = os.environ.get("HMFAST_COBAYA_USE_GPU", "1").strip() in (
    "1",
    "true",
    "True",
    "yes",
    "YES",
)
# JAX GPU allocator limits (must run before ``import jax``).
# Default: 50% of each process's visible device; multiple ranks on one GPU split that 50%.
if _gpu:
    if _w_mpi > 1:
        _sharing = max(1, int(math.ceil(float(_w_mpi) / float(_NUM_GPUS))))
        if _sharing > 1:
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            _frac = max(0.15, 0.5 / float(_sharing))
            os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", f"{_frac:.4g}")
        else:
            os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import numpy as np
import jax
import jax.numpy as jnp
from cobaya.theory import Theory

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer


def _ell_d_l_from_cl(ell: jnp.ndarray, cl: jnp.ndarray) -> jnp.ndarray:
    """D_ell = ell(ell+1) C_ell / (2 pi) (same convention as CMB D_ell)."""
    return ell * (ell + 1.0) * cl / (2.0 * jnp.pi)


# Scatter-map synthetic binned files (e.g. ``Dl_binned_fullsky_signal_only_real0.txt``) store
# :math:`10^{12} \times D_\ell` to match the map / tszpower pipeline; multiply native hmfast
# :math:`D_\ell` by this before passing to the Gaussian likelihood.
_DL_ELL_DATA_SCALE: float = 1.0e12

# Reference cosmology for one-time HaloModel construction + JIT warmup (LCDM priors centre).
_FIDUCIAL_COSMO = dict(
    H0=67.4,
    omega_cdm=0.12,
    omega_b=0.022,
    ln1e10A_s=2.97,
    n_s=0.96,
    tau_reio=0.0544,
)
# Seed GNFW ``B`` for profile/tracer construction in :meth:`initialize` only.
_FIDUCIAL_B = 1.41


class HMFastTSZFullSky(Theory):
    """
    Full-sky tSZ auto spectrum: total (1h+2h) Limber C_l into D_ell at data ell centers,
    then scaled by :data:`_DL_ELL_DATA_SCALE` so outputs match ``10^{12} × D_ell`` in the
    tutorial data files.
    """

    output = ["Cl_sz"]

    params = {
        "H0": None,
        "omega_b": None,
        "omega_cdm": None,
        "ln10_10A_s": None,
        "n_s": None,
        "tau_reio": None,
        "B": None,
    }

    # Path to the same D_l file used by the likelihood (for ell centers only).
    data_vector_file: str = ""

    # Halo-model grids (adjust if needed for accuracy vs speed tradeoff)
    n_mass: int = 48
    n_z: int = 96
    log10_m_min: float = 11.0
    log10_m_max: float = 15.5
    z_min: float = 0.01
    z_max: float = 2.0

    def initialize(self):
        if not self.data_vector_file:
            raise ValueError("HMFastTSZFullSky: set data_vector_file in the YAML or info dict.")
        D = np.loadtxt(self.data_vector_file)
        self.ell_data = np.asarray(D[:, 0], dtype=np.float64)
        self.m = np.logspace(self.log10_m_min, self.log10_m_max, self.n_mass)
        self.z = np.linspace(self.z_min, self.z_max, self.n_z)
        self.l = np.asarray(self.ell_data, dtype=np.float64)
        self.log.info(
            "HMFastTSZFullSky: %d ell bins, ell in [%.1f, %.1f]",
            len(self.ell_data),
            float(self.ell_data[0]),
            float(self.ell_data[-1]),
        )
        self.log.info(
            "JAX devices (platform=%s): %s",
            jax.devices()[0].platform if jax.devices() else "?",
            [str(d) for d in jax.devices()],
        )
        _want_gpu = os.environ.get("HMFAST_COBAYA_USE_GPU", "1").strip() in (
            "1",
            "true",
            "True",
            "yes",
            "YES",
        )
        try:
            _backend = jax.default_backend()
        except Exception:
            _backend = "?"
        self.log.info("HMFastTSZFullSky: JAX default_backend=%s", _backend)
        if _want_gpu and _backend == "cpu":
            self.log.warning(
                "HMFastTSZFullSky: JAX fell back to CPU though GPU was requested. "
                "Unset JAX_PLATFORMS or install CUDA jaxlib; check Cobaya logs on each MPI rank."
            )
        _rk_here = _mpi_rank()
        if _want_gpu:
            self.log.info(
                "HMFastTSZFullSky: CUDA_VISIBLE_DEVICES=%s (MPI size=%d, rank=%s, HMFAST_COBAYA_NUM_GPUS=%d)",
                os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
                _w_mpi,
                _rk_here if _rk_here is not None else "-",
                _NUM_GPUS,
            )
        if _w_mpi > 1:
            _rk = _rk_here
            _sharing = max(1, int(math.ceil(float(_w_mpi) / float(_NUM_GPUS))))
            if _sharing > 1:
                _hint = (
                    "independent chains; ranks share GPU(s) — per-step time often > np 1 on one card."
                )
            else:
                _hint = "independent chains; one rank per GPU — per-chain step time ~exclusive GPU."
            self.log.info(
                "HMFastTSZFullSky: MPI world_size=%d rank=%s — %s",
                _w_mpi,
                _rk if _rk is not None else "?",
                _hint,
            )
        self._cosmo_seed = Cosmology(emulator_set="lcdm:v1")
        self._prof_seed = GNFWPressureProfile(B=_FIDUCIAL_B)
        self._tsz = tSZTracer(profile=self._prof_seed)
        cosmo_fid = self._cosmo_seed.update(**_FIDUCIAL_COSMO)
        self._hm_seed = HaloModel(cosmology=cosmo_fid)

        super().initialize()

        t_warm = time.perf_counter()
        lq = jnp.asarray(self.l)
        mq = jnp.asarray(self.m)
        zq = jnp.asarray(self.z)
        block = getattr(jax, "block_until_ready", lambda x: x)
        block(self._hm_seed.cl_1h(self._tsz, None, lq, mq, zq))
        block(self._hm_seed.cl_2h(self._tsz, None, lq, mq, zq))
        c_slightly = self._cosmo_seed.update(**{**_FIDUCIAL_COSMO, "H0": 67.41})
        hm_step = self._hm_seed.update(cosmology=c_slightly)
        block(hm_step.cl_1h(self._tsz, None, lq, mq, zq))
        block(hm_step.cl_2h(self._tsz, None, lq, mq, zq))
        self.log.info(
            "HMFastTSZFullSky: emulator + HaloModel init + JIT warmup in %.4f s",
            time.perf_counter() - t_warm,
        )

    def calculate(self, state, want_derived=True, **params_values):
        t0 = time.perf_counter()
        H0 = float(params_values["H0"])
        omega_b = float(params_values["omega_b"])
        omega_cdm = float(params_values["omega_cdm"])
        ln1e10A_s = float(params_values["ln10_10A_s"])
        n_s = float(params_values["n_s"])
        tau_reio = float(params_values["tau_reio"])
        B = float(params_values["B"])

        cosmo = self._cosmo_seed.update(
            H0=H0,
            omega_cdm=omega_cdm,
            omega_b=omega_b,
            ln1e10A_s=ln1e10A_s,
            n_s=n_s,
            tau_reio=tau_reio,
        )
        hm = self._hm_seed.update(cosmology=cosmo)
        tsz = self._tsz.update(profile=self._prof_seed.update(B=B))

        l = jnp.asarray(self.l)
        m = jnp.asarray(self.m)
        z = jnp.asarray(self.z)

        cl_1h = hm.cl_1h(tsz, None, l, m, z)
        cl_2h = hm.cl_2h(tsz, None, l, m, z)
        cl_tot = cl_1h + cl_2h
        Dl = _ell_d_l_from_cl(l, cl_tot) * _DL_ELL_DATA_SCALE
        block = getattr(jax, "block_until_ready", lambda x: x)
        block(Dl)

        state["Cl_sz"] = {
            "1h": np.asarray(Dl, dtype=np.float64),
            "2h": np.zeros(len(self.ell_data), dtype=np.float64),
        }
        self._current_state = state

        elapsed = time.perf_counter() - t0
        self.log.info("HMFastTSZFullSky: theory computed in %.4f s", elapsed)

    def get_Cl_sz(self):
        return self._current_state.get("Cl_sz", None)
