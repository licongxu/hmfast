"""Compare y_l(M, z) between hmfast and tszpower at a few sample points."""
from __future__ import annotations
import sys
import tensorflow as tf
try: tf.config.set_visible_devices([], "GPU")
except RuntimeError: pass

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.concentration import ConstantConcentration
from hmfast.halos.profiles import ParametricGNFWPressureProfile, GNFWPressureProfile
from hmfast.tracers import tSZTracer

sys.path.insert(0, "/scratch/scratch-lxu/tszsbi/tszpower")
from tszpower.config import classy_sz
from tszpower.initialise import initialise as _tszp_initialise
import tszpower.profiles as tszp_profiles
import tszpower.tsz as tszp_tsz

FID = dict(H0=67.66, omega_cdm=0.1193, omega_b=0.02242,
           ln10_10A_s=2.9718, n_s=0.9665, tau_reio=0.0544)
B = 1.41
_h = FID["H0"] / 100.0

_FIXED_ASTRO = {
    "M_min": 1e14 * _h, "M_max": 1e16 * _h, "z_min": 5e-3, "z_max": 3.0,
    "P0GNFW": 8.130, "c500": 1.156,
    "gammaGNFW": 0.3292, "alphaGNFW": 1.0620, "betaGNFW": 5.4807,
    "jax": 1, "cosmo_model": 0,
}

# ---- hmfast y_l ----
cosmo = Cosmology(emulator_set="lcdm:v1").update(
    H0=FID["H0"], omega_cdm=FID["omega_cdm"], omega_b=FID["omega_b"],
    ln1e10A_s=FID["ln10_10A_s"], n_s=FID["n_s"], tau_reio=FID["tau_reio"],
)
hm = HaloModel(cosmology=cosmo, mass_definition=MassDefinition(500, "critical"),
               concentration=ConstantConcentration(c=4.0),
               convert_masses=True, hm_consistency=False)
prof = GNFWPressureProfile(B=B)  # use plain GNFW (no parametric, same as tszpower compute_y0)
tsz = tSZTracer(profile=prof)
m_test = jnp.array([1e14, 3e14, 1e15])  # physical M_sun
z_test = jnp.array([0.1, 0.5, 1.0])
ell_test = jnp.array([10.0, 100.0, 1000.0])

# u_k_hmfast at (k_phys = (ell+0.5)/chi(z), m, z) then convert to y_l via kernel and Limber
def hmfast_yl(ell_val, m_val, z_val):
    chi = float(cosmo.angular_diameter_distance(z_val) * (1 + z_val))
    k_phys = (ell_val + 0.5) / chi
    u_k = prof.u_k(hm, jnp.array([k_phys]), jnp.array([m_val]), jnp.array([z_val]))
    kernel = tsz.kernel(cosmo, jnp.array([z_val]))[0]
    return float(u_k[0, 0, 0] * kernel)

# ---- tszpower y_l ----
init = dict(_FIXED_ASTRO)
init.update({"H0": FID["H0"], "omega_b": FID["omega_b"], "omega_cdm": FID["omega_cdm"],
             "ln10^{10}A_s": FID["ln10_10A_s"], "n_s": FID["n_s"],
             "tau_reio": FID["tau_reio"], "B": B})
classy_sz.set(init)
_tszp_initialise()
params = dict(init)

print(f"{'ell':>6s} {'M_phys[M_sun]':>15s} {'z':>6s} "
      f"{'hmfast y_l':>14s} {'tszpower y_l':>14s} {'ratio':>10s}")
for ell_val in ell_test:
    for m_val in m_test:
        for z_val in z_test:
            yl_h = hmfast_yl(float(ell_val), float(m_val), float(z_val))
            # tszpower y_l: use y_ell_complete which returns (ell_arr, y_ell_arr) for given M, z
            m_tszp = float(m_val) * _h
            ell_arr, y_ell_arr = tszp_profiles.y_ell_complete(
                jnp.array([float(z_val)]),
                jnp.array([m_tszp]),
                params_values_dict=params,
            )
            ell_arr = np.asarray(ell_arr)[0]
            y_ell_arr = np.asarray(y_ell_arr)[0]
            yl_t = float(np.interp(float(ell_val), ell_arr, y_ell_arr))
            ratio = yl_h / yl_t if yl_t != 0 else float('nan')
            print(f"{float(ell_val):6.0f} {float(m_val):15.3e} {float(z_val):6.2f} "
                  f"{yl_h:14.5e} {yl_t:14.5e} {ratio:10.5f}")
