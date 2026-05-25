"""Profile individual stages of the tSZ Cl pipeline to find bottlenecks.

Run with::

    PYTHONPATH=~/hmfast/src JAX_PLATFORMS=tpu HMFAST_JAX_ENABLE_X64=0 \
        python3 tpu/profile_stages.py

Reports median wall-time (5 runs, post-warmup) for each pipeline stage.
On lxu-persistent (TPU v6 lite) the dominant cost is the per-z vmap over
``profile.u_k(...)`` — see ``tpu/README.md`` for the analysis and the
acceleration path forward.
"""
import time
import jax
import jax.numpy as jnp

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer


def bench(name, fn, n=5):
    """Warm once, then time n runs."""
    r = fn()
    if hasattr(r, 'block_until_ready'):
        r.block_until_ready()
    elif isinstance(r, tuple):
        for x in r:
            if hasattr(x, 'block_until_ready'):
                x.block_until_ready()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        if hasattr(r, 'block_until_ready'):
            r.block_until_ready()
        elif isinstance(r, tuple):
            for x in r:
                if hasattr(x, 'block_until_ready'):
                    x.block_until_ready()
        ts.append(time.perf_counter() - t0)
    med = sorted(ts)[len(ts)//2]
    print(f"  {name:<45s} {med*1000:8.3f} ms  (all: {[f'{t*1000:.2f}' for t in ts]})")
    return r


def main():
    print(f"backend={jax.default_backend()}", flush=True)
    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)
    tsz = tSZTracer(profile=GNFWPressureProfile())

    m = jnp.logspace(10, 15, 64)
    z = jnp.linspace(0.01, 3.0, 32)
    ell = jnp.logspace(jnp.log10(10), jnp.log10(10000), 50)

    print("\n--- Stage timings (5 runs each, median) ---", flush=True)

    # 1. HMF
    bench("halo_mass_function(m, z)", lambda: hm.halo_mass_function.halo_mass_function(hm, m, z))

    # 2. Halo bias
    bench("halo_bias(m, z)", lambda: hm.halo_bias.halo_bias(hm, m, z))

    # 3. Concentration
    bench("c_delta(m, z)", lambda: hm.concentration.c_delta(hm, m, z))

    # 4. Background quantities
    bench("angular_diameter_distance(z)", lambda: cosmo.angular_diameter_distance(z))
    bench("hubble_parameter(z)", lambda: cosmo.hubble_parameter(z))
    bench("comoving_volume_element(z)", lambda: cosmo.comoving_volume_element(z))
    bench("tsz.kernel(cosmo, z)", lambda: tsz.kernel(cosmo, z))

    # 5. Profile u_k for ONE z slice (the inner body of the vmap)
    chi = cosmo.angular_diameter_distance(z) * (1 + z)
    k0 = (ell + 0.5) / chi[0]
    bench("u_k(k, m, z=[z0]) – 1 slice", lambda: tsz.profile.u_k(hm, k0, m, z[:1]))

    # 6. Profile u_k for ALL z (what the full pipeline actually does via vmap)
    def all_uk():
        def _one(zi):
            chi_i = cosmo.angular_diameter_distance(zi) * (1 + zi)
            ki = (ell + 0.5) / chi_i
            return tsz.profile.u_k(hm, ki, m, jnp.atleast_1d(zi))[:, :, 0]
        return jax.vmap(_one)(z)
    bench("u_k vmap over 32 z slices", all_uk)

    # 7. pk_1h for one z
    bench("pk_1h(k, m, z=[z0])", lambda: hm.pk_1h(tsz, tsz, k=k0, m=m, z=z[:1]))

    # 8. pk_2h for one z
    bench("pk_2h(k, m, z=[z0])", lambda: hm.pk_2h(tsz, tsz, k=k0, m=m, z=z[:1]))

    # 9. Linear P(k) interpolation
    cparams = cosmo._cosmo_params()
    h = cparams["h"]
    bench("P_lin vmap", lambda: jax.vmap(lambda zi: jnp.interp(h * k0, *cosmo.pk(zi, linear=True)))(z))

    # 10. Full cl_1h
    bench("cl_1h (full pipeline)", lambda: hm.cl_1h(tsz, tsz, l=ell, m=m, z=z))

    # 11. Full cl_2h
    bench("cl_2h (full pipeline)", lambda: hm.cl_2h(tsz, tsz, l=ell, m=m, z=z))

    # 12. Full cl_1h + cl_2h combined
    @jax.jit
    def compute_both(hm, tsz, ell, m, z):
        return hm.cl_1h(tsz, tsz, l=ell, m=m, z=z), hm.cl_2h(tsz, tsz, l=ell, m=m, z=z)
    bench("cl_1h + cl_2h (jit'd together)", lambda: compute_both(hm, tsz, ell, m, z))


if __name__ == "__main__":
    main()
