"""Breakdown: where does the 52ms in cl_1h actually go?

The hw_microbench shows FFT/Hankel is only ~0.8ms, so the bottleneck
must be elsewhere. This script profiles each sub-operation.
"""
import time
import jax
import jax.numpy as jnp

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.halos.profiles.pressure import PressureProfile
from hmfast.halos.mass_definition import MassDefinition, convert_m_delta
from hmfast.tracers import tSZTracer


def bench(name, fn, n=5):
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
        ts.append((time.perf_counter() - t0) * 1000)
    med = sorted(ts)[len(ts) // 2]
    print(f"  {name:<55s} {med:8.3f} ms  (all: {[f'{t:.2f}' for t in ts]})")
    return r, med


def main():
    print(f"backend={jax.default_backend()}", flush=True)

    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)
    profile = GNFWPressureProfile()
    tsz = tSZTracer(profile=profile)

    m = jnp.logspace(10, 15, 64)
    z = jnp.linspace(0.01, 3.0, 32)
    ell = jnp.logspace(jnp.log10(10), jnp.log10(10000), 50)

    print("\n=== A. Inside u_r (real-space profile) ===")

    # 1. Mass conversion: convert_m_delta (200crit → 500crit)
    mass_def_old = hm.mass_definition
    mass_def_500c = MassDefinition(500, "critical")
    @jax.jit
    def do_convert(m, z):
        c_old = hm.concentration.c_delta(hm, m, z)
        return convert_m_delta(cosmo, m, z, mass_def_old, mass_def_500c, c_old=c_old)
    bench("convert_m_delta (200c→500c) all m,z", lambda: do_convert(m, z))

    # 2. Concentration
    bench("c_delta(m, z)", lambda: hm.concentration.c_delta(hm, m, z))

    # 3. r_delta
    bench("r_delta (mass_def)", lambda: hm.mass_definition.r_delta(cosmo, m, z))

    # 4. Hubble parameter
    bench("hubble_parameter(z)", lambda: cosmo.hubble_parameter(z))

    # 5. u_r for 1 z-slice
    r_delta = hm.mass_definition.r_delta(cosmo, m, z)
    r_1z = profile.x[:, None, None] * r_delta[None, :, :1] * (1.0 + z[None, None, :1])
    bench("u_r (GNFW) 1 z-slice (1024×64×1)", lambda: profile.u_r(hm, r_1z[:, :, 0], m, z[:1]))

    print("\n=== B. Inside _u_k_hankel (transform pipeline) ===")

    # 6. The vmap(vmap(single_m_z)) that evaluates u_r + weights for Hankel
    x = jnp.asarray(profile.x)
    r_delta_native = hm.mass_definition.r_delta(cosmo, m, z)
    r = x[:, None, None] * r_delta_native[None, :, :] * (1.0 + z[None, None, :])

    @jax.jit
    def eval_integrand(r, m, z):
        W_x = jnp.where((x >= x[0]) & (x <= x[-1]), 1.0, 0.0)
        def single_m_z(r_vals, m_val, z_val):
            pr = jnp.squeeze(profile.u_r(hm, r_vals, m_val, z_val))
            return pr * x**0.5 * W_x
        return jax.vmap(
            jax.vmap(single_m_z, in_axes=(1, None, 0), out_axes=0),
            in_axes=(1, 0, None), out_axes=0,
        )(r, m, z)
    bench("vmap(vmap(u_r)) integrand (64m × 32z × 1024x)", lambda: eval_integrand(r, m, z))

    # 7. Just the Hankel FFT on a pre-computed integrand
    integrand = eval_integrand(r, m, z)
    bench("Hankel FFT on precomputed integrand (64×32×1024)", lambda: profile._hankel.transform(integrand))

    print("\n=== C. Full _u_ell_native ===")
    bench("_u_ell_native (all m,z)", lambda: profile._u_ell_native(hm, m, z))

    print("\n=== D. Full u_k with interpolation ===")
    chi = cosmo.angular_diameter_distance(z) * (1 + z)
    k0 = (ell + 0.5) / chi[0]
    bench("u_k (1 z-slice)", lambda: profile.u_k(hm, k0, m, z[:1]))

    # Full pipeline
    print("\n=== E. Full cl pipelines ===")
    bench("cl_1h", lambda: hm.cl_1h(tsz, tsz, l=ell, m=m, z=z))
    bench("cl_2h", lambda: hm.cl_2h(tsz, tsz, l=ell, m=m, z=z))

    @jax.jit
    def both(hm, tsz, ell, m, z):
        return hm.cl_1h(tsz, tsz, l=ell, m=m, z=z), hm.cl_2h(tsz, tsz, l=ell, m=m, z=z)
    bench("cl_1h + cl_2h (jit'd)", lambda: both(hm, tsz, ell, m, z))

    print("\n=== Time budget ===")
    print("If vmap(vmap(u_r)) dominates, the bottleneck is the 2048")
    print("evaluations of the GNFW real-space profile (including")
    print("convert_m_delta mass conversion). NOT the FFT.")


if __name__ == "__main__":
    main()
