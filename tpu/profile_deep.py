"""Deep profiling: find the exact 50ms gap in cl_1h.

Sub-ops sum to ~3ms but cl_1h costs 52ms. This script measures
each piece of the cl_1h pipeline to find the hidden bottleneck.
"""
import time
import jax
import jax.numpy as jnp

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
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
    print(f"  {name:<60s} {med:8.3f} ms")
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

    # ---- Emulator / cosmological computation ----
    print("\n=== 1. Cosmological emulator calls ===")
    bench("pk(z=0.5, linear=True) single z",
          lambda: cosmo.pk(0.5, linear=True))
    vmap_pk = jax.jit(lambda zz: jax.vmap(lambda zi: cosmo.pk(zi, linear=True)[1])(zz))
    bench("vmap(pk) over 32 z",
          lambda: vmap_pk(z),
          n=5)

    # sigma_R via TophatVar (used by HMF and bias)
    cparams = cosmo._cosmo_params()
    h = cosmo.H0 / 100.0
    try:
        bench("sigma_R for all m at z=0.5 (TophatVar)",
              lambda: hm.halo_mass_function._sigma_R(hm, m, jnp.array([0.5])))
    except Exception as e:
        print(f"  sigma_R: skipped ({e})")

    print("\n=== 2. Counter-terms (HMF + bias + integration) ===")
    bench("_counter_terms(m, z) standalone",
          lambda: hm._counter_terms(m, z))

    print("\n=== 3. Pieces of cl_1h (outside vmap) ===")
    bench("tracer.kernel(cosmo, z)",
          lambda: tsz.kernel(cosmo, z))
    bench("comoving_volume_element(z)",
          lambda: cosmo.comoving_volume_element(z))
    bench("_u_ell_native(m, z)",
          lambda: profile._u_ell_native(hm, m, z))
    bench("HMF halo_mass_function(m, z)",
          lambda: hm.halo_mass_function.halo_mass_function(hm, m, z))
    bench("halo_bias(m, z)",
          lambda: hm.halo_bias.halo_bias(hm, m, z))
    bench("angular_diameter_distance(z)",
          lambda: cosmo.angular_diameter_distance(z))

    # ---- The per-z-slice work ----
    print("\n=== 4. Per-z-slice operations (vmapped 32×) ===")

    # Pre-compute what cl_1h_pressure_fast computes outside vmap
    kernel1 = tsz.kernel(cosmo, z)
    ell_nat, u_ell = profile._u_ell_native(hm, m, z)
    dndlnm = hm.halo_mass_function.halo_mass_function(hm, m, z)
    logm = jnp.log(m)
    dm = jnp.diff(logm)
    w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5
    total_weights = dndlnm * w[:, None]
    d_A = cosmo.angular_diameter_distance(z)
    chi = d_A * (1 + z)
    n_min, _, _ = hm._counter_terms(m, z)

    xp_sample = ell_nat[:, 0, 0]
    fp_sample = u_ell[:, 0, 0]

    # Just the interp for 64 mass bins
    interp_fn = jax.jit(jax.vmap(
        lambda en, un: jnp.interp(ell, en, un),
        in_axes=(1, 1), out_axes=1
    ))
    bench("jnp.interp 64 mass bins (1 z)",
          lambda: interp_fn(ell_nat[:, :, 0], u_ell[:, :, 0]),
          n=5)

    # Full per-z body (interp + mass sum + damping) for 32 z
    @jax.jit
    def per_z_body(ell_nat, u_ell, kernel1, total_weights, n_min, chi, ell, m):
        def _one_z(zi_idx):
            def _interp_m(ell_n, u_n):
                return jnp.interp(ell, ell_n, u_n)
            u1_at_ell = jax.vmap(_interp_m, in_axes=(1, 1), out_axes=1)(
                ell_nat[:, :, zi_idx], u_ell[:, :, zi_idx]
            ) * kernel1[zi_idx]
            uk_sq = u1_at_ell * u1_at_ell
            pk_y = jnp.sum(uk_sq * total_weights[None, :, zi_idx], axis=1)
            pk_y = pk_y + n_min[zi_idx] * uk_sq[:, 0]
            ki = (ell + 0.5) / chi[zi_idx]
            damping = 1.0 - jnp.exp(-(ki / 0.01)**2)
            return pk_y * damping
        return jax.vmap(_one_z)(jnp.arange(32))

    bench("per-z body (interp+mass_sum+damp) ×32 [precomp]",
          lambda: per_z_body(ell_nat, u_ell, kernel1, total_weights, n_min, chi, ell, m))

    # Full pipeline WITHOUT counter_terms in vmap
    @jax.jit
    def cl_1h_no_ct(hm, tsz, ell, m, z):
        kernel1 = tsz.kernel(hm.cosmology, z)
        comov_vol = hm.cosmology.comoving_volume_element(z)
        ell_nat, u_ell = tsz.profile._u_ell_native(hm, m, z)
        dndlnm = hm.halo_mass_function.halo_mass_function(hm, m, z)
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5
        total_weights = dndlnm * w[:, None]
        d_A = hm.cosmology.angular_diameter_distance(z)
        chi = d_A * (1 + z)
        n_min, _, _ = hm._counter_terms(m, z)

        def _one_z(zi_idx):
            def _interp_m(ell_n, u_n):
                return jnp.interp(ell, ell_n, u_n)
            u1_at_ell = jax.vmap(_interp_m, in_axes=(1, 1), out_axes=1)(
                ell_nat[:, :, zi_idx], u_ell[:, :, zi_idx]
            ) * kernel1[zi_idx]
            uk_sq = u1_at_ell * u1_at_ell
            pk_y = jnp.sum(uk_sq * total_weights[None, :, zi_idx], axis=1)
            pk_y = pk_y + n_min[zi_idx] * uk_sq[:, 0]
            ki = (ell + 0.5) / chi[zi_idx]
            damping = 1.0 - jnp.exp(-(ki / 0.01)**2)
            return pk_y * damping
        P_y_grid = jax.vmap(_one_z)(jnp.arange(z.shape[0]))
        integrand = P_y_grid * comov_vol[:, None]
        return jnp.trapezoid(integrand, x=z, axis=0)

    bench("cl_1h reimplemented (counter_terms OUTSIDE vmap)",
          lambda: cl_1h_no_ct(hm, tsz, ell, m, z))

    print("\n=== 5. Reference: actual cl_1h ===")
    bench("cl_1h (original)", lambda: hm.cl_1h(tsz, tsz, l=ell, m=m, z=z))

    @jax.jit
    def both(hm, tsz, ell, m, z):
        return hm.cl_1h(tsz, tsz, l=ell, m=m, z=z), hm.cl_2h(tsz, tsz, l=ell, m=m, z=z)
    bench("cl_1h+cl_2h jitted", lambda: both(hm, tsz, ell, m, z))


if __name__ == "__main__":
    main()
