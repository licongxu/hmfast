"""Smoke test for the hmfast tSZ angular power-spectrum pipeline on TPU/CPU.

Writes the spectrum plot next to this script (i.e. inside ``tpu/``) named
``tsz_spectrum_<backend>.png``. The wrapper ``run_benchmark.sh`` and the
upstream ``sync_and_run.sh`` rely on that layout.
"""
import os
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Let hmfast pick the right dtype for the active platform. On TPU it disables
# x64 (mcfit's complex128 FFTs are unsupported there); on CPU/GPU it enables
# x64. See docs/tpu.md. Override with HMFAST_JAX_ENABLE_X64 / JAX_ENABLE_X64.
from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer


def main():
    print("==================================================")
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"JAX Devices: {jax.devices()}")
    print("==================================================")

    print("Initializing Cosmology and HaloModel...")
    t0 = time.time()
    cosmo = Cosmology(emulator_set="lcdm:v1")
    hm = HaloModel(cosmology=cosmo)
    print(f"Done. (Took {time.time()-t0:.2f}s)")

    print("Initializing tSZ Tracer (GNFW Profile)...")
    tsz = tSZTracer(profile=GNFWPressureProfile())

    m = jnp.logspace(10, 15, 64)
    z = jnp.linspace(0.01, 3.0, 32)
    ell = jnp.logspace(jnp.log10(10), jnp.log10(10000), 50)

    print("\n--- Compiling and executing tSZ angular power spectrum (Cl) ---")

    @jax.jit
    def compute_tsz_cl(hm, tsz, ell, m, z):
        cl1h = hm.cl_1h(tsz, tsz, l=ell, m=m, z=z)
        cl2h = hm.cl_2h(tsz, tsz, l=ell, m=m, z=z)
        return cl1h, cl2h

    t1 = time.time()
    print("First run (Compilation + Execution)...")
    cl1h, cl2h = compute_tsz_cl(hm, tsz, ell, m, z)
    cl1h.block_until_ready()
    cl2h.block_until_ready()
    t2 = time.time()
    print(f"First run completed in {t2 - t1:.4f} seconds.")

    print("Second run (Execution only)...")
    t3 = time.time()
    cl1h_fast, cl2h_fast = compute_tsz_cl(hm, tsz, ell, m, z)
    cl1h_fast.block_until_ready()
    cl2h_fast.block_until_ready()
    t4 = time.time()
    print(f"Second run completed in {t4 - t3:.4f} seconds.")

    cl_tot = cl1h_fast + cl2h_fast
    dl_tot = cl_tot * ell * (ell + 1) / (2 * jnp.pi)

    print("\n--- Sanity Checks ---")
    print(f"Cl shape: {cl1h_fast.shape}")
    print(f"Is Cl 1-halo all positive? {jnp.all(cl1h_fast > 0)}")
    print(f"Is Cl 2-halo all positive? {jnp.all(cl2h_fast > 0)}")
    print(f"Mean Cl: {jnp.mean(cl_tot):.4e}")

    print("\nGenerating tSZ Power Spectrum plot...")
    plt.figure(figsize=(8, 6))
    plt.loglog(ell, cl1h_fast * ell * (ell + 1) / (2 * jnp.pi), '--', label='1-halo', color='blue')
    plt.loglog(ell, cl2h_fast * ell * (ell + 1) / (2 * jnp.pi), ':', label='2-halo', color='green')
    plt.loglog(ell, dl_tot, '-', label='Total', color='red', linewidth=2)
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell^{yy} / 2\pi$')
    plt.title(f'tSZ Power Spectrum (Backend: {jax.default_backend()})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"tsz_spectrum_{jax.default_backend()}.png",
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
