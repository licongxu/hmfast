"""Hardware micro-benchmark: why the tSZ pipeline is slow on TPU.

Tests the exact operations the pipeline uses so we can see WHERE
the time goes and WHY GPU beats TPU on this workload.

Run on the TPU VM:
    PYTHONPATH=~/hmfast/src JAX_PLATFORMS=tpu HMFAST_JAX_ENABLE_X64=0 \
        python3 tpu/hw_microbench.py
"""
import time, json, sys
import jax
import jax.numpy as jnp

def bench(name, fn, n=10):
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
    med = sorted(ts)[len(ts)//2]
    print(f"  {name:<55s} {med:8.3f} ms")
    return med

def main():
    backend = jax.default_backend()
    print(f"backend = {backend}")
    print(f"devices = {jax.devices()}")
    print()

    # ---- 1. FFT: this is what mcfit Hankel does ----
    print("=== 1. FFT (what mcfit Hankel transform does) ===")

    # Single 1D FFT of length 1024 (one (m,z) pair)
    x1 = jax.random.normal(jax.random.PRNGKey(0), (1024,), dtype=jnp.float32)
    fft1 = jax.jit(lambda x: jnp.fft.rfft(x))
    bench("rfft  1×1024", lambda: fft1(x1), n=20)

    # Batched: 64 mass bins → 64 FFTs
    x64 = jax.random.normal(jax.random.PRNGKey(0), (64, 1024), dtype=jnp.float32)
    fft64 = jax.jit(lambda x: jnp.fft.rfft(x, axis=-1))
    bench("rfft  64×1024  (1 z-slice, all m)", lambda: fft64(x64), n=20)

    # Batched: 64×32 = 2048 FFTs (all m, all z)
    x2048 = jax.random.normal(jax.random.PRNGKey(0), (2048, 1024), dtype=jnp.float32)
    fft2048 = jax.jit(lambda x: jnp.fft.rfft(x, axis=-1))
    bench("rfft  2048×1024  (all m×z)", lambda: fft2048(x2048), n=20)

    # hfft (inverse half-complex FFT, also used by mcfit)
    c2048 = jax.random.normal(jax.random.PRNGKey(0), (2048, 513), dtype=jnp.complex64)
    hfft2048 = jax.jit(lambda x: jnp.fft.hfft(x, n=1024, axis=-1))
    bench("hfft  2048×1024  (all m×z)", lambda: hfft2048(c2048), n=20)

    # Full mcfit round-trip: rfft → multiply → hfft
    u = jax.random.normal(jax.random.PRNGKey(1), (513,), dtype=jnp.complex64)
    @jax.jit
    def mcfit_roundtrip(x, u):
        f = jnp.fft.rfft(x, axis=-1)
        g = f * u[None, :]
        return jnp.fft.hfft(g, n=1024, axis=-1) / 1024
    bench("mcfit roundtrip 2048×1024 (rfft→mul→hfft)", lambda: mcfit_roundtrip(x2048, u), n=20)

    print()
    print("=== 2. Matrix multiply (what TPU MXU is designed for) ===")

    # Small matmul (similar FLOP count to 2048 FFTs for comparison)
    A = jax.random.normal(jax.random.PRNGKey(0), (2048, 1024), dtype=jnp.float32)
    B = jax.random.normal(jax.random.PRNGKey(1), (1024, 1024), dtype=jnp.float32)
    mm1 = jax.jit(lambda a, b: a @ b)
    bench("matmul 2048×1024 @ 1024×1024", lambda: mm1(A, B), n=20)

    # Larger matmul
    A2 = jax.random.normal(jax.random.PRNGKey(0), (4096, 4096), dtype=jnp.float32)
    B2 = jax.random.normal(jax.random.PRNGKey(1), (4096, 4096), dtype=jnp.float32)
    mm2 = jax.jit(lambda a, b: a @ b)
    bench("matmul 4096×4096 @ 4096×4096", lambda: mm2(A2, B2), n=20)

    print()
    print("=== 3. Elementwise / VPU ops (memory-bandwidth bound) ===")

    v = jax.random.normal(jax.random.PRNGKey(0), (2048, 1024), dtype=jnp.float32)
    v2 = jax.random.normal(jax.random.PRNGKey(1), (2048, 1024), dtype=jnp.float32)
    v3 = jax.random.normal(jax.random.PRNGKey(2), (2048, 1024), dtype=jnp.float32)
    fexp = jax.jit(lambda x: jnp.exp(x))
    fsin = jax.jit(lambda x: jnp.sin(x))
    ffma = jax.jit(lambda a, b, c: a*b+c)
    flog = jax.jit(lambda x: jnp.log(jnp.abs(x) + 1e-30))
    bench("exp(x)  2048×1024", lambda: fexp(v), n=20)
    bench("sin(x)  2048×1024", lambda: fsin(v), n=20)
    bench("x*y+z   2048×1024  (fused)", lambda: ffma(v, v2, v3), n=20)
    bench("log(abs(x))  2048×1024", lambda: flog(v), n=20)

    print()
    print("=== 4. vmap dispatch overhead ===")

    # vmap 1 FFT over 32 z-slices (what the pipeline does)
    x_per_z = jax.random.normal(jax.random.PRNGKey(0), (32, 64, 1024), dtype=jnp.float32)
    fvmap = jax.jit(jax.vmap(lambda x: jnp.fft.rfft(x, axis=-1)))
    bench("vmap(rfft) over 32 z  (32×64×1024)", lambda: fvmap(x_per_z), n=20)

    # Compare: reshape to batch then single FFT
    x_flat = x_per_z.reshape(2048, 1024)
    fflat = jax.jit(lambda x: jnp.fft.rfft(x, axis=-1))
    bench("rfft on reshaped  (2048×1024)", lambda: fflat(x_flat), n=20)

    print()
    print("=== 5. jnp.interp (what fast-path does per z-slice) ===")

    xp = jnp.linspace(1, 10000, 1024)
    fp = jax.random.normal(jax.random.PRNGKey(0), (64, 1024), dtype=jnp.float32)
    x_new = jnp.linspace(10, 5000, 50)
    finterp1 = jax.jit(jax.vmap(lambda f: jnp.interp(x_new, xp, f)))
    bench("vmap(interp, 64 mass bins) 1 z", lambda: finterp1(fp), n=20)

    # 32 z-slices × 64 mass bins
    fp_all = jax.random.normal(jax.random.PRNGKey(0), (32, 64, 1024), dtype=jnp.float32)
    finterp2 = jax.jit(jax.vmap(jax.vmap(lambda f: jnp.interp(x_new, xp, f))))
    bench("vmap(vmap(interp)) 32z×64m", lambda: finterp2(fp_all), n=20)

    print()
    print("=== 6. Full mcfit Hankel as used by hmfast ===")
    try:
        from hmfast.halos.profiles import HankelTransform
        ht = HankelTransform(jnp.logspace(-6, 6, 1024), nu=0.5)
        f_in = jax.random.normal(jax.random.PRNGKey(0), (1024,), dtype=jnp.float32)
        bench("HankelTransform  1×1024  (single profile)", lambda: ht.transform(f_in), n=20)

        f_batch = jax.random.normal(jax.random.PRNGKey(0), (64, 32, 1024), dtype=jnp.float32)
        bench("HankelTransform  64×32×1024  (all m×z)", lambda: ht.transform(f_batch), n=10)
    except Exception as e:
        print(f"  (skipped: {e})")

    print()
    print("=== 7. Kernel launch / dispatch overhead ===")
    # Measure overhead of many tiny ops
    x_tiny = jnp.ones((1,), dtype=jnp.float32)
    @jax.jit
    def chain_100(x):
        for _ in range(100):
            x = x + 1.0
        return x
    bench("100 sequential adds (dispatch overhead)", lambda: chain_100(x_tiny), n=20)

    # No-op to measure pure dispatch
    @jax.jit
    def noop(x):
        return x + 0.0
    bench("no-op (pure dispatch latency)", lambda: noop(x_tiny), n=50)

    print()
    print("=== Summary ===")
    print("TPU MXU (systolic array) only accelerates dense matmul.")
    print("FFT, elementwise, interp all run on the VPU (vector unit),")
    print("which has ~10-100× lower throughput than the MXU.")
    print("GPU has thousands of general-purpose CUDA cores that handle")
    print("FFT natively. That's why GPU is faster for this workload.")


if __name__ == "__main__":
    main()
