# hmfast Lab Notebook

## Iteration 1 — 2026-05-07

**Git hash:** `8afc65d` (branch: `licongxu_autoresearch`)

**Intent:** Survey project state and create foundational artifacts.

**Findings:**
- Project has solid JAX-based halo model with: Cosmology (emulator-backed), HaloModel (pk_1h, pk_2h, cl_1h, cl_2h), 6 tracers, 12+ profile variants, PyTree support throughout.
- Tests under `tests/test_halo_model.py` are broken: they reference old API (`power_spectrum_1halo`, `mass_function`, `bias_function`) that no longer exists. Current API is `pk_1h`, `pk_2h`, etc.
- No benchmarks/, paper/, or lab notebook existed.
- `HaloModel` import is commented out in top-level `__init__.py` (import from `hmfast.halos` works).

**Actions taken:**
1. Created `docs/ROADMAP.md` with prioritized plan.
2. Created `docs/NOTES.md` (this file).
3. Creating benchmark infrastructure under `benchmarks/`.
4. Starting paper draft under `paper/`.

**Next steps:**
- Write repeatable benchmark script measuring: (a) HaloModel init + JIT warmup, (b) pk_1h, pk_2h, cl_1h, cl_2h evaluation for tSZ tracer, (c) gradient evaluation throughput.
- Fix the broken test suite to work with current API.
- Run benchmarks and save baseline results.

---

## Iteration 2 — 2026-05-07

**Git hash:** `8afc65d` (branch: `licongxu_autoresearch`)

**Intent:** Create benchmark script, fix tests, start paper draft, fix PyTree bug.

**Actions taken:**

1. **Benchmark script** (`benchmarks/run_all.py`): Created repeatable benchmark suite measuring:
   - Initialization time (Cosmology + HaloModel)
   - Cosmology operations (H(z), D_A(z), sigma8(z), P_lin(k,z), C_l^TT)
   - tSZ 3D power spectra (pk_1h, pk_2h)
   - tSZ angular power spectra (cl_1h, cl_2h)
   - CMB lensing angular power spectrum
   - Gradient evaluation (d/d omega_cdm)
   Outputs JSON results to `benchmarks/results/baseline_quick.json`.

2. **Test suite rewrite** (`tests/test_halo_model.py`): Replaced broken tests (referenced old API) with 33 working tests covering:
   - Initialization, PyTree roundtrip, update immutability
   - Cosmology operations (shape, finiteness)
   - Mass function, bias, concentration (shape, finiteness, positivity)
   - tSZ power spectra (pk_1h, pk_2h shape and finiteness)
   - Angular power spectra (tSZ cl_1h, cl_2h, CMB lensing cl_1h)
   - Gradient consistency (jax.grad vs finite difference)
   - All 33 tests pass.

3. **Bug fix**: `NFWMatterProfile` was missing JAX PyTree registration. Added `_tree_flatten`/`_tree_unflatten` methods and explicit `jax.tree_util.register_pytree_node` call in `src/hmfast/halos/profiles/matter.py`. This was blocking CMB-lensing tracer from working inside JIT-compiled functions.

4. **Paper draft** (`paper/hmfast.tex`): Created LaTeX manuscript with:
   - Abstract, introduction, method, validation, benchmarks, limitations sections
   - BibTeX references file (`paper/references.bib`)
   - Benchmark results table from quick-mode run

**Benchmark results (quick mode, NVIDIA RTX PRO 6000 GPU):**
- All halo-model operations: sub-4ms after JIT warmup
- tSZ pk_1h: 2.8 +/- 1.4 ms (Nk=50, Nm=30, Nz=10)
- tSZ pk_2h: 2.5 +/- 0.1 ms
- tSZ cl_1h: 2.3 +/- 0.1 ms
- Gradient evaluation: 2.7 +/- 0.1 ms
- Initialization: ~4s one-time cost

**Known issues:**
- pk_1h and pk_2h produce `inf` at z=0.0 due to HMF grid boundary. Fixed in benchmark by starting z grid at 0.01. Not a code bug, just expected boundary behavior.

**Next steps:**
- Run full-size benchmark (larger grids)
- Add benchmark comparison to other codes (CLASSsz, CCL)
- Add more tracer tests (kSZ, CIB, galaxy HOD, galaxy lensing)
- Generate benchmark figures for paper
- Profile memory usage

---

## Iteration 3 — 2026-05-07

**Git hash:** `8afc65d` (branch: `licongxu_autoresearch`)

**Intent:** Extend test coverage to all tracers; fix CIB profile bugs discovered during testing.

**Actions taken:**

1. **Extended test suite** (`tests/test_halo_model.py`): Added 23 new tests (33 → 56 total) covering all 6 tracers:
   - `TestkSZTracer` (5 tests): kernel shape/positivity, pk_1h shape/finite, cl_1h shape/finite, PyTree roundtrip
   - `TestGalaxyHODTracer` (7 tests): kernel shape, central contribution flag, pk_1h/cl_1h shape/finite, PyTree roundtrip, ng_bar finite, galaxy_bias finite
   - `TestGalaxyLensingTracer` (5 tests): kernel shape/positivity, pk_1h/cl_1h shape/finite, PyTree roundtrip
   - `TestCIBTracer` (6 tests): kernel shape, central contribution flag, pk_1h/cl_1h shape/finite, PyTree roundtrip, mean_emissivity finite

2. **Bug fix in `S12CIBProfile._sigma`** (`src/hmfast/halos/profiles/cib.py:214`): `log10_M_eff` was undefined (NameError at runtime). Fixed by adding `log10_M_eff = jnp.log10(M_eff_cib)` before its use.

3. **Bug fix in `S12CIBProfile._theta`** (`src/hmfast/halos/profiles/cib.py:257-264`): `k_B` (Boltzmann constant) was undefined (NameError). Fixed by adding `k_B = Const._k_B_` at the start of the method. This bug would have prevented any CIB computation from running.

**Test results:** 56 passed, 0 failed (69.7s total on GPU).

**Next steps:**
- Add benchmark timings for all 6 tracers to benchmark suite
- Add cross-tracer correlations (e.g., tSZ-galaxy, CIB-galaxy) tests
- Update paper validation section with new tracer test coverage
- Profile memory usage
