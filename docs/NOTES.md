# hmfast Lab Notebook

See [archive/NOTES_2026-05-07_early.md](archive/NOTES_2026-05-07_early.md) for iterations 1-4 (project setup, benchmarks, test suite, bug fixes).

## Iteration 5 — 2026-05-07

**Intent:** Add GPU vs CPU speedup and gradient accuracy figures to the paper.

**Actions:**
1. Added `figure_gpu_cpu_speedup()`: grouped bar chart + speedup factor chart. Shows 7.8-13.9x speedup for halo-model ops.
2. Added `figure_gradient_accuracy()`: relative error of AD vs FD gradient as function of step size, V-curve with min ~1e-7.
3. Updated paper with Figure 3 (speedup) and Figure 4 (gradient accuracy).

## Iteration 6 — 2026-05-07

**Intent:** Add cross-tracer tests and multi-tracer comparison figure.

**Actions:**
1. Added `TestCrossTracer` (7 tests): tSZ-kSZ, tSZ-galaxy, CIB-galaxy pk_1h/pk_2h/cl_1h/cl_2h.
2. Added `TestMultiTracerGradients` (3 tests): CMB lensing, galaxy HOD, CIB gradient vs FD.
3. Added `figure_tracer_comparison()`: all 6 tracers' pk_1h + GPU vs CPU timing.
4. 66 tests pass.

## Iteration 7 — 2026-05-07

**Intent:** Add physics sanity check tests (analytic limits).

**Actions:**
1. Added `TestPhysicsLimits` (5 tests): mass function integral, mass-weighted bias, P_2h vs P_lin, P_1h positivity, C_l positivity.
2. Updated paper validation section with analytic limits.
3. 71 tests pass. 7 paper figures generated.

**Next steps:** Memory profiling, final paper polish, comparison vs external codes.

## Iteration 8 — 2026-05-07

**Intent:** Memory profiling, paper polish (citations, fix "future work" for already-implemented cross-tracer, add CIB profile section).

**Actions:**
1. Memory profiling: forward pass ~24 MB at largest grid, gradient ~1.2 GB (mostly JAX overhead). Runs on 8 GB consumer GPUs.
2. Added Memory footprint subsection to paper.
3. Fixed paper: cross-tracer correlations moved from future work to current implementation.
4. Added inline citations (Tinker08, Tinker10, Duffy08, Bhattacharya13, Battaglia12, Zheng07, Shang12, JAX).
5. Added CIB profile subsection with Shang2012 reference.
6. LaTeX balanced, no forbidden dashes.

**Status:** Paper has 7 figures, 1 table, complete method/validation/benchmark sections. 71 tests pass.
**Next steps:** Batch cosmology evaluation, comparison vs CLASSsz/CCL (if feasible), final README update.

## Iteration 9 — 2026-05-07

**Intent:** Batch cosmology evaluation via vmap.

**Key finding:** `jax.vmap` over cosmology parameter enables batch evaluation of 8 cosmologies in 3.8ms total (0.48ms per cosmology) vs ~3ms per cosmology sequential. Batch gradient: 3.3ms per cosmology. Pipeline is fully vmap-compatible.

**Actions:**
1. Benchmarked vmap batch evaluation: 8 cosmologies in 3.8ms forward, 26.7ms gradient.
2. Added batch evaluation subsection to paper benchmarks.
3. Updated paper abstract to mention vmap compatibility.
4. Saved results to `benchmarks/results/batch_vmap.json`.

**Project status:** Paper has 7 figures + 1 table. 71 tests pass. All P0-P1 items complete. P2 partially complete (vmap done, CLASSsz comparison not feasible without install). P3 paper figures done, needs final review.

## Iteration 10 — 2026-05-07

**Intent:** Address code review findings on paper.

**Fixes:**
1. Abstract: "sub-3ms" → "~3ms" (P_2h and C_l^2h are 3.0-3.1ms)
2. Table header: "GPU Speedup" → "Speedup" (cosmology ops show <1x)
3. Tracer comparison caption: removed "2-3x speedup" claim, clarified small grid vs full grid
4. Removed unused bib entries (Arnaud2010, Battaglia2016)
5. Fixed shortauthors to AAS style
6. Added explanation that GPU advantage emerges for multi-dimensional integrals

**Final project state:** 71 tests pass, 7 figures, 1 table, complete paper draft. All P0-P2 items substantially complete.
