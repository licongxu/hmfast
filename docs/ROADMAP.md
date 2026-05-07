# hmfast Roadmap

## Goal
Make hmfast the fastest, most differentiable halo model package and produce a paper-quality draft demonstrating this.

## Priority Queue

### P0: Foundation (COMPLETE)
- [x] Lab notebook (docs/NOTES.md)
- [x] Roadmap (this file)
- [x] Repeatable benchmark script (benchmarks/run_all.py)
- [x] Fix broken tests (33→71 tests)
- [x] Baseline benchmark results (quick, full, full_cpu, batch_vmap)
- [x] Bug fix: NFWMatterProfile PyTree registration
- [x] Bug fix: CIB profile undefined k_B and log10_M_eff

### P1: Scientific Validation (MOSTLY COMPLETE)
- [x] All 6 tracer tests — 71 tests passing
- [x] Cross-tracer correlation tests (7 tests)
- [x] Multi-tracer gradient consistency (3 tests)
- [x] Analytic limit tests (5 tests: mass function, bias, P_2h, positivity)
- [x] Memory profiling (~24MB forward, ~1.2GB gradient on GPU)
- [ ] Comparison to external codes (CLASSsz, CCL) — requires separate install

### P2: Performance (MOSTLY COMPLETE)
- [x] Full-size benchmarks (GPU + CPU)
- [x] GPU vs CPU speedup (7.8-13.9x for halo-model ops)
- [x] Batch vmap evaluation (0.48ms/cosmo for 8-cosmo batch, scaling figure)
- [x] Runtime scaling figure

### P3: Paper Draft (NEAR COMPLETE)
- [x] Abstract + introduction + method + validation + benchmarks + conclusion
- [x] 8 figures: timing, GPU vs CPU, gradient accuracy, tSZ P(k) + C_l, scaling, batch vmap, tracer comparison
- [x] 1 table: GPU vs CPU benchmark results
- [x] Proper inline citations (JAX, Tinker08/10, Duffy08, Bhattacharya13, etc.)
- [ ] Fill acknowledgement placeholders
- [ ] Final proofread

### P4: Polish
- [ ] README update with badges and benchmark summary
- [ ] API documentation gaps filled
