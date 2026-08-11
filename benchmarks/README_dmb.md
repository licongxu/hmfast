# DMB GPU path: correctness and performance

## Correctness (priority)

hmfast’s DMB profiles implement the same BCM / GODMAX `BCM_18_wP` physics:

| Ingredient | Implementation |
|---|---|
| Gas / CGA / CLM fractions | Same formulas and default parameters as GODMAX |
| Adiabatic contraction \(\zeta\) | Same root \((\zeta-1)=a_\zeta[(M_i/M_f)^{n_\zeta}-1]\) on a 32-point \(\zeta\) grid |
| HSE → \(P_e\) | \(P=\int_r^{6R_{200c}} G\rho_g M(<r')/r'^2\,dr'\), then \(P_e=P_{\rm th}/1.932\) in eV/cm³ |

**Algorithmic difference (intentional, correctness-preserving on the work grid):**
enclosed mass and HSE use **cumulative trapezoid sweeps** instead of GODMAX’s
per-radius re-integration. On identical nodes that is the same trapezoid rule;
residuals vs GODMAX are from quadrature grids / interpolation, not a different model.

**Gate:** `tests/test_dmb_profiles.py::test_dmb_pe_and_rho_vs_godmax`
(and `test_dmb_pe_vs_godmax_second_halo`)

- Compare \(P_e\) and \(\rho_{\rm dmb}\) to GODMAX
- Median relative error \(<2\%\) on \(0.05\text{–}3\,R_{200c}\)
- Max relative error \(<5\%\) on \(0.05\text{–}2\,R_{200c}\)
  (outer HSE is more sensitive to quadrature grids; median remains the
  primary science gate)

## GPU vectorization

1. Precompute \(M_{\rm nfw}(<r)\), \(M_{\rm gas}\), \(M_{\rm cga}\) once (`_cum_mass`).
2. Solve \(\zeta(r)\) on a dense `(N_ζ, N_r)` grid with interpolated cumulative masses (no nested ∫ per trial \(\zeta\)).
3. HSE via one reverse cumulative sweep (`_reverse_cumtrapz`).
4. Many halos: `vmap` over flattened `(M,z)` in `_eval_field`.

## Benchmark (10% device memory)

Scripted results: [`dmb_gpu_vs_cpu.json`](dmb_gpu_vs_cpu.json)

Reproduce (uses `XLA_PYTHON_CLIENT_MEM_FRACTION=0.1`):

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
cd /path/to/hmfast
PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python - <<'PY'
# see commit history / re-run the GPU vs CPU timing block in agent notes
PY
```

Typical warm-path \(C_\ell^{yy}\) (`nM=32`, `nz=20`, `nell=40`) after ζ–HSE vectorization:

- DMB ≈ **1.3×** GNFW on GPU (not ~20×)
- DMB GPU ≈ **18×** faster than DMB CPU

GNFW remains a closed-form `Pe(r)`; a small residual gap is expected.
