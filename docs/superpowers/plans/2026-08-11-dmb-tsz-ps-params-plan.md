# Implementation Plan: DMB tSZ PS parameter-shape tutorial

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or implement directly in-session). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `tutorial/dmb_tsz_ps_params.ipynb` plotting \(C_\ell^{yy}\) shape vs DMB params with `hm_consistency=False`.

**Architecture:** Single Jupyter notebook following `tutorial/tsz_power_spectrum.ipynb` setup patterns and `DMBPressureProfile` / `tSZTracer` APIs. No library code changes.

**Tech Stack:** JAX, hmfast (`HaloModel`, `tSZTracer`, `DMBPressureProfile`), matplotlib, project venv `cmbagent_env`.

**Spec:** `docs/superpowers/specs/2026-08-11-dmb-tsz-ps-params-design.md`

---

## File map

| File | Role |
|------|------|
| `tutorial/dmb_tsz_ps_params.ipynb` | **Create** — only deliverable |
| `src/hmfast/**` | Unchanged (read-only reference) |

---

### Task 1: Create notebook skeleton + setup

**Files:**
- Create: `tutorial/dmb_tsz_ps_params.ipynb`

- [ ] **Step 1:** Markdown title + purpose + note that `hm_consistency=False` is required for tSZ/DMB shape scans
- [ ] **Step 2:** Code cell: `USE_GPU` toggle before `import jax`; imports; `jax_enable_x64`
- [ ] **Step 3:** Code cell: `Cosmology(emulator_set="lcdm:v1")`, `HaloModel(cosmology=cosmo, hm_consistency=False)`, grids `m`, `z`, `ell`

### Task 2: Fiducial \(C_\ell^{yy}\)

- [ ] **Step 4:** Build `tSZTracer(profile=DMBPressureProfile(num_points_trapz_int=32))`
- [ ] **Step 5:** Compute `cl_1h` + `cl_2h`; plot total + 1h/2h split
- [ ] **Step 6:** Smoke-run cell in venv; confirm finite positive spectrum

### Task 3: Parameter scans (absolute + ratio)

- [ ] **Step 7:** Helper `cl_yy(**dmb_kwargs)` returning numpy `C_ℓ`
- [ ] **Step 8:** 2×2 absolute loglog panels for `theta_ej_0`, `theta_co_0`, `log10_Mc0`, `mu_beta` with scan values from spec
- [ ] **Step 9:** 2×2 ratio panels \(C_ℓ / C_ℓ^{\rm fid}\)
- [ ] **Step 10:** Closing markdown takeaways
- [ ] **Step 11:** Execute notebook (or key cells) end-to-end; fix any runtime issues

### Task 4: Done criteria

- [ ] Notebook present under `tutorial/`
- [ ] Explicit `hm_consistency=False`
- [ ] Four params scanned; absolute + ratio plots
- [ ] No changes to `ref_packages/` or core library unless a bug blocks the tutorial
