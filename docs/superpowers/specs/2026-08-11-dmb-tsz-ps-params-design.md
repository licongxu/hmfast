# Design: DMB tSZ PS parameter-shape tutorial

**Date:** 2026-08-11  
**Status:** Approved (chat)  
**Deliverable:** `tutorial/dmb_tsz_ps_params.ipynb`

## Goal

Tutorial notebook that plots the tSZ angular power spectrum \(C_\ell^{yy}\) with the DMB pressure profile and shows how the **shape** changes when varying primary DMB feedback parameters one at a time.

## Halo-model consistency

Use `HaloModel(..., hm_consistency=False)`.

Rationale: the consistency counterterm extrapolates the lowest-mass bin (\(n_{\min} u(M_{\min})^2\) in 1h; bias-weighted analog in 2h). That term is for matter–linear-theory consistency, not for pressure/tSZ. With it on, DMB parameter scans also move the floor term and can distort the shape response. GODMAX-style \(C_\ell^{yy}\) integrates an explicit mass grid without this counterterm.

## Scope

**In**

- Fiducial DMB \(C_\ell^{yy}\) (1h+2h), optional 1h vs 2h split
- One-at-a-time scans: \(\theta_{ej,0}\), \(\theta_{co,0}\), \(\log_{10} M_{c,0}\), \(\mu_\beta\)
- Absolute \(C_\ell\) curves and ratio \(C_\ell / C_\ell^{\rm fid}\)
- Markdown note on `hm_consistency=False` and parameter meaning (GODMAX / Schneider–Giri)

**Out**

- Other observables (\(\kappa\kappa\), gty, kSZ, \(P(k)\))
- GNFW / B12 comparison (already in `dmb_observables.ipynb`)
- Library API changes

## Parameter scan values

Defaults from `DMBPressureProfile` / GODMAX `BCM_18_wP`:

| Parameter | Default | Scan |
|-----------|---------|------|
| `theta_ej_0` | 4.0 | {2, 4, 6} |
| `theta_co_0` | 0.1 | {0.05, 0.1, 0.2} |
| `log10_Mc0` | 14.83 | {14.0, 14.83, 15.5} |
| `mu_beta` | 0.21 | {0.1, 0.21, 0.4} |

## Numerical setup

- `Cosmology(emulator_set="lcdm:v1")`
- Modest grids for interactive runtime: `nM≈24`, `nz≈16`, `nell≈40`
- `DMBPressureProfile(num_points_trapz_int=32)`
- GPU toggle cell (default on), same pattern as `tutorial/tsz_power_spectrum.ipynb`

## Layout

1. Setup / imports / device
2. Cosmology + `HaloModel(hm_consistency=False)` + grids
3. Fiducial \(C_\ell^{yy}\) (+ 1h/2h)
4. 2×2 absolute curves (one param per panel)
5. 2×2 ratio to fiducial
6. Short takeaways

## Success criteria

- Notebook runs end-to-end in the project venv
- `hm_consistency=False` is explicit and documented
- Four parameters produce visibly different shape responses in the ratio panels
