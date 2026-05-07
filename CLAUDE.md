# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

**hmfast** is a research Python package (alpha, v0.1.0) for **fast, differentiable halo-model predictions** used in cosmology. It combines **JAX** for autodiff and JIT, **neural emulators** for background and linear power (see `src/hmfast/cosmology.py`, `emulator_load.py`), **mcfit** (pinned to `0.0.22`) for correlation-function transforms, and a modular split between **halos** (`HaloModel`, mass function, bias, concentration, profiles under `src/hmfast/halos/`) and **observable tracers** (`src/hmfast/tracers/`). Importing the package **may download emulator weights** (`download_emulators` in `src/hmfast/__init__.py`); keep that in mind for CI, air-gapped machines, or cold starts.

For deeper conventions (JAX pytrees, JIT static arguments, emulator loading boundaries, tracer–profile pairing), read `.claude/rules/` and the skills under `.claude/skills/` before large edits.

## Working style

- **Understand before editing.** Trace the call path (e.g. `Cosmology` → `HaloModel` → tracer → profile) and note where JIT boundaries and emulator loads occur; do not move heavy I/O or emulator initialization inside jitted functions.
- **Prefer small, reviewable changes** over broad refactors unless explicitly requested.
- **Justify non-trivial edits** with a clear scientific or numerical reason (units, limits, stability, correctness of the halo-model ingredients).
- **Be explicit about uncertainty.** If something is unclear from the code or literature, say so and suggest a **minimal** check (unit test, finite-difference derivative, comparison to a reference limit, shape/finite checks).
- **Preserve existing behavior and code** unless fixing a documented bug or implementing an agreed change. Add or adjust code in a way that keeps current APIs and semantics intact; avoid drive-by rewrites, mass reformatting, or overwriting logic you are not asked to change.
- **Validate changes.** Run `pytest` from the repository root after substantive edits; for numerical work, add or extend tests that assert shapes, finiteness, and regressions where feasible.

## Environment

Main working directory:

- `/home/lxu/scratch/compute_packages/hmfast`

Important data/output locations:

- `/scratch/scratch-lxu/...` for working outputs
- `/rds/rds-lxu/...` for large or persistent outputs

Python environment:

Activate with:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
```

Development install and tests (from the repo root, after activating the environment):

```bash
pip install -e ".[dev]"
pytest
```
