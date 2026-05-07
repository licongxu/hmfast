---
name: code-review
description: >-
  Reviews proposed changes to hmfast for correctness, JAX compatibility, API
  layering, test coverage, and hardware-cost surprises. Use when the user asks
  for a review, a sanity check on a diff, or before merging a feature branch.
---

# Code review

## What to check, in order

1. **Physics correctness.** Units, sign conventions, mass-definition consistency, integration limits.
2. **JAX safety.** No tracer leaks; no Python-level `if` / `for` on traced values; static args declared on `@jit`; emulator initialization not inside jitted scopes (see `.claude/rules/jax-and-numerics.md`).
3. **API layering.** Tracer / profile boundaries respected; no halo-model integrals duplicated outside `HaloModel` (see `.claude/rules/api-layering.md`).
4. **Tests.** New public behavior has at least a shape and finiteness test; numerical claims have a tolerance-based comparison.
5. **Hardware cost.** Estimate peak memory and runtime for the new code path. Flag anything that violates `.claude/rules/cluster-usage.md`.
6. **Style.** Type hints, docstrings, no print-debugging, no drive-by reformatting (see `.claude/rules/python-style.md`).

## Process

1. Read the diff: `git diff main...HEAD`.
2. Read each touched file in full, not just the hunks.
3. Run `pytest` for the touched modules; report failures verbatim.
4. Produce a written review with **Blocking**, **Suggested**, and **Nit** sections. Each item cites `file:line`. If a section is empty, write "none".

Do not modify code in a review pass. Suggest edits instead.
