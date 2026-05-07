---
description: pytest conventions and numerical-test patterns for hmfast
---

# Testing

## Layout

- Tests live under `tests/` and mirror the source path. `src/hmfast/halos/halo_model.py` corresponds to `tests/test_halo_model.py`.
- One assertion concept per test. Multiple `assert` lines are fine when they describe the same property (e.g. shape, finiteness, range).

## Numerical assertions

- Always assert **shape** and **finiteness** (`jnp.all(jnp.isfinite(...))`) before checking values.
- Compare floats with `jnp.allclose` and an explicit tolerance. Never use `==` on floats.
- For derivatives, test consistency between `jax.grad` and a finite difference with a small `eps`.
- For physics sanity, include **limit checks** (e.g. `b(M) -> 1` at the pivot mass, `xi(r -> large) -> 0`).

## Speed budget

- Each unit test should finish in **under 5 seconds** on a single GPU.
- Heavy regression tests go in `tests/slow/` and are skipped by default with `@pytest.mark.slow`.
- Do not introduce a test that requires downloading new emulator weights at test time.

## Running

```bash
pytest                                    # fast tests
pytest -m slow                            # opt in to the slow suite
pytest tests/test_halo_model.py::test_pk_shape -v
```

## Fixtures

- Reuse session-scoped fixtures for `Cosmology` and `HaloModel` to avoid repeated emulator loads.
- Make seeds explicit with `jax.random.PRNGKey(<int>)` so failures are reproducible.
