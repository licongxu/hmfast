---
description: Python style conventions for hmfast
---

# Python style

## Formatting

- **Black-compatible** formatting; line length **100** unless an existing module already uses a different limit.
- Use **double quotes** for strings unless the literal contains a `"` character.
- Imports grouped: stdlib, third-party, hmfast (relative). One blank line between groups.

## Typing and docstrings

- Add **type hints** for new public functions and methods. Use `jax.Array` (or `jnp.ndarray`) for JAX arrays in signatures.
- Docstrings: **NumPy style** with `Parameters`, `Returns`, `Notes`, `References`. Keep one-line summaries for trivial helpers.
- Reference physics in `Notes` with the equation label and a citation; keep derivations in the paper, not the docstring.

## Naming

- Functions and variables: `snake_case`. Classes: `CamelCase`. Module-level constants: `UPPER_SNAKE_CASE`.
- Use the literature symbol name for cosmological quantities (`sigma8`, `Omega_m`, `H0`). Do not invent new short forms.

## What not to do

- No bare `except:`. Catch the narrowest exception that is correct.
- No mutable default arguments (`def f(x=[])`).
- No `print` debugging in committed code; use the `logging` module.
- No drive-by reformatting of files you are not otherwise editing.
- No unrelated refactors bundled into a feature commit.
