---
description: Package structure, imports, and tracer/profile boundaries
---

# API layering

- **`Tracer`** (`tracers/base_tracer.py`) is abstract: each concrete tracer must declare **`_required_profile_type`** and implement **`kernel(cosmology, z)`**. Do not bypass profile type checks without a strong reason.
- **Halo profiles** live under `halos/profiles/` and inherit from **`HaloProfile`** hierarchy; match tracer category (pressure vs density vs matter vs HOD) to the profile used in sibling tracers.
- **`HaloModel`** (`halos/halo_model.py`) coordinates cosmology, mass function, bias, concentration, and profiles — avoid duplicating halo-model integrals in tracer modules; extend the halo model or tracer integration points instead.
- Exports: prefer **`from hmfast.halos import HaloModel`** until top-level `hmfast` re-exports are intentionally restored in `__init__.py`.
