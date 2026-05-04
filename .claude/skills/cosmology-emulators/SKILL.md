---
name: cosmology-emulators
description: >-
  Guides changes to hmfast Cosmology, emulator loading, and cosmological
  parameter sets. Use when editing cosmology.py, emulator_load.py, download.py,
  or adding new emulator_set keys.
---

# Cosmology and emulators

## Model selection

- Valid **`emulator_set`** strings are listed in **`_COSMO_MODELS`** in `cosmology.py`. Adding a new key requires:
  - Consistent **suffix** and **subdir** for on-disk layout,
  - Matching **`.npz`** artifacts and loader logic in **`emulator_load.py`**,
  - Entries in **`download.py`** if assets are fetched remotely.

## Loader behavior

- **`EmulatorLoader` / `EmulatorLoaderPCA`** abstract how networks are restored — follow existing patterns when adding outputs (e.g. new tensors or keys).

## Side effects

- Package **`__init__.py`** runs **`download_emulators`** — when touching download or default models, consider impact on **first import** (network, disk).

## Testing

- Smoke-test **`Cosmology(emulator_set=...)`** construction and one or two **`pk`** / distance calls with finite outputs after changes.
