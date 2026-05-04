---
description: Emulator sets, downloads, and bundled data paths for hmfast
---

# Emulators and data

- **Cosmology emulator sets** are defined in `cosmology.py` (`_COSMO_MODELS`). Only use documented `emulator_set` strings; extending them requires new emulator assets and loader wiring.
- **`download.py`** maps logical model names (`lcdm`, `ede-v2`, …) to remote **`.npz`** paths. Agents changing download logic must keep paths consistent with `EmulatorLoader` expectations in `emulator_load.py`.
- **Bundled files** under `src/hmfast/data/` ship with the package; use **`get_default_data_path`** / patterns in `download.py` rather than hardcoding user-specific directories.
- Importing **`hmfast`** triggers **`download_emulators`** in `__init__.py` — document side effects when adding headless CI or offline workflows (consider lazy download or env guards in future work).
