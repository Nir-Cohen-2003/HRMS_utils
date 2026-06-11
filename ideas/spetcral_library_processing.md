# Implementation Plan: Unified Spectral Library Processing API

## 1. Dependency Management
- **Action:** Add `parallel_rdkit` to `pixi.toml` under `[dependencies]`.
- **Note:** User will handle the specific installation source.

## 2. Refactor `nist_mspec.py` (Thin Parser)
- **Rename:** `read_MSPEC_file` logic moves to `parse_mspec`.
- **Logic:** Keep only regex extraction from raw text. Returns a `pl.LazyFrame`.
- **Output Columns:** `name`, `nist_id`, `db_id`, `instrument_type`, `instrument`, `mslevel`, `collision_energy_raw`, `ionization`, `ion_mode`, `precursor_type`, `precursor_mz`, `molecular_formula`, `inchikey`, `exact_mass`, `smiles`, `inchi`, `raw_spectrum_mz`, `raw_spectrum_intensity`.
- **Backward Compatibility:** Re-export `read_MSPEC_file` and `create_nist_dataframe` from the new `spectral_library` module.

## 3. Refactor `mgf.py` (Thin Parser)
- **Rename:** `read_mgf_to_dataframe` becomes `parse_mgf`.
- **Logic:** Returns a `pl.LazyFrame`.
- **Column Mapping (Unified Names):**
  - `NAME` -> `name`
  - `EXACTMASS` -> `exact_mass`
  - `FORMULA` -> `molecular_formula`
  - `INCHI` -> `inchi`
  - `INCHIAUX` -> `inchikey`
  - `SMILES` -> `smiles`
  - `MSLEVEL` -> `mslevel`
  - `ADDUCT` -> `precursor_type`
  - `PEPMASS` -> `precursor_mz`
  - `IONMODE` -> `ion_mode` (map "positive"/"negative" to "P"/"N")
  - `COLLISION_ENERGY` -> `collision_energy_raw`
  - `spectrum_mz` -> `raw_spectrum_mz`
  - `spectrum_intensity` -> `raw_spectrum_intensity`
- **MSn Support:** Retain `msn_` prefixed columns.

## 4. New Module: `src/hrms_utils/formats/spectral_library.py`

### 4.1. Core Dispatcher
- `_parse_file(path)`: Dispatches to `parse_mspec` or `parse_mgf` based on extension (`.msp`, `.mspec`, `.mgf`).

### 4.2. Processing Pipeline (`process_single_file`)
- **Metadata Annotation:** Flags for `is_orbitrap`, `is_ESI`, etc. (Moved from `nist_mspec.py`).
- **Collision Energy:** 
  - If `collision_energy_list` exists, calculate `mean` and set `multiple` flag.
  - Otherwise, parse from `collision_energy_raw`.
- **Precursor Formula:**
  - Derive `precursor_formula_array` from `molecular_formula` and `precursor_type`.
  - **Error Handling:** If formula is missing but InChIKey exists, raise `NotImplementedError` (future `parallel_rdkit` fix). If all molecular info is missing, filter the row.
- **Spectrum Cleaning:** Run `clean_and_normalize_spectrum` (Rust extension).
- **Scoring:** Add `explained_intensity` and `spectral_information_score`.

### 4.3. Multi-File Processing (`process_spectral_library`)
- **Deduplication:**
  - Group by `base_inchikey`.
  - Pairwise comparison using `spectral_similarity.explained_intensity` (Rust extension).
  - Parameters: `ms2_tolerance_ppm` (input), `threshold=0.99`.
  - If `A->B > 0.99` AND `B->A > 0.99`, keep only one.
- **PubChem Enrichment:** Optional join to retrieve missing SMILES/InChI from PubChem (logic from `process_spectral_library.py`).
- **MS-Ready Standardization:**
  - Collect unique SMILES per `base_inchikey`.
  - Call `msready_inchi_inchikey_parallel(smiles_list)`.
  - **Validation:** Fail if any result is null/empty.
  - Update all `smiles`, `inchi`, `inchikey`, and `base_inchikey` columns with the MS-ready versions.

## 5. Global Updates
- **Call Sites:** Update `scripts/process_spectral_library.py`, `tests/formats/test_nist_mspec.py`, and `experiments/` to use the new API.
- **Exports:** Update `src/hrms_utils/formats/__init__.py`.
- **Documentation:** Update all `.md` files in `docs/` to reflect new import paths and function names.
