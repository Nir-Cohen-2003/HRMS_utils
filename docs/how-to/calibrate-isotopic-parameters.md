# Calibrate isotopic parameters

The complete, executable workflow and exhaustive CLI reference are maintained in the
canonical [mzML → MS-DIAL → HRMS_utils workflow](../tutorials/01-msdial-chromatogram-annotation.md#7-optional-isotope-parameter-calibration).
Use that document for prerequisites, input schemas, every option/default/range,
source-mode interactions, artifacts, model mapping, and troubleshooting.

## Quick start

Calibration is run through Pixi from the repository root. Create output parent
folders first; the command fails fast when required inputs or output parents are
missing.

CSV ground truth requires `Molecular Formula` and `Monoisotopic Mass` columns. The
mass must use the same ion/adduct mass domain as `Precursor_mz_MSDIAL`:

```bash
mkdir -p /absolute/path/calibration
pixi run -e experiments isotope_fit -- \
  --chromatogram-dir /absolute/path/msdial-output \
  --ground-truth-csv /absolute/path/ground-truth.csv \
  --output-file /absolute/path/calibration/results.txt \
  --elements C
```

Library mode takes processed library Parquet files. `--library-ion-mode P` is the
standalone compatibility default; select `N` for negative calibration:

```bash
pixi run -e experiments isotope_fit -- \
  --chromatogram-dir /absolute/path/msdial-output \
  --spectral-library /absolute/path/library.parquet \
  --output-file /absolute/path/calibration/results.txt \
  --library-ion-mode N \
  --elements C S Cl Br \
  --pairs-output-file /absolute/path/calibration/fitting_pairs.xlsx
```

Library and CSV sources may be combined. Fitting-pair XLSX export is library-only;
CSV combined with `--emit-pairs` or `--pairs-output-file` is rejected before fitting.
The default chromatogram glob is `**/*.mdpeak`; change it with
`--chromatogram-glob`. Existing report/plot outputs are overwritten.

Save the effective runtime model directly for later reuse:

```bash
pixi run -e experiments isotope_fit -- \
  --chromatogram-dir /absolute/path/msdial-output \
  --spectral-library /absolute/path/library.parquet \
  --output-file /absolute/path/calibration/results.txt \
  --library-ion-mode N \
  --save-model /absolute/path/calibration/model.json
```

The resulting pretty-printed `model.json` has no checksum and is intentionally
human-editable. Load that one file for a complete generic pipeline run with
`hrms-msdial-annotate --polarity negative --load-model /absolute/path/calibration/model.json`
(and the required metadata);
this skips spectral-library validation, matching, and calibration while retaining
MS-DIAL processing and annotation. The loader validates the schema version, seven
runtime fields, numeric values, finiteness, and model-safe constraints.

For direct Python annotation, use `load_calibrated_isotopic_model()` and pass its
result as `isotopic_model=` to `annotate_chromatogram_with_formulas()`.

Check the live CLI contract for the installed checkout with:

```bash
pixi run -e experiments isotope_fit -- --help
```


## Annotation-time error controls

The [installed API reference](../reference/isotopic-calibration-api.md#error-bound-controls)
explains `calibrated_error_bound_scale` (finite positive, default 1) and optional
keyword-only `minimum_element_count_error` / `maximum_element_count_error` maps
(default `None`). For example, both maps `{"C": 2}` give the rounded calibrated
carbon estimate ±2, subject to physical/global limits. These controls affect
annotation, not fitting or the saved baseline model. Missing/below-threshold isotope
peaks keep their existing one-sided behavior without count clamps.
