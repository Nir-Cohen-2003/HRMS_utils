# Positive-mode mzML → MS-DIAL → calibrated formula annotation handoff

This file records the historical positive-mode analysis run and compatibility usage.
Current generic usage is documented in [the installed API reference](docs/reference/isotopic-calibration-api.md).

## Polarity/count-control implementation update

- `hrms_utils.msdial_annotation` is canonical. Generic config/shortcuts and
  `hrms-msdial-annotate --polarity {positive,negative}` require explicit polarity.
- Positive aliases/CLI/tasks remain available with their original defaults.
- Negative methods are exact positive copies except Negative / `[M-H]-`; calibration
  filters `N` for negative and `P` for positive. Standalone calibration defaults to `P`.
- New outputs use exact source `.mdpeak` stems, not generated `table_001_` prefixes.
  Duplicate stems fail before copies/annotation; existing analysis output is untouched.
- `calibrated_error_bound_scale=1.0` scales only annotation-time log-width amplitudes.
  Saved seven-field models stay baseline; manifests record baseline/effective models.
- Optional keyword-only `minimum_element_count_error` / `maximum_element_count_error`
  maps (`C/S/Cl/Br`, integer 0..Int32 max, default None) clamp directional integer
  margins around `floor(n_hat+0.5)`, after width scaling, minimum first then maximum,
  with physical/global limits. Both C maps set to 2 give rounded expected count ±2.
- Count clamps are NOT_APPLIED without a detected calibrated center (missing
  precursor or absent/below-threshold isotope). Native default-None inference and the
  24-integer schema remain unchanged. Complete wrappers take maps in parameter bags.
- CLI maps are JSON. Loaded models skip library calibration, **not MS-DIAL**.
  No polarity provenance or checksum is added to editable model files.
- No new real-data MS-DIAL run was performed for this update. Historical counts and
  prefixed filenames below describe the prior positive run, not the new naming rule.


## Implementation status

The orchestration script is:

```text
experiments/run_positive_msdial_annotation.py
```

The corresponding Pixi task is:

```text
run_positive_msdial_annotation
```

The script is fixture-tested and a real end-to-end run has been completed previously. The local MS-DIAL executable and its runtime companions were found, and executable permission was verified without changing it. Existing generated analysis outputs are not modified by the model-persistence changes.

The canonical detailed documentation is:

```text
docs/tutorials/01-msdial-chromatogram-annotation.md
```

## Default configuration

| Setting | Default |
|---|---|
| mzML glob | `analysis/*.mzML` |
| MS-DIAL executable | `MSDIAL/MSDIALCUI` |
| MS-DIAL method | `experiments/msdial_console_params_positive.txt` |
| Spectral library | `/data/nirc/spectral_libs/iibr_260615/iibr_260708.parquet` |
| Output directory | `analysis/msdial-positive-annotation` |
| Calibration elements | `C` |
| Annotation precursor tolerance | `3.0 ppm` |
| Annotation raw-fragment tolerance | `5.0 ppm` |
| Annotation normalized-fragment tolerance | `5.0 ppm` |
| Overwrite existing output | Disabled |
| Load model | Disabled; `--load-model PATH` skips library calibration |
| Save model | Disabled; `--save-model PATH` saves a fit-now model |

All mzML files are treated as positive-ion-mode inputs. The script does not infer polarity and does not expose a negative-mode option.

## Positive MS-DIAL method

The tracked positive method is:

```text
experiments/msdial_console_params_positive.txt
```

It was copied from `experiments/msdial_console_params.txt`, leaving the original negative method unchanged. Its relevant positive-mode settings are:

```text
Ion mode: Positive
adduct list: [M+H]+
```

Only `[M+H]+` is enabled.

The method format is MS-DIAL-version-sensitive. The script verifies that the method contains exactly one positive ion-mode setting, but this does not prove that every method key is valid for the installed MS-DIAL release.

## Required metadata

The command requires an inline JSON object supplied with `--metadata-values`. It must contain exactly:

- `file_type`: nonempty string
- `class_id`: nonempty string
- `acquisition_type`: nonempty string
- `factor`: nonempty string
- `batch_order`: positive integer

Example:

```json
{
  "file_type": "Sample",
  "class_id": "1",
  "acquisition_type": "Sample",
  "factor": "NA",
  "batch_order": 1
}
```

These example categorical values must be checked against the installed MS-DIAL version and the intended experiment. The pipeline deliberately does not invent or silently substitute metadata values.

## Running the pipeline

From the repository root:

```bash
pixi run -e experiments run_positive_msdial_annotation -- \
  --metadata-values '{
    "file_type": "Sample",
    "class_id": "1",
    "acquisition_type": "Sample",
    "factor": "NA",
    "batch_order": 1
  }'
```

A compact one-line form is:

```bash
pixi run -e experiments run_positive_msdial_annotation -- \
  --metadata-values '{"file_type":"Sample","class_id":"1","acquisition_type":"Sample","factor":"NA","batch_order":1}'
```

### Available pipeline options

```text
--mzml-glob GLOB
--msdial-path PATH
--spectral-library PATH
--output-directory PATH
--msdial-method PATH
--metadata-values JSON
--calibration-elements {C,S,Cl,Br} [{C,S,Cl,Br} ...]
--overwrite
```

Example with overrides:

```bash
pixi run -e experiments run_positive_msdial_annotation -- \
  --mzml-glob 'analysis/**/*.mzML' \
  --msdial-path MSDIAL/MSDIALCUI \
  --msdial-method experiments/msdial_console_params_positive.txt \
  --spectral-library /data/nirc/spectral_libs/iibr_260615/iibr_260708.parquet \
  --output-directory analysis/msdial-positive-annotation \
  --metadata-values '{"file_type":"Sample","class_id":"1","acquisition_type":"Sample","factor":"NA","batch_order":1}' \
  --calibration-elements C S Cl Br
```

Pass `--overwrite` only when an existing output directory should be replaced. Publication is staged and performed atomically so an earlier successful output is preserved if processing fails before publication.

## Processing stages

The script performs these steps:

1. Expand, sort, and deduplicate the mzML glob.
2. Fail if no mzML files match or if duplicate sample stems are present.
3. Validate the local MS-DIAL executable and its `.dll`, `.runtimeconfig.json`, and `.deps.json` companions.
4. Validate the positive method and metadata JSON.
5. Validate the spectral-library schema and confirm that positive-mode (`ion_mode == "P"`) records exist without eagerly loading the complete library.
6. Generate an absolute-path MS-DIAL metadata CSV.
7. Run MS-DIAL using an argument list rather than shell interpolation:

   ```text
   MSDIALCUI lcms -i metadata.csv -o msdial-output -m positive-method.txt -p
   ```

8. Capture MS-DIAL stdout, stderr, and the exact command.
9. Discover `.mdpeak` files and retain only files whose tab-delimited headers satisfy `get_chromatogram()`.
10. Fail if MS-DIAL produced no parser-compatible feature tables.
11. Search the positive-mode portion of the configured spectral library to construct assumed formula ground truth.
12. Fit the calibrated isotopic-intensity model.
13. Convert the fitted parameters by name into `CalibratedIsotopicModel`.
14. Select monoisotopic features with nonempty MS1 isotope and MS/MS spectra.
15. Deduce isotope bounds, perform mass decomposition, and annotate fragment formulas using explicit 3/5/5 ppm tolerances.
16. Publish the results, manifest, and generated schema description.

## Save and reuse a model

Calibration can save the seven effective runtime parameters in a deliberately
human-editable JSON file:

```bash
pixi run -e experiments isotope_fit -- \\
  --chromatogram-dir msdial-output \\
  --spectral-library /data/nirc/spectral_libs/iibr_260615/iibr_260708.parquet \\
  --output-file calibration/results.txt \\
  --save-model calibration/model.json
```

The file has `schema_version: 1` and a `model` object containing exactly `alpha`,
`beta`, `offset`, `hetero_lower_a`, `hetero_lower_b`, `hetero_upper_a`, and
`hetero_upper_b`. It has no checksum. Users may edit valid finite values directly.
The loader validates the version, fields, numeric types, finiteness, and model-safe
mathematical constraints.

To run the complete positive workflow with a previously saved model:

```bash
pixi run -e experiments run_positive_msdial_annotation -- \\
  --load-model calibration/model.json \\
  --metadata-values '{"file_type":"Sample","class_id":"1","acquisition_type":"Sample","factor":"NA","batch_order":1}'
```

`--load-model` still runs MS-DIAL, table parsing, mass decomposition, and annotation,
but skips spectral-library validation, library matching, and isotope calibration.
The library option remains available for fit-now runs. Fit-now pipeline runs write
`calibration/model.json` inside the output by default; `--save-model` can additionally
write a copy elsewhere. `--load-model` and `--save-model` cannot be combined.

Public Python helpers are:

```python
from hrms_utils import (
    load_calibrated_isotopic_model,
    save_calibrated_isotopic_model,
)
from hrms_utils.hrms_core import CalibratedIsotopicModel

save_calibrated_isotopic_model(model, "model.json")
model = load_calibrated_isotopic_model("model.json")
```

## Library calibration behavior

The default library is mixed-polarity, but calibration explicitly keeps only:

```text
ion_mode == "P"
```

Required library columns include:

```text
ion_mode
precursor_mz
cleaned_normalized_mz
cleaned_normalized_intensity
precursor_formula_array
name
inchikey
smiles
```

The library search uses these fixed defaults:

| Calibration setting | Value |
|---|---:|
| Precursor tolerance | `5 ppm` |
| MS2 tolerance | `10 ppm` |
| Dot-product threshold | `> 0.9` |
| Spectral-information-score threshold | `> 1.0` |

The calibration procedure:

1. Joins candidate library records by nominal mass.
2. Applies the precursor-mass tolerance.
3. Cleans the query spectrum and calculates its spectral-information score.
4. Rejects query spectra below the information-score threshold.
5. Calculates dot-product similarity against library spectra.
6. Rejects matches below the dot-product threshold.
7. Keeps a feature only when all passing library matches agree on one molecular formula.
8. If several spectra match that formula, keeps the highest-scoring match.
9. Rejects library records without an InChIKey.
10. Uses the accepted formula's C/S/Cl/Br counts as assumed calibration labels.

Calibration defaults to carbon (`C`). Use `--calibration-elements C S Cl Br` to include all supported isotope targets. Selected elements contribute to one shared fitted model; this is not an independent model or independent validation for every element.

## Calibration model mapping

The fitter returns these parameters:

```text
alpha
beta
offset
hetero_lower_a
hetero_lower_b
hetero_upper_a
hetero_upper_b
```

They are copied by name into `CalibratedIsotopicModel` and passed directly to formula annotation. The script does not parse the human-readable calibration report back into a runtime model.

## Annotation tolerances and semantics

The annotation call uses:

```text
precursor_mass_accuracy_ppm = 3.0
fragment_mass_accuracy_ppm = 5.0
normalized_fragment_mass_accuracy_ppm = 5.0
```

The operation:

1. Infers per-element minimum and maximum counts from the isotope pattern.
2. Performs bounded precursor-mass decomposition.
3. Explodes candidate formulas so each output row represents one precursor-formula candidate.
4. Cleans MS/MS spectra and assigns candidate formulas to fragments.
5. Calculates precursor and fragment mass errors and explained intensity.

## Output layout

The default publication directory is:

```text
analysis/msdial-positive-annotation/
```

Expected contents:

```text
analysis/msdial-positive-annotation/
├── msdial-metadata.csv
├── msdial-output/
│   ├── msdial.stdout.txt
│   ├── msdial.stderr.txt
│   ├── msdial.command.json
│   └── ... MS-DIAL output files ...
├── calibration-input/
│   └── ... parser-compatible feature tables ...
├── calibration/
│   ├── results.txt
│   ├── model.json
│   ├── results_calibration_bounds.png
│   └── ... optional calibration plots ...
├── annotations/
│   └── table_*.parquet
├── summaries/
│   └── table_*.csv
├── manifest.json
└── SCHEMA.md
```

`manifest.json` records submitted mzML paths, discovered and compatible feature tables, commands, configuration, thresholds, fitted-model values, and output provenance. It distinguishes submitted input files from discovered outputs and does not claim a verified one-to-one mapping when MS-DIAL does not establish one.

`SCHEMA.md` is generated from the actual annotated dataframe and is intended to accompany the outputs into downstream analysis.

## Short downstream schema description

### Nested Parquet files

Each `annotations/*.parquet` file preserves the original parsed feature information and nested spectra. There is one row per candidate precursor formula, so one input feature may produce multiple output rows.

Important input-derived columns include:

| Column | Meaning |
|---|---|
| `Peak ID` | MS-DIAL feature identifier |
| `RT (min)` | Retention time |
| `Precursor_mz_MSDIAL` | Measured precursor m/z used for decomposition |
| `Height` | Feature height |
| `Precursor_type_MSDIAL` | MS-DIAL adduct/precursor label |
| `Isotope` | MS-DIAL isotope index |
| `ms1_isotopes_m/z` | Observed MS1 isotope masses |
| `ms1_isotopes_intensity` | Observed absolute isotope intensities |
| `msms_m/z` | Observed fragment masses |
| `msms_intensity` | Observed fragment intensities |

Important annotation columns include:

| Column | Typical Polars type | Meaning |
|---|---|---|
| `min_bounds` | `Array(Int32, 12)` | Isotope-derived lower element-count bounds |
| `max_bounds` | `Array(Int32, 12)` | Isotope-derived upper element-count bounds |
| `precursor_formula` | `Array(Int32, 12)` | Candidate precursor ion-formula counts |
| `precursor_formula_str` | `String` | Human-readable candidate formula |
| `precursor_errors_ppm` | `Float64` | Precursor mass error |
| `cleaned_msms_mz` | `List(Float64)` | Retained/normalized fragment masses |
| `cleaned_msms_intensity` | `List(Float64)` | Corresponding fragment intensities |
| `cleaned_spectrum_formulas` | `List(Array(Int32, 12))` | Candidate formulas assigned to fragments |
| `cleaned_spectrum_formulas_str` | `List(String)` | Human-readable fragment formulas |
| `cleaned_fragment_errors_ppm` | `List(Float64)` | Fragment mass errors |
| `explained_intensity` | `Float64` | Fraction of MS2 intensity explained by assigned fragments |

The 12-element formula-array order is:

```text
H, C, N, O, F, Na, P, S, Cl, K, Br, I
```

Parquet is the canonical output because it preserves nested lists and fixed-size arrays.

### Scalar CSV summaries

Each `summaries/*.csv` contains scalar columns suitable for ordinary downstream tools:

```text
Peak ID
RT (min)
Precursor_mz_MSDIAL
Height
Precursor_type_MSDIAL
Isotope
precursor_formula_str
precursor_errors_ppm
explained_intensity
```

The summaries deliberately omit nested spectra, bounds, and formula arrays.

## Scientific limitations

1. **No adduct neutralization:** the measured precursor m/z is decomposed directly. Even though the MS-DIAL method allows only `[M+H]+`, the code does not subtract a proton or otherwise convert the ion into a neutral mass.
2. **Ion-formula candidates:** reported formulas are ion-formula hypotheses, not guaranteed neutral molecular formulas.
3. **Not compound identifications:** a formula candidate does not identify a unique structure or compound.
4. **Library matches are assumed labels:** the similarity and formula-consistency filters do not establish an identification false-discovery rate.
5. **In-sample calibration:** the same analytical batch supplies fitting observations and receives annotations. Calibration coverage is not held-out validation.
6. **Carbon-only default:** the default carbon fit supplies the shared runtime model but does not establish calibrated coverage for sulfur, chlorine, or bromine.
7. **MS-DIAL compatibility is runtime-gated:** the tracked method and local executable have not yet completed a real compatibility pilot. The script accepts output only when actual `.mdpeak` headers satisfy the parser contract.
8. **MSP is not parser input:** the positive method requests MSP spectrum export, but the HRMS Utils parser requires compatible tabular `.mdpeak` feature tables. The runtime header gate prevents silently treating MSP as feature-table input.

## Validation completed

The implementation was validated without running real MS-DIAL or the full spectral library:

- Focused pipeline and calibration tests passed.
- CLI default and method-override behavior were tested.
- The positive method was checked for positive mode and `[M+H]+` only.
- The local executable was confirmed executable when present; clean checkouts skip this optional-local-bundle assertion.
- Ruff passed.
- Pyrefly passed.
- Documentation build passed.
- `git diff --check` passed.
- Generated documentation-site artifacts were removed.
- Independent final review returned `PASS` with no findings.

## Before the first real run

1. Put the positive-mode mzML files under `analysis/`, or provide another `--mzml-glob`.
2. Confirm `MSDIAL/MSDIALCUI` and its runtime companion files remain available and executable.
3. Verify `experiments/msdial_console_params_positive.txt` against the installed MS-DIAL version.
4. Confirm the metadata JSON values accepted by the installed console.
5. Confirm that the MS-DIAL method produces parser-compatible tab-delimited `.mdpeak` files in addition to requested MSP exports.
6. Ensure the output parent directory exists and has enough free space for MS-DIAL output, calibration intermediates, and nested Parquet files.
7. Run without `--overwrite` first.
8. Inspect `msdial.stdout.txt`, `msdial.stderr.txt`, `manifest.json`, `SCHEMA.md`, and calibration plots before treating annotations as usable.

## Repository state at handoff

The changes are currently unstaged on branch `main`; no commit was created. Existing unrelated untracked user data was preserved.
