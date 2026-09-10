# Installed calibration and annotation API

The workflow is available from ordinary installed `hrms_utils` Python and console
commands; imports do not require the repository's `experiments` directory.

## Complete workflow: explicit polarity

```python
from pathlib import Path
from hrms_utils import MsdialWorkflowParameters, run_complete_msdial_workflow

output = run_complete_msdial_workflow(
    "analysis/*.mzML",  # file, directory, or glob
    Path("library.parquet"),
    MsdialWorkflowParameters(
        msdial_path=Path("MSDIAL/MSDIALCUI"),
        output_directory=Path("analysis/results"),
        metadata_values={
            "file_type": "Sample", "class_id": "1",
            "acquisition_type": "Sample", "factor": "NA", "batch_order": 1,
        },
        calibrated_error_bound_scale=1.0,
        minimum_element_count_error={"C": 2},
        maximum_element_count_error={"C": 2},
    ),
    polarity="negative",  # required: "positive" or "negative"; never inferred
)
```

`MsdialAnnotationConfig(polarity=...)` and the keyword-only shortcut
`run_msdial_annotation(polarity=..., ...)` expose the complete CLI option surface.
Both require polarity with no default. The complete wrapper takes polarity only as
its required keyword; its reusable parameters bag does not duplicate polarity.

Omitted methods select packaged `msdial_console_params_{polarity}.txt`. Negative
settings match positive byte-for-byte except `Ion mode: Negative` and
`adduct list: [M-H]-` (positive uses `[M+H]+`). An explicit `msdial_method` override
must contain exactly one matching ion-mode setting; custom adduct lists are allowed.
The omitted output directory is `analysis/msdial-{polarity}-annotation`.

### Fit now or load a model

| Option | Execution |
|---|---|
| No `load_model_path` / `--load-model` | MS-DIAL → matching library `P` (positive) or `N` (negative) rows → calibration → baseline model save → annotation |
| `load_model_path=Path("model.json")` / `--load-model model.json` | MS-DIAL → load model → annotation; no library validation, access, matching, or fitting |

For loaded-model runs, pass `None` as the complete wrapper's spectral library and
set `load_model_path` in its parameters. Loading **does not skip MS-DIAL**.
`load_model_path` and `save_model_path` are mutually exclusive. A fit-now run always
writes `calibration/model.json`; `save_model_path` optionally saves another copy.

```bash
hrms-msdial-annotate --polarity negative \
  --mzml-glob 'analysis/*.mzML' \
  --spectral-library library.parquet \
  --metadata-values '{"file_type":"Sample","class_id":"1","acquisition_type":"Sample","factor":"NA","batch_order":1}' \
  --calibrated-error-bound-scale 2 \
  --minimum-element-count-error '{"C":2}' \
  --maximum-element-count-error '{"C":2}'
```

For reuse, replace `--spectral-library library.parquet` with
`--load-model calibration/model.json`. Use `--help` for all unchanged metadata,
resource, calibration-element, output, and overwrite options.

## Error-bound controls

`calibrated_error_bound_scale` defaults to `1.0` and must be finite and positive
(not a boolean). The workflow derives an annotation-only model by multiplying just
`hetero_lower_a` and `hetero_upper_a`. It never mutates the baseline loaded/fitted
model or saved file. These are **heteroscedastic log-error widths**, not ppm or
linear count deviations:

```text
lower = n_hat * exp(-scale * width_lower)
upper = n_hat * exp(+scale * width_upper)
```

Thus scale 2 doubles log widths, not necessarily the resulting integer margins.
It is relative to persisted runtime coefficients, not any widened fitter plot.

`minimum_element_count_error` and `maximum_element_count_error` each default to
`None`. Supply optional mappings with exactly spelled keys `C`, `S`, `Cl`, `Br`.
Values must be integers from 0 through 2,147,483,647; booleans, floats, negatives,
and unknown elements are rejected. An overlapping minimum must not exceed its
maximum. Omitted elements remain unchanged; empty maps also make no changes.

These arguments are keyword-only on `run_msdial_annotation`,
`run_positive_msdial_annotation`, `annotate_chromatogram_with_formulas`, and
`.mass_decomposition.deduce_isotopic_pattern`. Complete wrappers receive them
only through their parameters bags (no duplicate override precedence). New fields
are appended to legacy dataclasses, preserving prior positional construction.

For each requested element with a detected calibrated isotope peak, native code uses:

```text
n_hat = alpha * (isotope_intensity + offset)^beta * p0 / (p1 * precursor_intensity)
center = floor(n_hat + 0.5)
L = ceil(lower)                 # after log-width scaling
U = floor(upper)
lower_error = max(0, center - L)
upper_error = max(0, U - center)
# Independently for each direction:
error = max(configured_minimum, error)  # if supplied
error = min(configured_maximum, error)  # if supplied
bounds = [center - lower_error, center + upper_error]
```

The reconstructed interval is intersected with nonnegative and global element
limits. `min=max=2` means rounded expected count ±2 subject to these limits.
`min={"C":3}` sets a three-count floor; `max={"C":3}` sets a three-count ceiling.
The center is **not** the midpoint of asymmetric endpoints. In a narrow interval
whose rounded endpoints exclude the center, directional errors start at zero;
requesting clamps can intentionally restore the rounded center. Caps can leave an
empty interval; no candidate is invented at a cap.

Missing precursors and absent/below-threshold isotope peaks have no calibrated
center: count clamps are **NOT_APPLIED**, preserving existing missing-precursor and
one-sided detection-limit behavior. Both maps `None` use the unchanged native path.
The output remains `Array(Int32, 24)` (12 minima then 12 maxima, element order
`H,C,N,O,F,Na,P,S,Cl,K,Br,I`); no per-row status column is added. The manifest records
the requested maps and applicability policy, not a claim that every row was clamped.

## Standalone calibration and editable model

```python
from pathlib import Path
from hrms_utils import calibrate_isotopic_parameters, load_calibrated_isotopic_model

fitted = calibrate_isotopic_parameters(
    chromatogram_dir=Path("msdial-output"),
    output_file=Path("calibration/results.txt"),
    spectral_library_paths=(Path("library.parquet"),),
    library_ion_mode="N",  # standalone default remains "P" for compatibility
    elements=("C",),
    save_model_path=Path("calibration/model.json"),
)
model = load_calibrated_isotopic_model("calibration/model.json")
```

The calibration CLI provides `--library-ion-mode {P,N}` (default `P`). CSV-only
calibration is unaffected. Existing advanced options remain available through
`IsotopeCalibrationConfig` and `run_isotope_calibration(config)`. Library matching
preserves square-root-intensity dot product (`intensity_power=0.5`, `mass_power=0.0`).

The one human-editable JSON contains schema version 1 and seven runtime parameters,
with no checksum or context file. Model reuse is operator-controlled: the file
contains no evidence of polarity suitability or held-out calibration quality.

## Outputs and positive compatibility

Outputs keep the existing layout: reports/plots, `manifest.json`, `SCHEMA.md`, and
nested Parquet annotations plus scalar CSV summaries. Newly written files use exact
source `.mdpeak` stems, including spaces:

```text
calibration-input/sample.mdpeak   # fit-now only
annotations/sample.parquet
summaries/sample.csv
```

There is no generated `table_001_` prefix or sanitization. Case-insensitive duplicate
mzML stems fail before MS-DIAL; duplicate compatible table stems fail before
calibration copies or annotation. `manifest.output_mappings` links each source table
to its outputs (calibration input is null in loaded mode). The manifest records
polarity, scale, maps, baseline `model.parameters`, and effective
`model.annotation_parameters`.

Existing positive Python names, `hrms-positive-msdial-annotate`, and the positive
experiment/Pixi task remain forced-positive compatibility interfaces without a
polarity argument or deprecation warnings. The new filename convention applies to
these interfaces too: migrate consumers of `table_*.parquet` to `*.parquet` or
manifest mappings. Existing generated analysis results are not renamed.

The generic repository task is `pixi run -e experiments run_msdial_annotation --
--polarity negative ...`; repository tasks use editable experiment methods, while
installed APIs never read `experiments`.

Annotation defaults remain 3/5/4 ppm for the direct chromatogram function and 3/5/5
ppm for complete workflows. Formula decomposition uses measured precursor m/z
directly without adduct neutralization. All candidate rows remain; these are
ion-formula hypotheses, not compound identifications. Fixture tests are not a real
negative-data MS-DIAL validation.
