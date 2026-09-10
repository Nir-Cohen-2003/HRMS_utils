# Complete mzML → MS-DIAL → HRMS_utils workflow

This is the standalone hand-off document for converting mzML data into MS-DIAL
features, reading those features with HRMS_utils, optionally calibrating isotopic
parameters, and producing isotope-constrained formula and fragment annotations.
The pipeline is:

```text
mzML → MS-DIAL 5 LC-MS processing/deconvolution → tabular feature export
     → get_chromatogram() → optional blank/filtering
     → optional isotope calibration → annotate_chromatogram_with_formulas()
     → Parquet (recommended) or a scalar CSV summary
```

MS-DIAL is external software. HRMS_utils does not download, install, or validate an
MS-DIAL executable for you. All Python commands below are run through Pixi; do not
run repository Python files with a system Python.

## 1. Prerequisites and versioning

Install [Pixi](https://pixi.sh/) and clone this repository. From the repository root:

```bash
pixi install
pixi run python -c "import hrms_utils; print(hrms_utils.hrms_core.__version__)"
```

You need:

- centroided or profile mzML files whose MS-DIAL method settings match the data;
- an MS-DIAL console release downloaded from the [official MS-DIAL Workbench
  releases](https://github.com/systemsomicslab/MsdialWorkbench/releases);
- an MS-DIAL method file exported or written for the *same console version*;
- enough disk space for MS-DIAL intermediate and project output;
- a processed HRMS_utils spectral-library Parquet only if library-mode isotope
  calibration is wanted.

The official [MS-DIAL 5 console guide](https://systemsomicslab.github.io/msdial5tutorial/consoleapp.html)
currently documents `MSDIALCUI` on Linux and `MSDIALCUI.exe` on Windows. Extract the
release outside this repository. The repository-local `MSDIAL/MSDIALCUI` is the
pipeline default and must already have executable permission; the workflow never
changes its mode. For another installation, make its binary executable before use:

```bash
chmod +x /absolute/path/MSDIALCUI
```

The upstream guide claims macOS support, but this repository does not verify a macOS
console asset. Pin the executable, method file, and metadata format as one tested
workflow. MS-DIAL method keys and accepted values are version-specific; this document
intentionally does not enumerate upstream keys that HRMS_utils cannot guarantee. Use
the [official method example](https://github.com/systemsomicslab/msdial5tutorial/blob/main/msdial_console_param4lipidomics.txt)
and the [official console documentation](https://systemsomicslab.github.io/msdial5tutorial/consoleapp.html)
for the authoritative method schema.

For an HRMS LC-MS method, select values appropriate to the instrument and data,
including the correct polarity, centroid/profile MS1 and MS2 data type, `Target
omics`, `Ionization: ESI`, and `Machine category: LCMS`. Set an
instrument-appropriate MS1 alignment tolerance explicitly. Do not copy a tolerance
or an ion/adduct list without checking the acquisition.

## 2. Prepare the MS-DIAL 5 input metadata CSV

The documented MS-DIAL 5 console route takes a metadata CSV rather than a single mzML
argument. Use absolute paths. The first eight columns must be in this order:

```text
file_path,file_name,file_type,class_id,acquisition_type,batch_order,analytical_order,factor
```

Each row describes one input file. `file_path` points to an mzML file; `file_name`
is its sample name; the remaining values must use the values accepted by the selected
MS-DIAL release. Keep polarity and acquisition type consistent with the method. For
example, a minimal CSV has the following shape (replace values with valid upstream
values):

```csv
file_path,file_name,file_type,class_id,acquisition_type,batch_order,analytical_order,factor
/absolute/data/sample.mzML,sample,Sample,1,Sample,1,1,NA
```

Do not assume that this example's categorical values are valid for every MS-DIAL
version. Validate them against the release's console guide.

## 3. Run MS-DIAL deconvolution

The direct, documented MS-DIAL 5 path is the recommended path because it takes the
metadata CSV and exposes the current console interface. On Linux:

```bash
pixi run -e experiments /absolute/path/MSDIALCUI lcms \
  -i /absolute/path/input_metadata.csv \
  -o /absolute/path/msdial-output \
  -m /absolute/path/method.txt \
  -p
```

On Windows PowerShell:

```powershell
& "C:\absolute\path\MSDIALCUI.exe" lcms `
  -i "C:\absolute\path\input_metadata.csv" `
  -o "C:\absolute\path\msdial-output" `
  -m "C:\absolute\path\method.txt" `
  -p
```

`-p` requests project output in the documented MS-DIAL 5 console workflow. Output
names and optional exports depend on the release and method. The console guide
currently describes per-sample `.mdpeak` and `.mdmsp`, alignment `.mdalign`, `.mdmsp`
and `.mzTab`, and a timestamped `.mdproject` when project output is requested.
These files are not interchangeable: HRMS_utils needs a compatible tab-delimited
feature table, not an arbitrary MSP, project, alignment, or mzTab file.

### Existing HRMS_utils MS-DIAL wrapper (limitations are important)

`hrms_utils.formats.msdial.run_msdial_lcmsdda()` exists, but it is a legacy wrapper,
not a user-facing MS-DIAL 5 CLI. It has not been verified against a real current
MS-DIAL 5 console run. It accepts an **input directory**, whereas the documented v5
route accepts a metadata CSV; it does not add `-p`; and it combines older executable
conventions with the v5 `lcms` command. Prefer the direct command above.

The wrapper signature is:

```python
run_msdial_lcmsdda(
    input_dir: Path | str,
    output_dir: Path | str,
    polarity: Literal["positive", "negative"],
    params_file: Path | str | None = None,
    config: MSDialRunnerConfig | None = None,
) -> subprocess.CompletedProcess
```

Arguments and behavior:

| Argument | Type/default | Meaning |
|---|---|---|
| `input_dir` | `Path | str`, required | Existing directory containing raw input files. |
| `output_dir` | `Path | str`, required | Output directory; created if absent. |
| `polarity` | `"positive"` or `"negative"`, required | Rejects any other value. |
| `params_file` | `Path | str | None`, `None` | Existing custom parameter file. If omitted, a legacy parameter file is generated in `output_dir`. |
| `config` | `MSDialRunnerConfig | None`, `None` | Executable and generated-parameter settings. `None` uses defaults and executable auto-detection. |

`MSDialRunnerConfig` is a dataclass with exactly these currently exposed fields:

| Field | Type/default | Meaning |
|---|---|---|
| `msdial_path` | `Path | None = None` | Explicit executable. `None` searches common locations and `packages/msdial`. |
| `threads` | `int = 20` | Thread count used only when generating a parameter file. |
| `minimum_peak_height` | `int = 100000` | Minimum peak height used only when generating a parameter file. |

If no `params_file` is supplied, `_generate_params_file()` writes
`Msdial-lcms-dda-positive-params.txt` or `Msdial-lcms-dda-negative-params.txt`,
including the repository's positive or negative adduct list. The wrapper invokes the
executable as `executable lcms -i input_dir -o output_dir -m params_file`, captures
stdout/stderr, and raises on a nonzero return code or recognized error text. The
returned `CompletedProcess` is therefore available only after a successful run.

`tests/formats/test_msdial_cli.py` tests executable discovery, generated positive and
negative parameter files, and a real run when local mzML fixtures and an executable are
available. It is not a v5 metadata-CSV integration test. The real-run tests are
skipped when fixtures are absent and may create persistent output under
`tests/output/msdial_cli`.

`experiments/msdial_console_params.txt` is an unused negative-mode example for the
legacy parameter-file path and is unchanged. The automated runner selects
`msdial_console_params_{polarity}.txt`: packaged for installed APIs, under
`experiments/` for repository tasks. The negative snapshot matches the positive
snapshot except `Ion mode: Negative` and `adduct list: [M-H]-` (positive: `[M+H]+`). Its export settings remain
version-sensitive (it requests MSP and mzTab exports), so the runner still requires an
actual parser-compatible `.mdpeak` header after MS-DIAL and fails if none is produced.
Supply a version-validated method with `--msdial-method` when needed.

### One-command polarity-explicit pipeline

For a complete local run, use the generic installed command or repository task.
Polarity is required and never inferred. The runner sorts and deduplicates mzML paths,
creates the MS-DIAL metadata CSV, runs the direct `MSDIALCUI lcms` command, calibrates
isotopic bounds from `P` (positive) or `N` (negative) rows of the configured library, and annotates
every parser-compatible feature table:

```bash
pixi run -e experiments run_msdial_annotation -- \
  --polarity negative \
  --mzml-glob 'analysis/*.mzML' \
  --msdial-path 'MSDIAL/MSDIALCUI' \
  --metadata-values '{"file_type":"Sample","class_id":"1","acquisition_type":"Sample","factor":"NA","batch_order":1}' \
  --spectral-library '/data/nirc/spectral_libs/iibr_260615/iibr_260708.parquet' \
  --output-directory 'analysis/msdial-negative-annotation'
```

The default values are exactly the paths shown above, except that
`--metadata-values` and `--polarity` are required; `--msdial-method` selects the
matching polarity snapshot unless overridden. `hrms-msdial-annotate` exposes the same
options from installed Python. The old `hrms-positive-msdial-annotate` and positive
Pixi task remain forced-positive compatibility aliases without a polarity option. The metadata JSON must contain
exactly `file_type`, `class_id`, `acquisition_type`, and `factor` as nonempty strings,
and `batch_order` as a positive integer. Values must not contain control characters.
The generated CSV has the eight MS-DIAL columns in the required order and uses absolute
mzML paths; `analytical_order` is assigned deterministically in sorted-path order.

Options supported by this orchestration are:

| Option | Default | Meaning |
|---|---|---|
| `--mzml-glob` | `analysis/*.mzML` | Input glob. Every match must be a regular file with case-sensitive `.mzML` suffix; duplicate stems fail. |
| `--msdial-path` | `MSDIAL/MSDIALCUI` | Local executable. It must be executable and have sibling `.dll`, `.runtimeconfig.json`, and `.deps.json` files. |
| `--polarity` | required | Exactly `positive` or `negative`. |
| `--msdial-method` | polarity-selected snapshot | Exactly one matching ion-mode setting; custom adducts allowed for explicit overrides. |
| `--metadata-values` | required | JSON object described above. |
| `--spectral-library` | `/data/nirc/spectral_libs/iibr_260615/iibr_260708.parquet` | Processed library with the required schema and at least one row in the selected `P`/`N` mode. |
| `--output-directory` | `analysis/msdial-{polarity}-annotation` | New publication directory. Existing output requires `--overwrite`. |
| `--calibration-elements` | `C` | One or more of `C S Cl Br`; selected elements share one fitted model during fit-now runs. |
| `--load-model` | disabled | Load one human-editable model JSON and skip spectral-library validation, matching, and calibration. |
| `--save-model` | disabled | Also save the newly fitted model JSON at this path; incompatible with `--load-model`. |
| `--calibrated-error-bound-scale` | `1.0` | Finite positive multiplier for runtime heteroscedastic log-error widths; baseline model is unchanged. |
| `--minimum-element-count-error` | disabled | JSON integer count-margin floors for `C/S/Cl/Br`. |
| `--maximum-element-count-error` | disabled | JSON integer count-margin ceilings for `C/S/Cl/Br`. |
| `--overwrite` | disabled | Atomically replace an existing output directory only after the new run succeeds. |

The script invokes exactly:

```text
MSDIALCUI lcms -i metadata.csv -o msdial-output -m method.txt -p
```

It captures stdout, stderr, and the return code. It never calls the legacy
`run_msdial_lcmsdda()` wrapper, chmods or installs MS-DIAL, alters the executable, or
collects the complete library into memory. Calibration search settings are fixed to
precursor 5 ppm, MS2 10 ppm, dot-product strictly greater than 0.9, and information
score strictly greater than 1.0. Annotation uses precursor 3 ppm and both raw and
normalized fragment tolerances of 5 ppm.

The output directory is published atomically only after all steps succeed:

```text
msdial-negative-annotation/
├── msdial-metadata.csv
├── msdial-output/                 # retained MS-DIAL files and diagnostics
├── calibration-input/             # validated tables used for calibration
├── calibration/results.txt        # plus calibration PNGs (fit-now runs)
├── calibration/model.json         # seven baseline runtime parameters (fit-now runs)
├── annotations/<source-stem>.parquet    # nested Polars annotation results
├── summaries/<source-stem>.csv          # scalar downstream summaries
├── manifest.json                  # relative paths, command, config, model, thresholds
└── SCHEMA.md                      # exact generated Parquet/CSV schema and caveats
```

New outputs use exact source table stems, without a generated `table_001_` prefix;
case-insensitive duplicate table stems fail before calibration copies or annotation.
The manifest records output mappings, baseline/effective models, polarity, scale,
requested count maps, and applicability policy. Existing generated results are not
renamed. `--load-model` skips library access/fitting, but still runs MS-DIAL and does
not create calibration directories.

For detected peaks, native count clamps use `floor(n_hat + 0.5)` as the center,
not an endpoint midpoint. After scaling and rounding endpoints, directional margins
are clamped minimum-first then maximum, with physical/global limits. Both maps
`{"C":2}` mean center ±2. Missing/below-threshold isotope peaks have no calibrated
center and remain **NOT_APPLIED**. See the [installed API reference](../reference/isotopic-calibration-api.md)
for exact equations, integer validation, parameter-bag usage, and model reuse caveats.

The prior positive workflow completed a real dataset run. Negative support is
fixture-tested, not validated on a real negative dataset. Validate method metadata
and inspect the first `.mdpeak` before interpreting results. Wrong-polarity methods
or libraries, missing companions, invalid headers, and console failures stop without
replacing previous outputs.

## 4. Validate and parse an MS-DIAL feature export

Before processing a batch, inspect a representative output. The filename extension
alone is not sufficient: an `.mdpeak` file can still be the wrong MS-DIAL artifact.
`get_chromatogram()` reads a **headered, tab-delimited** table. It reads these exact
source headers (literal tab separators, not commas):

```text
Peak ID	Scan	RT left(min)	RT (min)	RT right (min)	Precursor m/z	Height	Adduct	Isotope	MSMS spectrum	MS1 isotopes
```

The source columns are parsed with these types:

| Source header | Type |
|---|---|
| `Peak ID` | `Int64` |
| `Scan` | `Int64` |
| `RT left(min)`, `RT (min)`, `RT right (min)`, `Precursor m/z`, `Height` | `Float64` |
| `Adduct` | `String` |
| `Isotope` | `Int32` |
| `MSMS spectrum`, `MS1 isotopes` | `String` encoded spectra, converted to lists |

Use:

```bash
head -n 2 /absolute/path/msdial-output/sample.mdpeak
```

```python
from pathlib import Path
from hrms_utils.formats.msdial import get_chromatogram

chromatogram = get_chromatogram(Path("/absolute/path/msdial-output/sample.mdpeak"))
print(chromatogram.schema)
print(chromatogram.head())
```

The exact returned Polars schema is:

| Returned column | Polars type |
|---|---|
| `Peak ID` | `Int64` |
| `RT (min)`, `Precursor_mz_MSDIAL`, `Height` | `Float64` |
| `Precursor_type_MSDIAL` | `String` |
| `Isotope` | `Int32` |
| `msms_m/z`, `msms_intensity` | `List(Float64)` |
| `isobars` | `List(Int64)` |
| `msms_m/z_cleaned`, `msms_intensity_cleaned` | `List(Float64)` |
| `energy_is_too_low`, `energy_is_too_high` | `Boolean` |
| `ms1_isotopes_m/z`, `ms1_isotopes_intensity` | `List(Float64)` |

The parser computes `isobars`, cleaned MS/MS lists, and energy annotations. MS/MS
intensities are max-normalized during parsing; MS1 isotope intensities remain
absolute. The annotation function below consumes the raw `msms_m/z` and
`msms_intensity` columns, not the parser's cleaned columns.

## 5. Preprocess features (optional blank subtraction and filtering)

Annotation does not automatically select monoisotopic features or remove unusable
spectra. Use Polars expressions. For example:

```python
import polars as pl
from hrms_utils.formats.msdial import get_chromatogram

chromatogram = get_chromatogram("/absolute/path/sample.mdpeak")
usable = chromatogram.filter(
    pl.col("Isotope") == 0,
    pl.col("ms1_isotopes_m/z").is_not_null(),
    pl.col("ms1_isotopes_m/z").list.len() > 0,
    pl.col("ms1_isotopes_intensity").is_not_null(),
    pl.col("ms1_isotopes_intensity").list.len() > 0,
    pl.col("msms_m/z").is_not_null(),
    pl.col("msms_m/z").list.len() > 0,
    pl.col("msms_intensity").is_not_null(),
    pl.col("msms_intensity").list.len() > 0,
)
```

To subtract a blank, parse both files and call `subtract_blank_frame(sample_df,
blank_df, config)`. Its `blank_config` fields are:

| Field | Type/default | Meaning |
|---|---|---|
| `ms1_mass_tolerance` | `float = 3e-6` | Relative MS1 tolerance; values above `0.0001` are interpreted as ppm and multiplied by `1e-6`. |
| `dRT_min` | `float = 0.1` | Minimum RT difference in minutes for ordinary matching. |
| `ratio` | `float | int = 5` | Sample/blank intensity ratio threshold. |
| `use_ms2` | `bool = False` | Enable MS/MS matching. |
| `dRT_min_with_ms2` | `float = 0.3` | RT difference in minutes when MS/MS is available. |
| `ms2_fit` | `float = 0.85` | Required MS/MS fit when enabled. |
| `ms2_mass_tolerance` | `float = 5e-6` | Relative MS2 tolerance. |
| `noise_threshold` | `float = 0.005` | Noise threshold used by MS/MS matching. |

## 6. Annotate isotope bounds, formulas, and fragments

Call the function with a Polars `DataFrame` having the required columns shown in the
parser schema:

```python
from hrms_utils.formats.msdial import annotate_chromatogram_with_formulas

annotated = annotate_chromatogram_with_formulas(
    usable,
    max_bounds=None,
    precursor_mass_accuracy_ppm=3.0,
    fragment_mass_accuracy_ppm=5.0,
    normalized_fragment_mass_accuracy_ppm=4.0,
    isotopic_mass_accuracy_ppm=2.0,
    isotopic_minimum_intensity=5e4,
)
```

Every currently exposed argument is listed here:

| Argument | Type/default/units | Meaning |
|---|---|---|
| `chromatogram` | `pl.DataFrame`, required | Parsed feature table. Required columns are `Precursor_mz_MSDIAL`, `ms1_isotopes_m/z`, `ms1_isotopes_intensity`, `msms_m/z`, and `msms_intensity`. |
| `max_bounds` | `dict | None = None` | Optional user-specified upper element-count bounds for isotope deduction. `None` deduces bounds from the data. |
| `precursor_mass_accuracy_ppm` | `float = 3.0` ppm | Precursor tolerance used by isotope deduction and bounded precursor decomposition. |
| `fragment_mass_accuracy_ppm` | `float = 5.0` ppm | Raw fragment m/z tolerance used during fragment cleaning. |
| `normalized_fragment_mass_accuracy_ppm` | `float = 4.0` ppm | Normalized-mass tolerance used during fragment cleaning/normalization. |
| `isotopic_mass_accuracy_ppm` | `float = 2.0` ppm | Tolerance for matching observed isotope positions. |
| `isotopic_minimum_intensity` | `float = 50000.0` absolute intensity | Minimum MS1 isotope intensity considered by pattern deduction. |
| `isotopic_model` | `CalibratedIsotopicModel`, default shown below | Calibrated heteroscedastic model used to turn isotope intensities into element-count bounds. |

The built-in model default is:

```python
CalibratedIsotopicModel(
    alpha=1.32762, beta=0.981853, offset=0.0,
    hetero_lower_a=5.5, hetero_lower_b=-0.2,
    hetero_upper_a=16.7, hetero_upper_b=-0.29,
)
```

The operation is: isotope evidence produces per-feature element bounds; bounded mass
decomposition generates precursor candidates; candidate formulas are exploded into
rows; raw MS/MS lists are cleaned and fragment formulas/errors are calculated. A
single input feature can therefore produce multiple output rows. Failed isotope
pattern deduction is filtered out, and no final candidate ranking or selection is
performed.

### Formula order and annotation output schema

All formula arrays have 12 entries in this exact order:

```text
H, C, N, O, F, Na, P, S, Cl, K, Br, I
```

The isotope plugin's bound vector has 24 entries: the 12 minima followed by the 12
maxima. The returned DataFrame retains all parsed columns and adds:

| Added column | Polars type | Meaning |
|---|---|---|
| `min_bounds`, `max_bounds` | `Array(Int32, 12)` | Per-element isotope-derived minimum/maximum counts. |
| `precursor_formula` | `Array(Int32, 12)` | One candidate formula per exploded row. |
| `precursor_formula_str` | `String` | Human-readable candidate formula. |
| `precursor_errors_ppm` | `Float64` | Precursor mass error for that candidate. |
| `cleaned_msms_mz`, `cleaned_msms_intensity` | `List(Float64)` | Retained cleaned/normalized fragments and intensities. |
| `cleaned_spectrum_formulas` | `List(Array(Int32, 12))` | Formula arrays assigned to retained fragments. |
| `cleaned_spectrum_formulas_str` | `List(String)` | String form of fragment formulas. |
| `cleaned_fragment_errors_ppm` | `List(Float64)` | Fragment mass errors in ppm. |
| `explained_intensity` | `Float64` | Cleaned MS/MS intensity sum divided by the original raw intensity sum. |

Important scientific semantics:

- `Precursor_mz_MSDIAL` is decomposed directly. The function does **not** convert
  adduct m/z to neutral mass or infer charge; do not interpret candidates as neutral
  formulas for arbitrary adducts.
- Isotopic evidence constrains C, S, Cl, and Br. Other element counts are bounded by
  the mass-decomposition configuration and supplied bounds.
- Candidates are formula hypotheses, not compound identifications.
- Tolerances are ppm unless explicitly marked as absolute intensity or minutes.

## 7. Optional isotope-parameter calibration

Calibration fits the relationship needed to infer isotope-count bounds from observed
intensities. It is advisory: its report is not loaded automatically by annotation.
Use representative, independently acquired data for validation; fitting coverage alone
does not establish generalization.

### Calibration inputs and source modes

The required chromatogram directory must contain parser-compatible tabular exports.
The default file glob is `**/*.mdpeak`; change it with `--chromatogram-glob`.
Calibration keeps monoisotopic (`Isotope == 0`) features with MS1 and MS/MS data.

There are two ground-truth sources:

1. **Library mode:** one or more processed library Parquet files, filtered by `--library-ion-mode` (`P` by default, `N` for negative). The
   script reads `ion_mode`, `precursor_mz`, `cleaned_normalized_mz`,
   `cleaned_normalized_intensity`, `precursor_formula_array`, `name`, `inchikey`, and
   `smiles`. The formula array uses the 12-element order above. Library matching also
   uses the query feature's raw `msms_m/z` and `msms_intensity`.
2. **CSV mode:** a CSV containing at least `Molecular Formula` and `Monoisotopic Mass`.
   Extra columns are ignored. The mass is compared directly to
   `Precursor_mz_MSDIAL`; supply values in the same ion/adduct mass domain. Each
   formula must match unambiguously within each chromatogram file.

Library and CSV sources may be used separately or together. At least one source is
required. XLSX fitting-pair export requires library metadata: any CSV run combined
with `--emit-pairs` or `--pairs-output-file` fails before calibration. Create output
parent directories first; missing inputs and parents fail fast. Existing requested
report/plot files are overwritten.

### Complete calibration CLI reference

The Pixi task is:

```bash
pixi run -e experiments isotope_fit -- --help
```

Run it as `pixi run -e experiments isotope_fit -- [options]`. Required options:

| Option | Type, repeatability, default | Valid values/meaning |
|---|---|---|
| `--chromatogram-dir PATH` | `Path`, required, not repeatable | Existing directory containing feature files. |
| `--output-file PATH` | `Path`, required, not repeatable | Text report path. Its parent must already exist. |

Input/source options:

| Option | Type, repeatability, default | Valid values/meaning |
|---|---|---|
| `--chromatogram-glob GLOB` | `str`, once, `**/*.mdpeak` | Glob relative to chromatogram directory. Must match at least one file. |
| `--spectral-library PATH` | `Path`, repeatable (`action=append`), no default | Processed library Parquet filtered to `--library-ion-mode`. Repeat for multiple libraries. Can combine with CSV. |
| `--ground-truth-csv PATH` | `Path`, once, omitted by default | CSV with `Molecular Formula` and `Monoisotopic Mass`. Can combine with library. |
| `--elements ELEMENT [ELEMENT ...]` | one or more strings, once, default `C` | Choices are exactly `C`, `S`, `Cl`, `Br`; controls which target isotopes are fitted. Together they form one model, not independent models. |

Tolerance and quality options (all are `float`, supplied once, with defaults below):

| Option | Default | Valid range enforced | Meaning |
|---|---:|---|---|
| `--ground-truth-ms1-tolerance-ppm` | `5.0` | finite and `> 0` | CSV ground-truth precursor matching tolerance. |
| `--precursor-tolerance-ppm` | `5.0` | finite and `> 0` | Library precursor matching tolerance. |
| `--ms2-tolerance-ppm` | `10.0` | finite and `> 0` | MS/MS matching tolerance used in library similarity. |
| `--dot-product-threshold` | `0.9` | finite, `0 ≤ value ≤ 1` | Minimum library/query dot-product similarity. |
| `--info-score-threshold` | `1.0` | finite and `≥ 0` | Minimum library spectral information score. |
| `--ms1-mass-tolerance-ppm` | `5.0` | finite and `> 0` | MS1 isotope/mass matching tolerance used during fitting-data construction. |
| `--isotopic-mass-tolerance-ppm` | `2.0` | finite and `> 0` | Isotope-position matching tolerance. |
| `--minimum-isotopic-peak-intensity` | `10000.0` | finite and `≥ 0` | Minimum observed isotope intensity used for calibration data. |
| `--target-success-rate` | `0.99` | `0 < value ≤ 1` | Desired fraction of calibration points inside fitted bounds. |

Output options:

| Option | Type/default | Interaction |
|---|---|---|
| `--emit-pairs` | boolean flag, default false | Requests fitting-pair XLSX. Requires library mode; CSV + this flag fails. Without `--pairs-output-file`, the default is `<output-stem>_fitting_pairs.xlsx` beside the report. |
| `--pairs-output-file PATH` | `Path`, omitted by default | Requests fitting-pair XLSX at this exact path and implies pair emission. Requires library mode and an existing parent directory. |
| `--library-ion-mode {P,N}` | `P` | Library polarity filter; does not affect CSV-only calibration. |
| `--save-model PATH` | `Path`, omitted by default | Saves the seven effective runtime parameters as pretty-printed, human-editable `model.json`; existing target files are refused. |

The argparse parser does not impose ranges on the glob, paths, element list beyond the
listed choices, or `--pairs-output-file`; filesystem and source validation happens
when the command starts. It also does not expose every field of the internal
`isotopic_config` dataclass. The CLI-exposed options above are the complete public
command-line set. Internal defaults not exposed as flags include fragment cleaning
(5 ppm), DBE bounds (-0.5 to 30), RT tolerance (30 seconds), plotting settings, and
mass-accuracy threshold (200 Da).

### Calibration examples

CSV-only calibration (no pair export):

```bash
mkdir -p /absolute/path/calibration
pixi run -e experiments isotope_fit -- \
  --chromatogram-dir /absolute/path/msdial-output \
  --ground-truth-csv /absolute/path/ground-truth.csv \
  --output-file /absolute/path/calibration/results.txt \
  --elements C S Cl Br \
  --ground-truth-ms1-tolerance-ppm 5
```

Library calibration with fitting pairs:

```bash
mkdir -p /absolute/path/calibration
pixi run -e experiments isotope_fit -- \
  --chromatogram-dir /absolute/path/msdial-output \
  --spectral-library /absolute/path/library.parquet \
  --output-file /absolute/path/calibration/results.txt \
  --elements C S Cl Br \
  --pairs-output-file /absolute/path/calibration/fitting_pairs.xlsx
```

Mixed-source calibration is allowed, but pair export is not:

```bash
pixi run -e experiments isotope_fit -- \
  --chromatogram-dir /absolute/path/msdial-output \
  --spectral-library /absolute/path/library.parquet \
  --ground-truth-csv /absolute/path/ground-truth.csv \
  --output-file /absolute/path/calibration/results.txt
```

### Calibration artifacts and fitted model

Add `--save-model /absolute/path/model.json` to save a reusable model while retaining
all existing report and plot outputs:

```bash
pixi run -e experiments isotope_fit -- \
  --chromatogram-dir /absolute/path/msdial-output \
  --spectral-library /absolute/path/library.parquet \
  --output-file /absolute/path/calibration/results.txt \
  --save-model /absolute/path/model.json
```

For `results.txt`, the script writes the requested report and a sibling
`results_calibration_bounds.png`. It writes `results_error_percentiles.png` when
percentile plotting is enabled by the internal configuration. Library pair mode also
writes the requested fitting-pairs XLSX. The report fits:

```text
expected = alpha * (observed + offset) ** beta
expected_lower = expected_hat * exp(-wL)
expected_upper = expected_hat * exp(+wU)
wL = aL * (observed + offset) ** bL
wU = aU * (observed + offset) ** bU
```

Map report values to the runtime `CalibratedIsotopicModel` as follows:

| Report field | Model field |
|---|---|
| `alpha`, `beta`, `offset` | `alpha`, `beta`, `offset` |
| `aL`, `bL` | `hetero_lower_a`, `hetero_lower_b` |
| `aU`, `bU` | `hetero_upper_a`, `hetero_upper_b` |

The report is a human-readable handoff. `model.json` is the simple reusable handoff;
it has no checksum and is intentionally human-editable. Its format is:

```json
{
  "schema_version": 1,
  "model": {
    "alpha": 1.5,
    "beta": 0.98,
    "offset": 0.0,
    "hetero_lower_a": 1.7,
    "hetero_lower_b": -0.1,
    "hetero_upper_a": 18.0,
    "hetero_upper_b": -0.29
  }
}
```

Load it for the complete positive pipeline without providing a spectral library:

```bash
pixi run -e experiments run_positive_msdial_annotation -- \
  --load-model /absolute/path/model.json \
  --metadata-values '{"file_type":"Sample","class_id":"1","acquisition_type":"Sample","factor":"NA","batch_order":1}'
```

The pipeline still runs MS-DIAL and formula annotation, but skips library validation,
matching, and isotope calibration. Fit-now runs automatically write
`calibration/model.json` inside the output directory. The manifest records whether the
model was `fitted` or `loaded`.

For a programmatic calibration, `main(config)` returns an `IsotopicToleranceModel`,
which can be converted explicitly:

```python
from experiments.find_optimal_isotopic_parameters import main, isotopic_config
from hrms_utils.formats.msdial import annotate_chromatogram_with_formulas
from hrms_utils.hrms_core import CalibratedIsotopicModel

config = isotopic_config(
    chromatogram_dir=..., output_file=...,  # supply source settings too
)
fitted = main(config)
model = CalibratedIsotopicModel(
    alpha=fitted.alpha,
    beta=fitted.beta,
    offset=fitted.offset,
    hetero_lower_a=fitted.hetero_lower_a,
    hetero_lower_b=fitted.hetero_lower_b,
    hetero_upper_a=fitted.hetero_upper_a,
    hetero_upper_b=fitted.hetero_upper_b,
)
annotated = annotate_chromatogram_with_formulas(usable, isotopic_model=model)
```

At the lower level, use the public persistence helpers:

```python
from hrms_utils import load_calibrated_isotopic_model
model = load_calibrated_isotopic_model("model.json")
annotated = annotate_chromatogram_with_formulas(usable, isotopic_model=model)
```

The loader checks schema version, all seven required fields, real finite numeric
values, and plugin-safe mathematical constraints. Human edits to valid values load
normally. It does not require calibration, plotting, or spectral-library dependencies.
At the lower level, pass the same model to the Polars expression
`deduce_isotopic_pattern(..., model=model)`.

## 8. Export the result

Parquet is the recommended output because it preserves nested lists and fixed-size
Polars arrays:

```python
annotated.write_parquet("/absolute/path/annotated.parquet")
```

For a portable CSV summary, select scalar fields explicitly rather than flattening
nested arrays implicitly:

```python
annotated.select([
    "Peak ID", "RT (min)", "Precursor_mz_MSDIAL", "Height",
    "Precursor_type_MSDIAL", "Isotope", "precursor_formula_str",
    "precursor_errors_ppm", "explained_intensity",
]).write_csv("/absolute/path/annotation-summary.csv")
```

The full annotated Parquet contains every parsed column plus the annotation columns
listed above. Because precursor candidates are exploded, preserve `Peak ID` and
`precursor_formula_str` together when joining results back to source features.

## 9. Troubleshooting and limitations

- **`FileNotFoundError` for MS-DIAL:** pass the absolute executable path to the direct
  command, or to `MSDialRunnerConfig(msdial_path=...)` for the legacy wrapper. Ensure
  Linux execute permission is set.
- **MS-DIAL rejects the command or method:** use the method file and metadata schema
  from the exact release. Method keys are version-specific; consult the upstream
  console guide rather than copying `experiments/msdial_console_params.txt`.
- **Parser reports missing columns or malformed spectra:** inspect the first line and
  a data row. The parser needs the exact tab-delimited headers and encoded spectrum
  fields listed in section 4. MSP, `.mdproject`, `.mdalign`, and arbitrary `.mdpeak`
  files are not guaranteed to work.
- **No features survive annotation:** check `Isotope == 0`, nonempty MS1 isotope
  lists, intensity scale, isotope ppm tolerance, and the default minimum isotope
  intensity. Also check that the selected MS-DIAL polarity/method matches the data.
- **Too many precursor candidates:** tighten `precursor_mass_accuracy_ppm`, provide
  appropriate `max_bounds`, improve isotope data, or rank/filter candidates downstream.
- **No fragments are retained:** check raw `msms_m/z` and `msms_intensity`, then
  adjust fragment and normalized-fragment ppm tolerances for the instrument. The
  function does not use parser-cleaned MS/MS columns as its input.
- **Calibration has no matches:** ensure chromatogram files are parser-compatible,
  CSV masses use the same ion/adduct domain as `Precursor_mz_MSDIAL`, library files
  contain the selected library ion mode and required columns, and tolerances are appropriate.
- **CSV + fitting-pairs fails:** this is intentional. Pair rows require library names,
  InChIKeys, and spectra; remove `--emit-pairs`/`--pairs-output-file` or use
  library-only calibration.
- **Calibration report is not automatically applied:** construct
  `CalibratedIsotopicModel` explicitly as shown above and pass it to annotation.
- **Scientific interpretation:** formula candidates are not identifications; direct
  m/z decomposition does not neutralize adducts; and calibration should be evaluated
  on independent representative data.

For the concise calibration companion, see
[Calibrate isotopic parameters](../how-to/calibrate-isotopic-parameters.md). This
document is the exhaustive workflow and option reference.
