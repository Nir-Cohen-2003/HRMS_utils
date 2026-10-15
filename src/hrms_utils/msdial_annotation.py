"""Run the polarity-explicit mzML -> MS-DIAL -> calibrated annotation workflow."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import polars as pl

from hrms_utils.isotope_calibration import (
    IsotopicToleranceModel,
    isotopic_config,
    main as calibrate_isotopic_parameters,
)
from hrms_utils.formats.msdial import (
    MSDIAL_columns_to_read,
    annotate_chromatogram_with_formulas,
    get_chromatogram,
)
from hrms_utils.element_count_error import (
    validate_carbon_tolerance,
    validate_element_count_errors,
)
from hrms_utils.hrms_core import CalibratedIsotopicModel
from hrms_utils.isotope_model import (
    load_calibrated_isotopic_model,
    save_calibrated_isotopic_model,
)

DEFAULT_LIBRARY = Path("/data/nirc/spectral_libs/iibr_260615/iibr_260708.parquet")
Polarity = Literal["positive", "negative"]
LIBRARY_ION_MODES = {"positive": "P", "negative": "N"}
ION_MODE_VALUES = {
    "positive": frozenset({"positive", "pos", "+"}),
    "negative": frozenset({"negative", "neg", "-"}),
}


def _validate_polarity(polarity: Polarity) -> None:
    if polarity not in ("positive", "negative"):
        raise ValueError("polarity must be 'positive' or 'negative'.")


def default_method(polarity: Polarity) -> Path:
    _validate_polarity(polarity)
    return Path(__file__).with_name(f"msdial_console_params_{polarity}.txt")


def _output_directory(config: MsdialAnnotationConfig) -> Path:
    return (
        config.output_directory
        if config.output_directory is not None
        else Path(f"analysis/msdial-{config.polarity}-annotation")
    )


REQUIRED_METADATA_KEYS = frozenset(
    {"file_type", "class_id", "acquisition_type", "factor", "batch_order"}
)
METADATA_COLUMNS = (
    "file_path",
    "file_name",
    "file_type",
    "class_id",
    "acquisition_type",
    "batch_order",
    "analytical_order",
    "factor",
)
REQUIRED_LIBRARY_COLUMNS = frozenset(
    {
        "ion_mode",
        "precursor_mz",
        "cleaned_normalized_mz",
        "cleaned_normalized_intensity",
        "precursor_formula_array",
        "name",
        "inchikey",
        "smiles",
    }
)


@dataclass(frozen=True, kw_only=True)
class MsdialAnnotationConfig:
    """Configuration for one processing run with an explicit polarity."""

    polarity: Polarity

    mzml_glob: str = "analysis/*.mzML"
    msdial_path: Path = Path("MSDIAL/MSDIALCUI")
    spectral_library_path: Path = DEFAULT_LIBRARY
    output_directory: Path | None = None
    msdial_method: Path | None = None
    metadata_values: Mapping[str, Any] | None = None
    calibration_elements: tuple[str, ...] = ("C",)
    overwrite: bool = False
    load_model_path: Path | None = None
    save_model_path: Path | None = None
    calibrated_error_bound_scale: float = 1.0
    minimum_element_count_error: Mapping[str, int] | None = None
    maximum_element_count_error: Mapping[str, int] | None = None
    carbon_tolerance: int | float | None = None


@dataclass(frozen=True)
class ValidatedInputs:
    mzml_paths: tuple[Path, ...]
    method_path: Path
    msdial_path: Path
    library_path: Path | None
    load_model_path: Path | None


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_metadata_values(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if values is None:
        raise ValueError(
            "--metadata-values is required and must contain a JSON object."
        )
    if set(values) != REQUIRED_METADATA_KEYS:
        unknown = sorted(set(values) - REQUIRED_METADATA_KEYS)
        missing = sorted(REQUIRED_METADATA_KEYS - set(values))
        raise ValueError(
            f"Metadata keys must be exactly {sorted(REQUIRED_METADATA_KEYS)}; missing={missing}, unknown={unknown}."
        )
    result = dict(values)
    for key in ("file_type", "class_id", "acquisition_type", "factor"):
        value = result[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Metadata value {key!r} must be a nonempty string.")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError(f"Metadata value {key!r} contains a control character.")
    batch_order = result["batch_order"]
    if (
        isinstance(batch_order, bool)
        or not isinstance(batch_order, int)
        or batch_order <= 0
    ):
        raise ValueError("Metadata value 'batch_order' must be a positive integer.")
    return result


def _parse_metadata_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"metadata values are not valid JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("metadata values must be a JSON object.")
    return _validate_metadata_values(parsed)


def _find_mzml_paths(pattern: str) -> tuple[Path, ...]:
    matches = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No mzML files matched --mzml-glob {pattern!r}.")
    if any(not path.is_file() for path in matches):
        bad = [str(path) for path in matches if not path.is_file()]
        raise ValueError(f"mzML glob matched non-files: {bad}")
    if any(path.suffix != ".mzML" for path in matches):
        bad = [str(path) for path in matches if path.suffix != ".mzML"]
        raise ValueError(
            f"mzML glob matched paths with the wrong suffix (expected .mzML): {bad}"
        )
    resolved = sorted({_absolute(path) for path in matches})
    stems: dict[str, Path] = {}
    for path in resolved:
        key = path.stem.casefold()
        if key in stems:
            raise ValueError(
                f"Duplicate mzML stems are not allowed: {stems[key]} and {path}"
            )
        stems[key] = path
    return tuple(resolved)


def _validate_method_polarity(method_path: Path, polarity: Polarity) -> None:
    settings: list[str] = []
    for line in method_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lstrip("#").strip().casefold() in {
            "ion mode",
            "ionization mode",
        }:
            settings.append(value.strip().casefold())
    if len(settings) != 1 or settings[0] not in ION_MODE_VALUES[polarity]:
        raise ValueError(
            f"MS-DIAL method must contain exactly one {polarity} ion-mode setting "
            f"(for example 'Ion mode: {polarity.title()}'); polarity is not inferred."
        )


def _validate_library(library_path: Path, polarity: Polarity) -> None:
    schema = pl.scan_parquet(library_path).collect_schema()
    missing = sorted(REQUIRED_LIBRARY_COLUMNS - set(schema.names()))
    if missing:
        raise ValueError(f"Spectral library is missing required columns: {missing}")
    selected = (
        pl.scan_parquet(library_path)
        .filter(pl.col("ion_mode") == LIBRARY_ION_MODES[polarity])
        .select("ion_mode")
        .limit(1)
        .collect(engine="streaming")
    )
    if selected.height == 0:
        raise ValueError(
            f"Spectral library contains no {polarity}-mode rows (ion_mode == {LIBRARY_ION_MODES[polarity]!r})."
        )


def _validate_output_safety(output: Path, inputs: ValidatedInputs) -> None:
    if output.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {output}")
    output_resolved = _absolute(output)
    source_paths = [*inputs.mzml_paths, inputs.method_path, inputs.msdial_path]
    if inputs.library_path is not None:
        source_paths.append(inputs.library_path)
    for source in source_paths:
        source_resolved = _absolute(source)
        if output_resolved == source_resolved:
            raise ValueError(
                f"Output directory overlaps an input/resource path: {output} and {source}"
            )
        if not source.is_file() and _is_relative_to(output_resolved, source_resolved):
            raise ValueError(
                f"Output directory overlaps an input/resource directory: {output} and {source}"
            )
        if _is_relative_to(source_resolved, output_resolved):
            raise ValueError(
                f"Output directory would contain an input/resource path: {output} and {source}"
            )
    if not output.parent.exists() or not output.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory does not exist: {output.parent}"
        )
    if not os.access(output.parent, os.W_OK):
        raise PermissionError(
            f"Output parent directory is not writable: {output.parent}"
        )
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output}")


def _validate_inputs(config: MsdialAnnotationConfig) -> ValidatedInputs:
    _validate_polarity(config.polarity)
    _validate_error_bound_scale(config.calibrated_error_bound_scale)
    validate_element_count_errors(
        config.minimum_element_count_error, config.maximum_element_count_error
    )
    validate_carbon_tolerance(config.carbon_tolerance)
    if config.load_model_path is not None and config.save_model_path is not None:
        raise ValueError("--load-model and --save-model cannot be used together.")
    load_model_path = (
        _absolute(config.load_model_path) if config.load_model_path is not None else None
    )
    if load_model_path is not None:
        # Validate before starting the expensive MS-DIAL subprocess.
        load_calibrated_isotopic_model(load_model_path)
    mzml_paths = _find_mzml_paths(config.mzml_glob)
    method_path = _absolute(
        config.msdial_method
        if config.msdial_method is not None
        else default_method(config.polarity)
    )
    msdial_path = _absolute(config.msdial_path)
    library_path = _absolute(config.spectral_library_path)
    if not method_path.is_file():
        raise FileNotFoundError(f"MS-DIAL method does not exist: {method_path}")
    if not msdial_path.is_file() or not os.access(msdial_path, os.X_OK):
        raise FileNotFoundError(
            f"MSDIALCUI is missing or not executable: {msdial_path}"
        )
    for suffix in (".dll", ".runtimeconfig.json", ".deps.json"):
        sibling = msdial_path.with_suffix(suffix)
        if not sibling.is_file():
            raise FileNotFoundError(f"MS-DIAL runtime companion is missing: {sibling}")
    _validate_method_polarity(method_path, config.polarity)
    _validate_metadata_values(config.metadata_values)
    if load_model_path is None:
        if not library_path.is_file():
            raise FileNotFoundError(f"Spectral library does not exist: {library_path}")
        _validate_library(library_path, config.polarity)
    inputs = ValidatedInputs(
        mzml_paths,
        method_path,
        msdial_path,
        None if load_model_path is not None else library_path,
        load_model_path,
    )
    _validate_output_safety(_absolute(_output_directory(config)), inputs)
    if not config.calibration_elements or any(
        element not in {"C", "S", "Cl", "Br"} for element in config.calibration_elements
    ):
        raise ValueError(
            "Calibration elements must contain one or more of C, S, Cl, Br."
        )
    return inputs


def _write_metadata_csv(
    paths: Sequence[Path], values: Mapping[str, Any], destination: Path
) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(METADATA_COLUMNS)
        for order, path in enumerate(paths, start=1):
            writer.writerow(
                [
                    str(path),
                    path.stem,
                    values["file_type"],
                    values["class_id"],
                    values["acquisition_type"],
                    values["batch_order"],
                    order,
                    values["factor"],
                ]
            )


def _msdial_command(
    inputs: ValidatedInputs, metadata_csv: Path, output: Path
) -> list[str]:
    return [
        str(inputs.msdial_path),
        "lcms",
        "-i",
        str(metadata_csv),
        "-o",
        str(output),
        "-m",
        str(inputs.method_path),
        "-p",
    ]


def _run_msdial(
    inputs: ValidatedInputs, metadata_csv: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    output.mkdir()
    command = _msdial_command(inputs, metadata_csv, output)
    result = subprocess.run(
        command,
        cwd=inputs.msdial_path.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    (output / "msdial.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output / "msdial.stderr.txt").write_text(result.stderr, encoding="utf-8")
    (output / "msdial.command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MS-DIAL failed with exit status {result.returncode}; see msdial diagnostics."
        )
    return result


def _header_is_compatible(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().rstrip("\r\n")
    return set(MSDIAL_columns_to_read).issubset(first_line.split("\t"))


def _discover_tables(
    msdial_output: Path,
    parse_table: Callable[[Path], pl.DataFrame] = get_chromatogram,
) -> tuple[list[Path], list[tuple[Path, pl.DataFrame]]]:
    discovered = sorted(
        path for path in msdial_output.rglob("*.mdpeak") if path.is_file()
    )
    compatible: list[tuple[Path, pl.DataFrame]] = []
    for path in discovered:
        if _header_is_compatible(path):
            try:
                compatible.append((path, parse_table(path)))
            except Exception:
                continue
    if not compatible:
        raise RuntimeError(
            "MS-DIAL produced no parser-compatible .mdpeak feature tables."
        )
    return discovered, compatible


def _usable_spectra(dataframe: pl.DataFrame) -> pl.DataFrame:
    return dataframe.filter(
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


def _model_from_fit(fitted: IsotopicToleranceModel) -> CalibratedIsotopicModel:
    # Polars plugin kwargs must contain native Python scalars. scipy curve_fit returns
    # NumPy scalar values, which the plugin serializer cannot decode reliably.
    return CalibratedIsotopicModel(
        alpha=float(fitted.alpha),
        beta=float(fitted.beta),
        offset=float(fitted.offset),
        hetero_lower_a=float(fitted.hetero_lower_a),
        hetero_lower_b=float(fitted.hetero_lower_b),
        hetero_upper_a=float(fitted.hetero_upper_a),
        hetero_upper_b=float(fitted.hetero_upper_b),
    )


def _validate_table_stems(tables: Sequence[tuple[Path, pl.DataFrame]]) -> None:
    stems: dict[str, Path] = {}
    for path, _ in tables:
        key = path.stem.casefold()
        if key in stems:
            raise ValueError(
                f"Duplicate mdpeak stems are not allowed: {stems[key]} and {path}"
            )
        stems[key] = path


def _validate_error_bound_scale(scale: float) -> None:
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or scale <= 0
        or scale > sys.float_info.max
        or not math.isfinite(scale)
    ):
        raise ValueError(
            "calibrated_error_bound_scale must be a finite positive number."
        )


def _annotation_model(
    model: CalibratedIsotopicModel, scale: float
) -> CalibratedIsotopicModel:
    _validate_error_bound_scale(scale)
    effective = replace(
        model,
        hetero_lower_a=float(model.hetero_lower_a * scale),
        hetero_upper_a=float(model.hetero_upper_a * scale),
    )
    if not math.isfinite(effective.hetero_lower_a) or not math.isfinite(
        effective.hetero_upper_a
    ):
        raise ValueError(
            "calibrated_error_bound_scale overflows the model error amplitudes."
        )
    return effective


@dataclass(frozen=True)
class _WorkflowDependencies:
    run_msdial: Callable[..., Any]
    parse_table: Callable[[Path], pl.DataFrame]
    calibrate: Callable[[isotopic_config], IsotopicToleranceModel]
    annotate: Callable[..., pl.DataFrame]
    load_model: Callable[[Path], CalibratedIsotopicModel]
    save_model: Callable[..., None]


def _workflow_dependencies() -> _WorkflowDependencies:
    # Resolve at call time so compatibility adapters can inject local patches without
    # mutating shared generic globals or contaminating subsequent negative runs.
    return _WorkflowDependencies(
        _run_msdial,
        get_chromatogram,
        calibrate_isotopic_parameters,
        annotate_chromatogram_with_formulas,
        load_calibrated_isotopic_model,
        save_calibrated_isotopic_model,
    )


def _path_reference(path: Path, run_root: Path) -> str:
    return os.path.relpath(path, start=run_root)


def _write_schema(
    path: Path,
    annotated: Sequence[tuple[str, pl.DataFrame]],
    summary_columns: Sequence[str],
    polarity: Polarity,
) -> None:
    first = annotated[0][1]
    lines = [
        f"# {polarity.title()} MS-DIAL annotation output schema",
        "",
        "Each `annotations/*.parquet` file contains one row per candidate precursor formula.",
        "The input precursor m/z is decomposed directly; no adduct neutralization is performed.",
        "Formula candidates are ion-formula hypotheses, not neutral formulas or compound identifications.",
        "Library matches are assumed labels, not certified truth or FDR-controlled identifications.",
        "",
        "## Nested Parquet columns",
        "",
        "| Column | Polars dtype |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | `{dtype}` |" for name, dtype in first.schema.items())
    lines.extend(
        [
            "",
            "`precursor_formula` has 12 counts in `H,C,N,O,F,Na,P,S,Cl,K,Br,I` order; `min_bounds` and `max_bounds` have the same order.",
            "One input feature may expand to multiple candidate rows.",
            "",
            "## Scalar CSV summaries",
            "",
            "Each `summaries/*.csv` contains these scalar columns (nested spectra and formula arrays are omitted):",
            "",
            ", ".join(f"`{column}`" for column in summary_columns),
            "",
            "Calibration uses the same batch as annotation; fitting coverage is not holdout validation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _publish(stage: Path, output: Path, overwrite: bool) -> None:
    if output.is_symlink():
        raise ValueError(
            f"Output directory became a symlink before publication: {output}"
        )
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists; pass --overwrite: {output}"
        )
    backup: Path | None = None
    if output.exists():
        backup = output.with_name(f".{output.name}.previous")
        if backup.exists() or backup.is_symlink():
            shutil.rmtree(backup)
        output.rename(backup)
    try:
        stage.rename(output)
    except Exception:
        if backup is not None and not output.exists():
            backup.rename(output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def main(config: MsdialAnnotationConfig) -> Path:
    return _run_workflow(config, _workflow_dependencies())


def _run_workflow(
    config: MsdialAnnotationConfig, dependencies: _WorkflowDependencies
) -> Path:
    values = _validate_metadata_values(config.metadata_values)
    inputs = _validate_inputs(config)
    output = _absolute(_output_directory(config))
    if output.exists() and not config.overwrite:
        raise FileExistsError(
            f"Output directory already exists; pass --overwrite: {output}"
        )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        metadata_csv = stage / "msdial-metadata.csv"
        _write_metadata_csv(inputs.mzml_paths, values, metadata_csv)
        msdial_output = stage / "msdial-output"
        dependencies.run_msdial(inputs, metadata_csv, msdial_output)
        discovered, compatible = _discover_tables(
            msdial_output, dependencies.parse_table
        )
        _validate_table_stems(compatible)

        model_source = "fitted"
        if inputs.load_model_path is not None:
            model = dependencies.load_model(inputs.load_model_path)
            model_source = "loaded"
        else:
            assert inputs.library_path is not None, (
                "A spectral library is required when fitting a model."
            )
            calibration_input = stage / "calibration-input"
            calibration_input.mkdir()
            for table_path, _ in compatible:
                shutil.copy2(
                    table_path,
                    calibration_input / f"{table_path.stem}.mdpeak",
                )
            calibration_report = stage / "calibration" / "results.txt"
            calibration_report.parent.mkdir()
            selected = set(config.calibration_elements)
            fitted = dependencies.calibrate(
                isotopic_config(
                    chromatogram_dir=calibration_input,
                    chromatogram_glob="*.mdpeak",
                    spectral_library_paths=(inputs.library_path,),
                    library_ion_mode=LIBRARY_ION_MODES[config.polarity],
                    output_file=calibration_report,
                    use_library_search_for_ground_truth=True,
                    use_mass_list_for_ground_truth=False,
                    precursor_tolerance_ppm=5.0,
                    ms2_tolerance_ppm=10.0,
                    dot_product_threshold=0.9,
                    info_score_threshold=1.0,
                    use_carbon="C" in selected,
                    use_sulfur="S" in selected,
                    use_chlorine="Cl" in selected,
                    use_bromine="Br" in selected,
                )
            )
            calibration_report_text = calibration_report.read_text(encoding="utf-8")
            if (
                "Calibration hits total (before alignment): 0"
                in calibration_report_text
            ):
                raise ValueError(
                    f"The {config.polarity} spectral-library search produced zero calibration matches; "
                    "check library spectra, precursor/MS2 tolerances, and quality thresholds."
                )
            model = _model_from_fit(fitted)
            calibration_model_path = stage / "calibration" / "model.json"
            dependencies.save_model(model, calibration_model_path)
            if config.save_model_path is not None:
                dependencies.save_model(
                    model, config.save_model_path, overwrite=config.overwrite
                )
        effective_model = _annotation_model(model, config.calibrated_error_bound_scale)
        annotations = stage / "annotations"
        summaries = stage / "summaries"
        annotations.mkdir()
        summaries.mkdir()
        annotated_tables: list[tuple[str, pl.DataFrame]] = []
        summary_columns = [
            "Peak ID",
            "RT (min)",
            "Precursor_mz_MSDIAL",
            "Height",
            "Precursor_type_MSDIAL",
            "Isotope",
            "precursor_formula_str",
            "precursor_errors_ppm",
            "explained_intensity",
        ]
        for table_path, dataframe in compatible:
            annotated = dependencies.annotate(
                _usable_spectra(dataframe),
                precursor_mass_accuracy_ppm=3.0,
                fragment_mass_accuracy_ppm=5.0,
                normalized_fragment_mass_accuracy_ppm=5.0,
                isotopic_model=effective_model,
                minimum_element_count_error=config.minimum_element_count_error,
                maximum_element_count_error=config.maximum_element_count_error,
                carbon_tolerance=config.carbon_tolerance,
            )
            name = table_path.stem
            annotated.write_parquet(annotations / f"{name}.parquet")
            available_summary_columns = [
                column for column in summary_columns if column in annotated.columns
            ]
            annotated.select(available_summary_columns).write_csv(
                summaries / f"{name}.csv"
            )
            annotated_tables.append((name, annotated))
        _write_schema(
            stage / "SCHEMA.md", annotated_tables, summary_columns, config.polarity
        )
        manifest = {
            "command": "hrms-msdial-annotate",
            "msdial_command": _msdial_command(inputs, metadata_csv, msdial_output),
            "config": {
                "polarity": config.polarity,
                "calibrated_error_bound_scale": config.calibrated_error_bound_scale,
                "carbon_tolerance": config.carbon_tolerance,
                "minimum_element_count_error": None
                if config.minimum_element_count_error is None
                else dict(config.minimum_element_count_error),
                "maximum_element_count_error": None
                if config.maximum_element_count_error is None
                else dict(config.maximum_element_count_error),
                "mzml_glob": config.mzml_glob,
                "msdial_path": _path_reference(inputs.msdial_path, stage),
                "msdial_method": _path_reference(inputs.method_path, stage),
                "spectral_library_path": (
                    None
                    if inputs.library_path is None
                    else _path_reference(inputs.library_path, stage)
                ),
                "calibration_elements": list(config.calibration_elements),
                "overwrite": config.overwrite,
                "metadata_values": values,
            },
            "submitted_mzml": [
                _path_reference(path, stage) for path in inputs.mzml_paths
            ],
            "discovered_mdpeak": [_path_reference(path, stage) for path in discovered],
            "compatible_mdpeak": [
                _path_reference(path, stage) for path, _ in compatible
            ],
            "output_mappings": [
                {
                    "source_mdpeak": _path_reference(path, stage),
                    "calibration_input_mdpeak": f"calibration-input/{path.stem}.mdpeak"
                    if model_source == "fitted"
                    else None,
                    "annotation": f"annotations/{path.stem}.parquet",
                    "summary": f"summaries/{path.stem}.csv",
                }
                for path, _ in compatible
            ],
            "element_count_error_policy": {
                "center": "floor(n_hat + 0.5)",
                "order": "scale log widths, ceil/floor endpoints, nonnegative directional margins, minimum, maximum, physical/global caps",
                "without_calibrated_center": "NOT_APPLIED (missing precursor or absent/below-threshold isotope)",
            },
            "model": {
                "mode": model_source,
                "source": model_source,
                "path": (
                    "calibration/model.json"
                    if model_source == "fitted"
                    else str(inputs.load_model_path)
                ),
                "parameters": asdict(model),
                "calibrated_error_bound_scale": config.calibrated_error_bound_scale,
                "annotation_parameters": asdict(effective_model),
            },
            "thresholds": {
                "calibration_precursor_ppm": 5.0,
                "calibration_ms2_ppm": 10.0,
                "calibration_dot_product_greater_than": 0.9,
                "calibration_information_score_greater_than": 1.0,
                "annotation_precursor_ppm": 3.0,
                "annotation_fragment_ppm": 5.0,
                "annotation_normalized_fragment_ppm": 5.0,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        _publish(stage, output, config.overwrite)
    except Exception as error:
        # Preserve failed staging output for MS-DIAL/parser diagnostics. The stage is
        # outside the publication path, and deleting it would discard the only stdout,
        # stderr, and intermediate feature tables needed to repair a failed run.
        print(f"Pipeline failed; retained staging diagnostics at {stage}: {error}")
        raise
    return output


def build_argument_parser() -> argparse.ArgumentParser:
    return _build_argument_parser(positive_compatibility=False)


def _build_argument_parser(*, positive_compatibility: bool) -> argparse.ArgumentParser:
    defaults = MsdialWorkflowParameters()
    parser = argparse.ArgumentParser(
        description="Run mzML through MS-DIAL and calibrated HRMS_utils annotation."
    )
    if not positive_compatibility:
        parser.add_argument(
            "--polarity", choices=("positive", "negative"), required=True
        )
    parser.add_argument("--mzml-glob", default="analysis/*.mzML")
    parser.add_argument("--msdial-path", type=Path, default=defaults.msdial_path)
    parser.add_argument("--spectral-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--output-directory", type=Path, default=defaults.output_directory
    )
    parser.add_argument("--msdial-method", type=Path, default=defaults.msdial_method)
    parser.add_argument(
        "--metadata-values",
        type=_parse_metadata_json,
        required=True,
        help="JSON object with exactly file_type, class_id, acquisition_type, factor, batch_order.",
    )
    parser.add_argument(
        "--calibration-elements",
        nargs="+",
        choices=("C", "S", "Cl", "Br"),
        default=list(defaults.calibration_elements),
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        help="Load one validated model.json and skip library calibration.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        help="Also save the fitted model.json outside the output directory.",
    )
    parser.add_argument(
        "--calibrated-error-bound-scale",
        type=float,
        default=1.0,
        help="Finite positive multiplier for annotation log-error widths (baseline model unchanged).",
    )
    parser.add_argument(
        "--minimum-element-count-error",
        type=_parse_element_count_error_json,
        help='JSON integer margin floors for detected C/S/Cl/Br estimates, e.g. {"C":2}.',
    )
    parser.add_argument(
        "--maximum-element-count-error",
        type=_parse_element_count_error_json,
        help="JSON integer margin ceilings, applied after floors; no effect without a calibrated center.",
    )
    parser.add_argument(
        "--carbon-tolerance",
        type=float,
        default=None,
        help="Nonnegative carbon-count tolerance around the raw calibrated estimate.",
    )
    if positive_compatibility:
        parser.set_defaults(
            msdial_method=default_method("positive"),
            output_directory=Path("analysis/msdial-positive-annotation"),
        )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_element_count_error_json(value: str) -> dict[str, int]:
    try:
        parsed = json.loads(value)
        minimum, _ = validate_element_count_errors(parsed, None)
        if minimum is None:
            raise ValueError("element count errors must be a JSON object, not null")
        return minimum
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def configuration_from_arguments(
    arguments: argparse.Namespace,
) -> MsdialAnnotationConfig:
    return MsdialAnnotationConfig(
        polarity=arguments.polarity,
        mzml_glob=arguments.mzml_glob,
        msdial_path=arguments.msdial_path,
        spectral_library_path=arguments.spectral_library,
        output_directory=arguments.output_directory,
        msdial_method=arguments.msdial_method,
        metadata_values=arguments.metadata_values,
        calibration_elements=tuple(arguments.calibration_elements),
        overwrite=arguments.overwrite,
        load_model_path=arguments.load_model,
        save_model_path=arguments.save_model,
        calibrated_error_bound_scale=arguments.calibrated_error_bound_scale,
        minimum_element_count_error=arguments.minimum_element_count_error,
        maximum_element_count_error=arguments.maximum_element_count_error,
        carbon_tolerance=arguments.carbon_tolerance,
    )


@dataclass(frozen=True)
class MsdialWorkflowParameters:
    """Optional parameters; the complete wrapper receives polarity separately."""

    msdial_path: Path = Path("MSDIAL/MSDIALCUI")
    msdial_method: Path | None = None
    output_directory: Path | None = None
    metadata_values: Mapping[str, Any] | None = None
    calibration_elements: tuple[str, ...] = ("C",)
    overwrite: bool = False
    load_model_path: Path | None = None
    save_model_path: Path | None = None
    calibrated_error_bound_scale: float = 1.0
    minimum_element_count_error: Mapping[str, int] | None = None
    maximum_element_count_error: Mapping[str, int] | None = None
    carbon_tolerance: int | float | None = None


def _source_to_glob(source: str | Path) -> str:
    source_path = Path(source).expanduser()
    if source_path.is_file():
        if source_path.suffix != ".mzML":
            raise ValueError(f"mzML source must have .mzML suffix: {source_path}")
        return str(source_path)
    if source_path.is_dir():
        return str(source_path / "*.mzML")
    return str(source)


def run_msdial_annotation(
    *,
    polarity: Polarity,
    mzml_glob: str = "analysis/*.mzML",
    msdial_path: Path = Path("MSDIAL/MSDIALCUI"),
    spectral_library_path: Path = DEFAULT_LIBRARY,
    output_directory: Path | None = None,
    msdial_method: Path | None = None,
    metadata_values: Mapping[str, Any] | None = None,
    calibration_elements: tuple[str, ...] = ("C",),
    overwrite: bool = False,
    load_model_path: Path | None = None,
    save_model_path: Path | None = None,
    calibrated_error_bound_scale: float = 1.0,
    minimum_element_count_error: Mapping[str, int] | None = None,
    maximum_element_count_error: Mapping[str, int] | None = None,
    carbon_tolerance: int | float | None = None,
) -> Path:
    """Run explicitly polarity-selected MS-DIAL annotation with the complete CLI option surface."""
    return main(
        MsdialAnnotationConfig(
            polarity=polarity,
            mzml_glob=mzml_glob,
            msdial_path=msdial_path,
            spectral_library_path=spectral_library_path,
            output_directory=output_directory,
            msdial_method=msdial_method,
            metadata_values=metadata_values,
            calibration_elements=calibration_elements,
            overwrite=overwrite,
            load_model_path=load_model_path,
            save_model_path=save_model_path,
            calibrated_error_bound_scale=calibrated_error_bound_scale,
            minimum_element_count_error=minimum_element_count_error,
            maximum_element_count_error=maximum_element_count_error,
            carbon_tolerance=carbon_tolerance,
        )
    )


def run_complete_msdial_workflow(
    mzml_source: str | Path,
    spectral_library: str | Path | None,
    parameters: MsdialWorkflowParameters,
    *,
    polarity: Polarity,
) -> Path:
    """Run MS-DIAL, calibration or model loading, and formula annotation."""
    if spectral_library is None and parameters.load_model_path is None:
        raise ValueError(
            "spectral_library may be None only when load_model_path is set"
        )
    library_path = (
        Path(spectral_library).expanduser()
        if spectral_library is not None
        else DEFAULT_LIBRARY
    )
    return run_msdial_annotation(
        polarity=polarity,
        mzml_glob=_source_to_glob(mzml_source),
        msdial_path=parameters.msdial_path,
        spectral_library_path=library_path,
        output_directory=parameters.output_directory,
        msdial_method=parameters.msdial_method,
        metadata_values=parameters.metadata_values,
        calibration_elements=parameters.calibration_elements,
        overwrite=parameters.overwrite,
        load_model_path=parameters.load_model_path,
        save_model_path=parameters.save_model_path,
        calibrated_error_bound_scale=parameters.calibrated_error_bound_scale,
        minimum_element_count_error=parameters.minimum_element_count_error,
        maximum_element_count_error=parameters.maximum_element_count_error,
        carbon_tolerance=parameters.carbon_tolerance,
    )


if __name__ == "__main__":
    main(configuration_from_arguments(build_argument_parser().parse_args()))
