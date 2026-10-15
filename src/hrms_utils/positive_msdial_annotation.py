"""Positive-only compatibility API over the canonical polarity-explicit workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from hrms_utils import msdial_annotation as _implementation
from hrms_utils.msdial_annotation import (  # noqa: F401
    DEFAULT_LIBRARY,
    METADATA_COLUMNS,
    MSDIAL_columns_to_read,
    IsotopicToleranceModel,
    _absolute,
    _find_mzml_paths,
    _model_from_fit,
    _run_msdial,
    _validate_metadata_values,
    _write_metadata_csv,
    annotate_chromatogram_with_formulas,
    calibrate_isotopic_parameters,
    get_chromatogram,
    load_calibrated_isotopic_model,
    save_calibrated_isotopic_model,
)

DEFAULT_METHOD = _implementation.default_method("positive")


@dataclass(frozen=True)
class PositiveMsdialAnnotationConfig:
    """Configuration for one complete positive-mode processing run."""

    mzml_glob: str = "analysis/*.mzML"
    msdial_path: Path = Path("MSDIAL/MSDIALCUI")
    spectral_library_path: Path = DEFAULT_LIBRARY
    output_directory: Path = Path("analysis/msdial-positive-annotation")
    msdial_method: Path | None = DEFAULT_METHOD
    metadata_values: Mapping[str, Any] | None = None
    calibration_elements: tuple[str, ...] = ("C",)
    overwrite: bool = False
    load_model_path: Path | None = None
    save_model_path: Path | None = None
    calibrated_error_bound_scale: float = 1.0
    minimum_element_count_error: Mapping[str, int] | None = None
    maximum_element_count_error: Mapping[str, int] | None = None
    carbon_tolerance: int | float | None = None


def _generic_config(
    config: PositiveMsdialAnnotationConfig,
) -> _implementation.MsdialAnnotationConfig:
    return _implementation.MsdialAnnotationConfig(
        polarity="positive",
        **{field.name: getattr(config, field.name) for field in fields(config)},
    )


def _method_is_positive(method_path: Path) -> None:
    _implementation._validate_method_polarity(method_path, "positive")


def _validate_library(library_path: Path) -> None:
    _implementation._validate_library(library_path, "positive")


def _validate_inputs(
    config: PositiveMsdialAnnotationConfig,
) -> _implementation.ValidatedInputs:
    return _implementation._validate_inputs(_generic_config(config))


def _workflow_dependencies() -> _implementation._WorkflowDependencies:
    return _implementation._WorkflowDependencies(
        _run_msdial,
        get_chromatogram,
        calibrate_isotopic_parameters,
        annotate_chromatogram_with_formulas,
        load_calibrated_isotopic_model,
        save_calibrated_isotopic_model,
    )


def main(config: PositiveMsdialAnnotationConfig) -> Path:
    return _implementation._run_workflow(
        _generic_config(config), _workflow_dependencies()
    )


def build_argument_parser() -> argparse.ArgumentParser:
    return _implementation._build_argument_parser(positive_compatibility=True)


def configuration_from_arguments(
    arguments: argparse.Namespace,
) -> PositiveMsdialAnnotationConfig:
    values = vars(arguments).copy()
    values["spectral_library_path"] = values.pop("spectral_library")
    values["load_model_path"] = values.pop("load_model")
    values["save_model_path"] = values.pop("save_model")
    values["calibration_elements"] = tuple(values["calibration_elements"])
    return PositiveMsdialAnnotationConfig(**values)


@dataclass(frozen=True)
class PositiveMsdialWorkflowParameters:
    """Convenience parameters for the complete positive-mode workflow."""

    msdial_path: Path = Path("MSDIAL/MSDIALCUI")
    msdial_method: Path | None = DEFAULT_METHOD
    output_directory: Path = Path("analysis/msdial-positive-annotation")
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


def run_positive_msdial_annotation(
    *,
    mzml_glob: str = "analysis/*.mzML",
    msdial_path: Path = Path("MSDIAL/MSDIALCUI"),
    spectral_library_path: Path = DEFAULT_LIBRARY,
    output_directory: Path = Path("analysis/msdial-positive-annotation"),
    msdial_method: Path | None = DEFAULT_METHOD,
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
    """Run positive MS-DIAL annotation with the complete CLI option surface."""
    return main(
        PositiveMsdialAnnotationConfig(
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


def run_complete_positive_msdial_workflow(
    mzml_source: str | Path,
    spectral_library: str | Path | None,
    parameters: PositiveMsdialWorkflowParameters,
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
    return run_positive_msdial_annotation(
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
