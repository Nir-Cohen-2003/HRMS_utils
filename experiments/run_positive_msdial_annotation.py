"""Compatibility adapter for the installed positive MS-DIAL workflow API."""

from __future__ import annotations

import argparse
from pathlib import Path

from hrms_utils import msdial_annotation as _generic
from hrms_utils import positive_msdial_annotation as _implementation
from hrms_utils.positive_msdial_annotation import (  # noqa: F401
    DEFAULT_LIBRARY,
    METADATA_COLUMNS,
    MSDIAL_columns_to_read,
    IsotopicToleranceModel,
    PositiveMsdialAnnotationConfig,
    PositiveMsdialWorkflowParameters,
    _find_mzml_paths,
    _method_is_positive,
    _model_from_fit,
    _run_msdial,
    _validate_inputs,
    _validate_library,
    _validate_metadata_values,
    _write_metadata_csv,
    annotate_chromatogram_with_formulas,
    calibrate_isotopic_parameters,
    configuration_from_arguments,
    get_chromatogram,
    load_calibrated_isotopic_model,
    run_complete_positive_msdial_workflow,
    run_positive_msdial_annotation,
    save_calibrated_isotopic_model,
)

DEFAULT_METHOD = Path("experiments/msdial_console_params_positive.txt")
# Shared runner preserves precursor_mass_accuracy_ppm=3.0, fragment_mass_accuracy_ppm=5.0,
# normalized_fragment_mass_accuracy_ppm=5.0.


def build_argument_parser() -> argparse.ArgumentParser:
    parser = _implementation.build_argument_parser()
    parser.set_defaults(msdial_method=DEFAULT_METHOD)
    return parser


def main(config: PositiveMsdialAnnotationConfig) -> Path:
    """Honor legacy local patches without mutating installed module globals."""
    dependencies = _generic._WorkflowDependencies(
        _run_msdial,
        get_chromatogram,
        calibrate_isotopic_parameters,
        annotate_chromatogram_with_formulas,
        load_calibrated_isotopic_model,
        save_calibrated_isotopic_model,
    )
    return _generic._run_workflow(_implementation._generic_config(config), dependencies)


if __name__ == "__main__":
    main(configuration_from_arguments(build_argument_parser().parse_args()))
