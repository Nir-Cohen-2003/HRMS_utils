"""Compatibility adapter for the installed isotope-calibration API."""
from __future__ import annotations

from pathlib import Path

from hrms_utils import isotope_calibration as _implementation
from hrms_utils.isotope_calibration import (
    IsotopicToleranceModel,
    build_argument_parser,
    configuration_from_arguments,
    fit_isotopic_tolerance_parameters,  # noqa: F401
    formula_to_array,  # noqa: F401
    get_chromatogram,  # noqa: F401
    isotopic_config,
    np,  # noqa: F401
    pl,  # noqa: F401
)

# Re-export the historical names, including private helpers imported by existing users/tests.
for _name, _value in vars(_implementation).items():
    if _name not in {"__name__", "__package__", "__loader__", "__spec__"}:
        globals()[_name] = _value

# Preserve the repository experiment's historical default path.
DEFAULT_PAIRS_OUTPUT_FILE = Path("experiments/isotopic_fitting_pairs.xlsx")


def main(config: isotopic_config) -> IsotopicToleranceModel:
    """Delegate while honoring monkeypatches applied to this legacy module."""
    for _name in (
        "get_chromatogram",
        "fit_isotopic_tolerance_parameters",
        "formula_to_array",
        "pl",
        "np",
    ):
        setattr(_implementation, _name, globals()[_name])
    return _implementation.main(config)


if __name__ == "__main__":
    arguments = build_argument_parser().parse_args()
    main(configuration_from_arguments(arguments))
