"""Human-editable persistence for calibrated isotopic models."""

from __future__ import annotations

import json
import math
import os
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from .hrms_core import CalibratedIsotopicModel

MODEL_PARAMETER_NAMES = (
    "alpha",
    "beta",
    "offset",
    "hetero_lower_a",
    "hetero_lower_b",
    "hetero_upper_a",
    "hetero_upper_b",
)
MODEL_SCHEMA_VERSION = 1


def _validated_model_values(values: Mapping[str, Any]) -> dict[str, float]:
    if set(values) != set(MODEL_PARAMETER_NAMES):
        missing = sorted(set(MODEL_PARAMETER_NAMES) - set(values))
        unknown = sorted(set(values) - set(MODEL_PARAMETER_NAMES))
        raise ValueError(
            "Model parameters must contain exactly the seven runtime fields; "
            f"missing={missing}, unknown={unknown}."
        )

    result: dict[str, float] = {}
    for name in MODEL_PARAMETER_NAMES:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"Model parameter {name!r} must be a real number, not {value!r}."
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Model parameter {name!r} must be finite, got {value!r}.")
        result[name] = numeric_value

    if result["alpha"] <= 0:
        raise ValueError("Model parameter 'alpha' must be greater than zero.")
    if result["offset"] < 0:
        raise ValueError("Model parameter 'offset' must be nonnegative.")
    for name in ("hetero_lower_a", "hetero_upper_a"):
        if result[name] < 0:
            raise ValueError(f"Model parameter {name!r} must be nonnegative.")
    return result


def _model_values(model: CalibratedIsotopicModel) -> dict[str, float]:
    return _validated_model_values(
        {name: getattr(model, name) for name in MODEL_PARAMETER_NAMES}
    )


def save_calibrated_isotopic_model(
    model: CalibratedIsotopicModel,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save a seven-parameter isotopic model as pretty-printed editable JSON.

    The file intentionally contains no checksum. Users may edit the seven values and
    reload the file after satisfying the same structural and mathematical validation.
    """
    destination = Path(path).expanduser()
    values = _model_values(model)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Model file already exists; pass overwrite=True: {destination}"
        )
    if not destination.parent.exists() or not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Model output directory does not exist: {destination.parent}"
        )

    payload = {"schema_version": MODEL_SCHEMA_VERSION, "model": values}
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_calibrated_isotopic_model(path: str | Path) -> CalibratedIsotopicModel:
    """Load and validate one human-editable isotopic model JSON file."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Model file does not exist: {source}")
    try:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Model file is not valid JSON: {source}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError("Model JSON root must be an object.")
    version = payload.get("schema_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != MODEL_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported model schema_version {version!r}; expected {MODEL_SCHEMA_VERSION}."
        )
    values = payload.get("model")
    if not isinstance(values, dict):
        raise TypeError("Model JSON field 'model' must be an object.")
    validated = _validated_model_values(values)
    return CalibratedIsotopicModel(**validated)


__all__ = [
    "MODEL_PARAMETER_NAMES",
    "MODEL_SCHEMA_VERSION",
    "load_calibrated_isotopic_model",
    "save_calibrated_isotopic_model",
]
