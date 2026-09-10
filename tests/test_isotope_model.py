import json
from pathlib import Path

import pytest

from hrms_utils.hrms_core import CalibratedIsotopicModel
from hrms_utils.isotope_model import (
    load_calibrated_isotopic_model,
    save_calibrated_isotopic_model,
)


MODEL = CalibratedIsotopicModel(
    alpha=1.5,
    beta=0.98,
    offset=0.0,
    hetero_lower_a=1.7,
    hetero_lower_b=-0.1,
    hetero_upper_a=18.0,
    hetero_upper_b=-0.29,
)


def test_model_json_round_trip_and_manual_edit(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    save_calibrated_isotopic_model(MODEL, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert set(payload["model"]) == {
        "alpha",
        "beta",
        "offset",
        "hetero_lower_a",
        "hetero_lower_b",
        "hetero_upper_a",
        "hetero_upper_b",
    }
    payload["model"]["alpha"] = 2.25
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    loaded = load_calibrated_isotopic_model(path)
    assert loaded.alpha == 2.25
    assert isinstance(loaded.alpha, float)


def test_model_json_rejects_malformed_values_and_version(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    save_calibrated_isotopic_model(MODEL, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported model schema_version"):
        load_calibrated_isotopic_model(path)
    payload["schema_version"] = 1
    payload["model"]["beta"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TypeError, match="real number"):
        load_calibrated_isotopic_model(path)
    payload["model"]["beta"] = "0.98"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TypeError, match="real number"):
        load_calibrated_isotopic_model(path)
    payload["model"]["beta"] = float("nan")
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_calibrated_isotopic_model(path)


def test_model_json_rejects_invalid_math_and_existing_target(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    save_calibrated_isotopic_model(MODEL, path)
    with pytest.raises(FileExistsError):
        save_calibrated_isotopic_model(MODEL, path)
    save_calibrated_isotopic_model(MODEL, path, overwrite=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"]["alpha"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="alpha"):
        load_calibrated_isotopic_model(path)
