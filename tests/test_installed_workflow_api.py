from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import hrms_utils.isotope_calibration as calibration_module
from hrms_utils.isotope_calibration import (
    calibrate_isotopic_parameters,
    isotopic_config,
)
from hrms_utils.isotope_model import load_calibrated_isotopic_model, save_calibrated_isotopic_model
from hrms_utils.positive_msdial_annotation import (
    PositiveMsdialWorkflowParameters,
    _source_to_glob,
    run_complete_positive_msdial_workflow,
)
from hrms_utils.hrms_core import CalibratedIsotopicModel


def test_human_editable_model_round_trip(tmp_path: Path) -> None:
    model = CalibratedIsotopicModel(
        alpha=1.5,
        beta=0.98,
        offset=0.0,
        hetero_lower_a=1.7,
        hetero_lower_b=-0.1,
        hetero_upper_a=18.0,
        hetero_upper_b=-0.29,
    )
    path = tmp_path / "model.json"
    save_calibrated_isotopic_model(model, path)
    payload = json.loads(path.read_text())
    payload["model"]["alpha"] = 2.0
    path.write_text(json.dumps(payload))
    assert load_calibrated_isotopic_model(path).alpha == 2.0


def test_convenience_source_resolution(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "sample.mzML"
    source.touch()
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return Path("result").resolve()

    monkeypatch.setattr(
        "hrms_utils.positive_msdial_annotation.run_positive_msdial_annotation",
        fake_run,
    )
    result = run_complete_positive_msdial_workflow(
        source,
        None,
        PositiveMsdialWorkflowParameters(
            load_model_path=tmp_path / "model.json",
            metadata_values={
                "file_type": "Sample",
                "class_id": "1",
                "acquisition_type": "Sample",
                "factor": "NA",
                "batch_order": 1,
            },
        ),
    )
    assert result == Path("result").resolve()
    assert captured["mzml_glob"] == str(source)
    assert captured["spectral_library_path"].name == "iibr_260708.parquet"


def test_package_workflow_import_does_not_import_experiments_or_pyplot() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; import hrms_utils.isotope_calibration, hrms_utils.positive_msdial_annotation; assert 'experiments' not in sys.modules; assert 'matplotlib.pyplot' not in sys.modules",
    ]
    subprocess.run(command, check=True)


def test_source_directory_resolves_to_mzml_glob(tmp_path: Path) -> None:
    assert _source_to_glob(tmp_path) == str(tmp_path / "*.mzML")


def test_installed_calibration_shortcut_requires_pair_path_for_emit_pairs(
    tmp_path: Path,
) -> None:
    import pytest

    with pytest.raises(ValueError, match="pairs_output_file is required"):
        calibrate_isotopic_parameters(
            chromatogram_dir=tmp_path,
            output_file=tmp_path / "results.txt",
            emit_pairs=True,
        )


def test_installed_calibration_shortcut_passes_explicit_pair_path(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_main(config):
        captured["config"] = config
        return "fitted"

    monkeypatch.setattr(calibration_module, "main", fake_main)
    pair_path = tmp_path / "pairs.xlsx"
    result = calibrate_isotopic_parameters(
        chromatogram_dir=tmp_path,
        output_file=tmp_path / "results.txt",
        pairs_output_file=pair_path,
        library_ion_mode="N",
    )

    assert result == "fitted"
    config = captured["config"]
    assert config.emit_pairs is True
    assert config.pairs_output_file == pair_path
    assert config.library_ion_mode == "N"


def test_legacy_isotopic_config_pair_defaults_remain_unchanged() -> None:
    config = isotopic_config()
    assert config.emit_pairs is False
    assert config.pairs_output_file == Path("experiments/isotopic_fitting_pairs.xlsx")
