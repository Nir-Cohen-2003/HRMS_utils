"""Polarity, runtime-bound controls, naming, and installed workflow contracts."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import polars as pl
import pytest

import hrms_utils.msdial_annotation as workflow
import hrms_utils.positive_msdial_annotation as positive
from hrms_utils.hrms_core import CalibratedIsotopicModel
from hrms_utils.isotope_model import (
    load_calibrated_isotopic_model,
    save_calibrated_isotopic_model,
)

METADATA = {
    "file_type": "Sample",
    "class_id": "1",
    "acquisition_type": "Sample",
    "factor": "NA",
    "batch_order": 1,
}
MODEL = CalibratedIsotopicModel(1.5, 0.98, 0.0, 1.7, -0.1, 18.0, -0.29)
ROOT = Path(__file__).parents[1]


def test_required_polarity_and_single_parameters_source() -> None:
    for function in (
        workflow.MsdialAnnotationConfig,
        workflow.run_msdial_annotation,
        workflow.run_complete_msdial_workflow,
    ):
        parameter = inspect.signature(function).parameters["polarity"]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        workflow.MsdialAnnotationConfig()
    with pytest.raises(TypeError):
        workflow.run_msdial_annotation()
    with pytest.raises(TypeError):
        workflow.run_complete_msdial_workflow(
            "data", None, workflow.MsdialWorkflowParameters()
        )
    assert (
        "polarity"
        not in inspect.signature(workflow.MsdialWorkflowParameters).parameters
    )
    for function in (
        workflow.run_complete_msdial_workflow,
        positive.run_complete_positive_msdial_workflow,
    ):
        assert (
            "minimum_element_count_error" not in inspect.signature(function).parameters
        )


@pytest.mark.parametrize("polarity", ["", "P", "Positive", None, False, "both"])
def test_invalid_runtime_polarity(polarity: Any) -> None:
    with pytest.raises(ValueError, match="polarity"):
        workflow._validate_inputs(workflow.MsdialAnnotationConfig(polarity=polarity))


@pytest.mark.parametrize("polarity,mode", [("positive", "P"), ("negative", "N")])
def test_method_resources_and_output_defaults(polarity: str, mode: str) -> None:
    path = workflow.default_method(polarity)
    workflow._validate_method_polarity(path, polarity)
    assert workflow.LIBRARY_ION_MODES[polarity] == mode
    assert workflow._output_directory(
        workflow.MsdialAnnotationConfig(polarity=polarity)
    ) == Path(f"analysis/msdial-{polarity}-annotation")
    assert path.read_bytes() == (ROOT / "experiments" / path.name).read_bytes()
    positive_bytes = workflow.default_method("positive").read_bytes()
    negative_bytes = workflow.default_method("negative").read_bytes()
    assert negative_bytes == positive_bytes.replace(
        b"Ion mode: Positive", b"Ion mode: Negative"
    ).replace(b"adduct list: [M+H]+", b"adduct list: [M-H]-")


@pytest.mark.parametrize(
    "polarity,alias",
    [("positive", "pos"), ("positive", "+"), ("negative", "neg"), ("negative", "-")],
)
def test_custom_method_validates_mode_not_adducts(
    tmp_path: Path, polarity: str, alias: str
) -> None:
    path = tmp_path / "method.txt"
    path.write_text(f"Ion mode: {alias}\nadduct list: custom\n")
    workflow._validate_method_polarity(path, polarity)
    for text in (
        "",
        "Ion mode: unknown\n",
        "Ion mode: Positive\nIon mode: Negative\n",
        f"Ion mode: {alias}\nIon mode: {alias}\n",
    ):
        path.write_text(text)
        with pytest.raises(ValueError, match="exactly one"):
            workflow._validate_method_polarity(path, polarity)
    path.write_text(f"Ion mode: {alias}\n")
    with pytest.raises(ValueError, match="exactly one"):
        workflow._validate_method_polarity(
            path, "negative" if polarity == "positive" else "positive"
        )


@pytest.mark.parametrize(
    "scale", [0.0, -1, True, "2", float("nan"), float("inf"), -float("inf")]
)
def test_invalid_scale_before_resources(scale: Any) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        workflow._validate_inputs(
            workflow.MsdialAnnotationConfig(
                polarity="positive", calibrated_error_bound_scale=scale
            )
        )


@pytest.mark.parametrize("scale", [1.0, 2.0, 0.5])
def test_scale_changes_only_log_width_amplitudes(scale: float) -> None:
    effective = workflow._annotation_model(MODEL, scale)
    assert effective is not MODEL
    assert effective == replace(
        MODEL,
        hetero_lower_a=MODEL.hetero_lower_a * scale,
        hetero_upper_a=MODEL.hetero_upper_a * scale,
    )
    assert MODEL.hetero_upper_a == 18.0


def test_scale_overflow_is_loud() -> None:
    with pytest.raises(ValueError, match="overflows"):
        workflow._annotation_model(MODEL, 1e308)


def _resources(tmp_path: Path, polarity: str) -> workflow.MsdialAnnotationConfig:
    source = tmp_path / "sample with space.mzML"
    source.touch()
    executable = tmp_path / "MSDIALCUI"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    for suffix in (".dll", ".runtimeconfig.json", ".deps.json"):
        executable.with_suffix(suffix).touch()
    library = tmp_path / "library.parquet"
    pl.DataFrame(
        {
            "ion_mode": ["P", "N"],
            "precursor_mz": [100.0, 100.0],
            "cleaned_normalized_mz": [[50.0]] * 2,
            "cleaned_normalized_intensity": [[1.0]] * 2,
            "precursor_formula_array": [[0] * 12] * 2,
            "name": ["pos", "neg"],
            "inchikey": ["P", "N"],
            "smiles": ["C", "C"],
        }
    ).write_parquet(library)
    return workflow.MsdialAnnotationConfig(
        polarity=polarity,
        mzml_glob=str(source),
        msdial_path=executable,
        spectral_library_path=library,
        output_directory=tmp_path / "output",
        metadata_values=METADATA,
    )


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    *,
    duplicate: bool = False,
) -> None:
    def run_msdial(
        inputs: workflow.ValidatedInputs, metadata: Path, output: Path
    ) -> None:
        captured["msdial_calls"] = captured.get("msdial_calls", 0) + 1
        captured["method"] = inputs.method_path
        output.mkdir()
        (output / "sample with space.mdpeak").write_text(
            "\t".join(workflow.MSDIAL_columns_to_read) + "\n"
        )
        if duplicate:
            (output / "nested").mkdir()
            (output / "nested/SAMPLE WITH SPACE.mdpeak").write_text(
                "\t".join(workflow.MSDIAL_columns_to_read) + "\n"
            )

    def calibrate(config: Any) -> CalibratedIsotopicModel:
        captured["calibration"] = config
        config.output_file.write_text("Calibration hits total (before alignment): 1\n")
        return MODEL

    def annotate(frame: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        captured["annotation"] = kwargs
        return frame.with_columns(pl.lit("C1").alias("precursor_formula_str"))

    frame = pl.DataFrame(
        {
            "Isotope": [0],
            "ms1_isotopes_m/z": [[100.0]],
            "ms1_isotopes_intensity": [[1e6]],
            "msms_m/z": [[50.0]],
            "msms_intensity": [[1.0]],
        }
    )
    monkeypatch.setattr(workflow, "_run_msdial", run_msdial)
    monkeypatch.setattr(workflow, "get_chromatogram", lambda path: frame)
    monkeypatch.setattr(workflow, "calibrate_isotopic_parameters", calibrate)
    monkeypatch.setattr(workflow, "annotate_chromatogram_with_formulas", annotate)


@pytest.mark.parametrize("polarity,mode", [("positive", "P"), ("negative", "N")])
@pytest.mark.parametrize("loaded", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.0, 0.5])
def test_fit_load_polarity_scale_maps_manifest_and_stems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    polarity: str,
    mode: str,
    loaded: bool,
    scale: float,
) -> None:
    config = _resources(tmp_path, polarity)
    captured: dict[str, Any] = {}
    _patch_pipeline(monkeypatch, captured)
    config = replace(
        config,
        calibrated_error_bound_scale=scale,
        minimum_element_count_error={"C": 2},
        maximum_element_count_error={"C": 2},
    )
    model_path = tmp_path / "model.json"
    if loaded:
        save_calibrated_isotopic_model(MODEL, model_path)
        before_bytes = model_path.read_bytes()
        config = replace(
            config,
            load_model_path=model_path,
            spectral_library_path=tmp_path / "absent.parquet",
        )

        def no_library(*args: Any) -> None:
            pytest.fail("Loaded mode accessed library")

        monkeypatch.setattr(workflow, "_validate_library", no_library)
    else:
        config = replace(config, save_model_path=model_path)
    output = workflow.main(config)
    assert captured["msdial_calls"] == 1
    assert captured["method"] == workflow.default_method(polarity)
    assert ("calibration" in captured) is not loaded
    if not loaded:
        assert captured["calibration"].library_ion_mode == mode
        assert (
            load_calibrated_isotopic_model(output / "calibration/model.json") == MODEL
        )
        assert load_calibrated_isotopic_model(model_path) == MODEL
    else:
        assert model_path.read_bytes() == before_bytes
        assert not (output / "calibration").exists()
        assert not (output / "calibration-input").exists()
    assert captured["annotation"]["isotopic_model"] == workflow._annotation_model(
        MODEL, scale
    )
    assert captured["annotation"]["minimum_element_count_error"] == {"C": 2}
    assert captured["annotation"]["maximum_element_count_error"] == {"C": 2}
    assert [
        captured["annotation"][key]
        for key in (
            "precursor_mass_accuracy_ppm",
            "fragment_mass_accuracy_ppm",
            "normalized_fragment_mass_accuracy_ppm",
        )
    ] == [3.0, 5.0, 5.0]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["polarity"] == polarity
    assert manifest["config"]["calibrated_error_bound_scale"] == scale
    assert manifest["config"]["minimum_element_count_error"] == {"C": 2}
    assert manifest["model"]["parameters"] == asdict(MODEL)
    assert manifest["model"]["annotation_parameters"] == asdict(
        workflow._annotation_model(MODEL, scale)
    )
    assert (
        "NOT_APPLIED"
        in manifest["element_count_error_policy"]["without_calibrated_center"]
    )
    assert set(json.loads(model_path.read_text())["model"]) == set(asdict(MODEL))
    mapping = manifest["output_mappings"][0]
    assert mapping["annotation"] == "annotations/sample with space.parquet"
    assert mapping["summary"] == "summaries/sample with space.csv"
    for value in mapping.values():
        if value is not None:
            assert (output / value).exists()
    assert (mapping["calibration_input_mdpeak"] is None) is loaded


def test_duplicate_stems_fail_before_calibration_or_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _resources(tmp_path, "negative")
    captured: dict[str, Any] = {}
    _patch_pipeline(monkeypatch, captured, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate mdpeak stems"):
        workflow.main(config)
    assert "calibration" not in captured and "annotation" not in captured
    assert not config.output_directory.exists()
    stage = next(tmp_path.glob(".output.staging-*"))
    assert not (stage / "calibration-input").exists()
    assert not (stage / "annotations").exists()


def test_library_preflight_requires_selected_mode(tmp_path: Path) -> None:
    config = _resources(tmp_path, "negative")
    library = pl.read_parquet(config.spectral_library_path).filter(
        pl.col("ion_mode") == "P"
    )
    library.write_parquet(config.spectral_library_path)
    workflow._validate_library(config.spectral_library_path, "positive")
    with pytest.raises(ValueError, match="no negative-mode"):
        workflow._validate_inputs(config)


def test_cli_required_polarity_json_maps_and_legacy_options() -> None:
    parser = workflow.build_argument_parser()
    arguments = ["--metadata-values", json.dumps(METADATA)]
    with pytest.raises(SystemExit):
        parser.parse_args(arguments)
    parsed = parser.parse_args(
        arguments
        + [
            "--polarity",
            "negative",
            "--minimum-element-count-error",
            '{"C":2}',
            "--maximum-element-count-error",
            '{"C":2}',
            "--calibrated-error-bound-scale",
            "2",
        ]
    )
    config = workflow.configuration_from_arguments(parsed)
    assert config.polarity == "negative" and config.msdial_method is None
    assert (
        config.minimum_element_count_error
        == config.maximum_element_count_error
        == {"C": 2}
    )
    legacy_parser = positive.build_argument_parser()
    generic_options = set(parser._option_string_actions)
    legacy_options = set(legacy_parser._option_string_actions)
    assert generic_options - legacy_options == {"--polarity"}
    assert not legacy_options - generic_options
    assert (
        positive.configuration_from_arguments(
            legacy_parser.parse_args(arguments)
        ).msdial_method
        == positive.DEFAULT_METHOD
    )
    for value in (
        "null",
        "[]",
        '{"C":true}',
        '{"C":2.0}',
        '{"c":2}',
        '{"C":-1}',
        '{"C":2147483648}',
        "{bad",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                arguments
                + ["--polarity", "positive", "--minimum-element-count-error", value]
            )


def test_complete_wrappers_forward_parameter_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def run(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return Path("result")

    for module, parameter_class, function, shortcut in (
        (
            workflow,
            workflow.MsdialWorkflowParameters,
            workflow.run_complete_msdial_workflow,
            "run_msdial_annotation",
        ),
        (
            positive,
            positive.PositiveMsdialWorkflowParameters,
            positive.run_complete_positive_msdial_workflow,
            "run_positive_msdial_annotation",
        ),
    ):
        monkeypatch.setattr(module, shortcut, run)
        parameters = parameter_class(
            load_model_path=Path("model.json"),
            calibrated_error_bound_scale=2,
            minimum_element_count_error={"C": 3},
            maximum_element_count_error={"S": 1},
        )
        keyword = {"polarity": "negative"} if module is workflow else {}
        assert function("*.mzML", None, parameters, **keyword) == Path("result")
        assert captured["minimum_element_count_error"] == {"C": 3}
        assert captured["maximum_element_count_error"] == {"S": 1}
        assert captured["calibrated_error_bound_scale"] == 2


def test_positive_adapter_preserves_mapping_and_positional_compatibility() -> None:
    metadata = MappingProxyType(METADATA)
    margins = MappingProxyType({"C": 2})
    config = positive.PositiveMsdialAnnotationConfig(
        "*.mzML",
        Path("MSDIALCUI"),
        Path("library.parquet"),
        Path("output"),
        None,
        metadata,
        ("C",),
        False,
        None,
        None,
        minimum_element_count_error=margins,
    )
    generic = positive._generic_config(config)
    assert generic.polarity == "positive"
    assert generic.metadata_values is metadata
    assert generic.minimum_element_count_error is margins
    assert generic.calibrated_error_bound_scale == 1.0


def test_positive_dependency_patches_do_not_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = workflow._workflow_dependencies()
    monkeypatch.setattr(positive, "_run_msdial", lambda *args: None)
    assert positive._workflow_dependencies().run_msdial is not original.run_msdial
    assert workflow._workflow_dependencies() == original


def test_installed_exports_resources_and_commands_outside_checkout(
    tmp_path: Path,
) -> None:
    script = """
import sys
from importlib.resources import files
import hrms_utils
assert 'experiments' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
for name in ('MsdialAnnotationConfig', 'MsdialWorkflowParameters', 'run_msdial_annotation', 'run_complete_msdial_workflow', 'run_positive_msdial_annotation'):
    assert hasattr(hrms_utils, name)
for polarity in ('positive', 'negative'):
    assert files('hrms_utils').joinpath(f'msdial_console_params_{polarity}.txt').is_file()
print(hrms_utils.__file__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "site-packages" in result.stdout
    for command in (
        "hrms-msdial-annotate",
        "hrms-positive-msdial-annotate",
        "hrms-isotope-calibrate",
    ):
        subprocess.run(
            [str(Path(sys.executable).with_name(command)), "--help"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
        )
