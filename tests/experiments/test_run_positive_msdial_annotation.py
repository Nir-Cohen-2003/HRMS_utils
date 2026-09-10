"""Focused tests for the positive-mode orchestration boundaries."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest


_REPOSITORY_ROOT = Path(__file__).parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_MODULE_PATH = _REPOSITORY_ROOT / "experiments" / "run_positive_msdial_annotation.py"
_MODULE_SPEC = importlib.util.spec_from_file_location("positive_annotation", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
positive_annotation = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = positive_annotation
_MODULE_SPEC.loader.exec_module(positive_annotation)


VALID_METADATA = {
    "file_type": "Sample",
    "class_id": "1",
    "acquisition_type": "Sample",
    "factor": "NA",
    "batch_order": 1,
}


def test_metadata_values_are_strict_and_safe() -> None:
    assert positive_annotation._validate_metadata_values(VALID_METADATA) == VALID_METADATA
    with pytest.raises(ValueError, match="exactly"):
        positive_annotation._validate_metadata_values({**VALID_METADATA, "unknown": "x"})
    with pytest.raises(ValueError, match="control character"):
        positive_annotation._validate_metadata_values({**VALID_METADATA, "factor": "x\n"})
    with pytest.raises(ValueError, match="positive integer"):
        positive_annotation._validate_metadata_values({**VALID_METADATA, "batch_order": True})


def test_metadata_csv_is_absolute_sorted_and_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "b.mzML"
    second = tmp_path / "a.mzML"
    first.touch()
    second.touch()
    destination = tmp_path / "metadata.csv"
    paths = positive_annotation._find_mzml_paths(str(tmp_path / "*.mzML"))
    positive_annotation._write_metadata_csv(paths, VALID_METADATA, destination)
    rows = list(__import__("csv").reader(destination.open(encoding="utf-8")))
    assert rows[0] == list(positive_annotation.METADATA_COLUMNS)
    assert rows[1][0] == str(second.resolve())
    assert rows[2][0] == str(first.resolve())
    assert rows[1][5:7] == ["1", "1"]


def test_positive_default_method_is_positive_and_m_h_only() -> None:
    method = _REPOSITORY_ROOT / positive_annotation.DEFAULT_METHOD
    text = method.read_text(encoding="utf-8")
    assert "Ion mode: Positive" in text
    assert "adduct list: [M+H]+" in text
    assert "[M-H]-" not in text


def test_cli_uses_positive_method_by_default_and_allows_override(tmp_path: Path) -> None:
    parser = positive_annotation.build_argument_parser()
    arguments = parser.parse_args(["--metadata-values", json.dumps(VALID_METADATA)])
    assert arguments.msdial_method == positive_annotation.DEFAULT_METHOD
    override = tmp_path / "validated-method.txt"
    overridden = parser.parse_args(
        [
            "--metadata-values",
            json.dumps(VALID_METADATA),
            "--msdial-method",
            str(override),
        ]
    )
    assert overridden.msdial_method == override


def test_positive_method_requires_one_positive_setting(tmp_path: Path) -> None:
    method = tmp_path / "method.txt"
    method.write_text("Ion mode: Positive\nMS1 data type: Centroid\n", encoding="utf-8")
    positive_annotation._method_is_positive(method)
    method.write_text("Ion mode: Negative\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive ion-mode"):
        positive_annotation._method_is_positive(method)


def test_model_mapping_uses_named_fields_and_native_scalars() -> None:
    fitted = SimpleNamespace(
        alpha=np.float64(1.0),
        beta=np.float64(2.0),
        offset=np.float64(3.0),
        hetero_lower_a=np.float64(4.0),
        hetero_lower_b=np.float64(5.0),
        hetero_upper_a=np.float64(6.0),
        hetero_upper_b=np.float64(7.0),
    )
    model = positive_annotation._model_from_fit(fitted)
    assert model.alpha == 1.0
    assert model.beta == 2.0
    assert model.hetero_lower_a == 4.0
    assert model.hetero_upper_b == 7.0
    assert all(type(value) is float for value in vars(model).values())


def test_local_msdial_is_executable_without_chmod() -> None:
    executable = _REPOSITORY_ROOT / "MSDIAL" / "MSDIALCUI"
    if not executable.exists():
        pytest.skip("local MS-DIAL bundle is not part of the repository checkout")
    assert executable.is_file()
    assert os.access(executable, os.X_OK)


def test_mzml_glob_rejects_duplicate_stems(tmp_path: Path) -> None:
    (tmp_path / "a.mzML").touch()
    (tmp_path / "A.mzML").touch()
    with pytest.raises(ValueError, match="Duplicate mzML stems"):
        positive_annotation._find_mzml_paths(str(tmp_path / "*.mzML"))


def test_annotation_tolerances_are_explicit_in_source() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "precursor_mass_accuracy_ppm=3.0" in source
    assert "fragment_mass_accuracy_ppm=5.0" in source
    assert "normalized_fragment_mass_accuracy_ppm=5.0" in source


def test_main_publishes_layout_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "sample.mzML").touch()
    msdial = tmp_path / "MSDIALCUI"
    msdial.write_text("#!/bin/sh\n", encoding="utf-8")
    msdial.chmod(0o755)
    for suffix in (".dll", ".runtimeconfig.json", ".deps.json"):
        msdial.with_suffix(suffix).touch()
    method = tmp_path / "positive-method.txt"
    method.write_text("Ion mode: Positive\n", encoding="utf-8")
    library = tmp_path / "library.parquet"
    pl.DataFrame(
        {
            "ion_mode": ["P"],
            "precursor_mz": [100.0],
            "cleaned_normalized_mz": [[50.0]],
            "cleaned_normalized_intensity": [[1.0]],
            "precursor_formula_array": [[0] * 12],
            "name": ["compound"],
            "inchikey": ["key"],
            "smiles": ["C"],
        }
    ).write_parquet(library)
    header = "\t".join(positive_annotation.MSDIAL_columns_to_read) + "\n"
    input_frame = pl.DataFrame(
        {
            "Isotope": [0],
            "ms1_isotopes_m/z": [[100.0]],
            "ms1_isotopes_intensity": [[100.0]],
            "msms_m/z": [[50.0]],
            "msms_intensity": [[1.0]],
        }
    )
    monkeypatch.setattr(
        positive_annotation,
        "_run_msdial",
        lambda inputs, metadata, output: (
            output.mkdir(),
            (output / "sample.mdpeak").write_text(header, encoding="utf-8"),
            SimpleNamespace(returncode=0),
        )[-1],
    )
    monkeypatch.setattr(positive_annotation, "get_chromatogram", lambda path: input_frame)
    monkeypatch.setattr(
        positive_annotation,
        "calibrate_isotopic_parameters",
        lambda config: (
            config.output_file.write_text("Calibration hits total (before alignment): 1\n", encoding="utf-8"),
            positive_annotation.IsotopicToleranceModel(
                alpha=1,
                beta=1,
                offset=0,
                band_lower_width=1,
                band_upper_width=1,
                hetero_lower_a=1,
                hetero_lower_b=1,
                hetero_upper_a=1,
                hetero_upper_b=1,
                success_rate=1,
            ),
        )[1],
    )
    monkeypatch.setattr(
        positive_annotation,
        "annotate_chromatogram_with_formulas",
        lambda dataframe, **kwargs: dataframe.with_columns(
            pl.lit("C1").alias("precursor_formula_str"),
            pl.lit(0.0).alias("precursor_errors_ppm"),
            pl.lit(1.0).alias("explained_intensity"),
        ),
    )
    output = tmp_path / "published"
    result = positive_annotation.main(
        positive_annotation.PositiveMsdialAnnotationConfig(
            mzml_glob=str(analysis / "*.mzML"),
            msdial_path=msdial,
            spectral_library_path=library,
            output_directory=output,
            msdial_method=method,
            metadata_values=VALID_METADATA,
        )
    )
    assert result == output.resolve()
    assert (output / "annotations/sample.parquet").exists()
    assert (output / "summaries/sample.csv").exists()
    assert (output / "manifest.json").exists()
    assert (output / "SCHEMA.md").exists()
    assert (output / "calibration/model.json").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["thresholds"]["annotation_fragment_ppm"] == 5.0
    assert manifest["model"]["source"] == "fitted"
    assert manifest["model"]["path"] == "calibration/model.json"
