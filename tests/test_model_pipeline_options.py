import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from hrms_utils.hrms_core import CalibratedIsotopicModel
from hrms_utils.isotope_model import save_calibrated_isotopic_model


ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "experiments" / "run_positive_msdial_annotation.py"
spec = importlib.util.spec_from_file_location(
    "positive_annotation_model_test", MODULE_PATH
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

METADATA = {
    "file_type": "Sample",
    "class_id": "1",
    "acquisition_type": "Sample",
    "factor": "NA",
    "batch_order": 1,
}
MODEL = CalibratedIsotopicModel(1.5, 0.98, 0.0, 1.7, -0.1, 18.0, -0.29)


def test_pipeline_load_model_skips_library_and_calibration(
    tmp_path: Path, monkeypatch
) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "sample.mzML").touch()
    executable = tmp_path / "MSDIALCUI"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    for suffix in (".dll", ".runtimeconfig.json", ".deps.json"):
        executable.with_suffix(suffix).touch()
    method = tmp_path / "method.txt"
    method.write_text("Ion mode: Positive\n", encoding="utf-8")
    model_path = tmp_path / "model.json"
    save_calibrated_isotopic_model(MODEL, model_path)
    header = "\t".join(module.MSDIAL_columns_to_read) + "\n"
    frame = pl.DataFrame(
        {
            "Isotope": [0],
            "ms1_isotopes_m/z": [[100.0]],
            "ms1_isotopes_intensity": [[100.0]],
            "msms_m/z": [[50.0]],
            "msms_intensity": [[1.0]],
        }
    )
    monkeypatch.setattr(
        module,
        "_run_msdial",
        lambda inputs, metadata, output: (
            output.mkdir(),
            (output / "sample.mdpeak").write_text(header, encoding="utf-8"),
            SimpleNamespace(returncode=0),
        )[-1],
    )
    monkeypatch.setattr(module, "get_chromatogram", lambda path: frame)
    monkeypatch.setattr(
        module,
        "_validate_library",
        lambda path: (_ for _ in ()).throw(AssertionError("library checked")),
    )
    monkeypatch.setattr(
        module,
        "calibrate_isotopic_parameters",
        lambda config: (_ for _ in ()).throw(AssertionError("calibrated")),
    )
    monkeypatch.setattr(
        module,
        "annotate_chromatogram_with_formulas",
        lambda dataframe, **kwargs: dataframe.with_columns(
            pl.lit("C1").alias("precursor_formula_str"),
            pl.lit(0.0).alias("precursor_errors_ppm"),
            pl.lit(1.0).alias("explained_intensity"),
        ),
    )
    output = module.main(
        module.PositiveMsdialAnnotationConfig(
            mzml_glob=str(analysis / "*.mzML"),
            msdial_path=executable,
            spectral_library_path=tmp_path / "missing.parquet",
            output_directory=tmp_path / "output",
            msdial_method=method,
            metadata_values=METADATA,
            load_model_path=model_path,
        )
    )
    assert not (output / "calibration").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["mode"] == "loaded"
    assert manifest["model"]["source"] == "loaded"
    assert manifest["model"]["parameters"]["alpha"] == 1.5


def test_pipeline_rejects_simultaneous_model_save_and_load() -> None:
    parser = module.build_argument_parser()
    arguments = parser.parse_args(
        [
            "--metadata-values",
            json.dumps(METADATA),
            "--load-model",
            "load.json",
            "--save-model",
            "save.json",
        ]
    )
    config = module.configuration_from_arguments(arguments)
    import pytest

    with pytest.raises(ValueError, match="cannot be used together"):
        module._validate_inputs(config)
