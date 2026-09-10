"""Focused validation tests for the isotopic calibration command."""

import importlib.util
from pathlib import Path

import polars as pl
import pytest

from hrms_utils.isotope_model import load_calibrated_isotopic_model


_MODULE_PATH = (
    Path(__file__).parents[2] / "experiments" / "find_optimal_isotopic_parameters.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "isotopic_calibration", _MODULE_PATH
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
isotopic_calibration = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(isotopic_calibration)


def _configuration(
    temporary_path: Path,
    *,
    use_library: bool = False,
    use_csv: bool = False,
    emit_pairs: bool = False,
    pairs_output_file: Path | None = None,
) -> isotopic_calibration.isotopic_config:
    chromatogram_dir = temporary_path / "chromatograms"
    chromatogram_dir.mkdir(exist_ok=True)
    (chromatogram_dir / "sample.mdpeak").touch()

    library_path = temporary_path / "library.parquet"
    library_path.touch()
    csv_path = temporary_path / "ground_truth.csv"
    csv_path.write_text("Molecular Formula,Monoisotopic Mass\nC6H12O6,180.0634\n")

    return isotopic_calibration.isotopic_config(
        chromatogram_dir=chromatogram_dir,
        output_file=temporary_path / "results.txt",
        use_library_search_for_ground_truth=use_library,
        use_mass_list_for_ground_truth=use_csv,
        spectral_library_paths=(library_path,) if use_library else (),
        ground_truth_csv_path=csv_path if use_csv else None,
        emit_pairs=emit_pairs,
        pairs_output_file=pairs_output_file,
    )


@pytest.mark.parametrize(
    ("use_library", "use_csv"),
    [(True, False), (False, True), (True, True)],
)
def test_validate_configuration_accepts_ground_truth_sources(
    tmp_path: Path, use_library: bool, use_csv: bool
) -> None:
    config = _configuration(tmp_path, use_library=use_library, use_csv=use_csv)

    matched_paths = isotopic_calibration._validate_configuration(config)

    assert matched_paths == [tmp_path / "chromatograms" / "sample.mdpeak"]


def test_validate_configuration_rejects_missing_ground_truth_source(
    tmp_path: Path,
) -> None:
    config = _configuration(tmp_path)

    with pytest.raises(ValueError, match="At least one ground-truth source"):
        isotopic_calibration._validate_configuration(config)


@pytest.mark.parametrize(
    "pair_configuration",
    [
        {"emit_pairs": True},
        {"pairs_output_file": Path("pairs.xlsx")},
    ],
)
def test_validate_configuration_rejects_pairs_for_csv_ground_truth(
    tmp_path: Path, pair_configuration: dict[str, object]
) -> None:
    pairs_output_file = pair_configuration.get("pairs_output_file")
    config = _configuration(
        tmp_path,
        use_csv=True,
        emit_pairs=bool(pair_configuration.get("emit_pairs", False)),
        pairs_output_file=(
            tmp_path / pairs_output_file
            if isinstance(pairs_output_file, Path)
            else None
        ),
    )

    with pytest.raises(ValueError, match="only available with --spectral-library"):
        isotopic_calibration._validate_configuration(config)


def _fit_fixture(formula_dtype: pl.DataType = pl.List(pl.Int64)) -> pl.DataFrame:
    formula = [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return pl.DataFrame(
        [
            {
                "Peak ID": index,
                "source_file": "sample.mdpeak",
                "Precursor_mz_MSDIAL": 100.0,
                "ms1_isotopes_m/z": [100.0, 101.003355],
                "ms1_isotopes_intensity": [1_000_000.0, 100_000.0 + index * 1_000.0],
                "formula_array": formula,
            }
            for index in range(6)
        ],
        schema={
            "Peak ID": pl.Int64,
            "source_file": pl.String,
            "Precursor_mz_MSDIAL": pl.Float64,
            "ms1_isotopes_m/z": pl.List(pl.Float64),
            "ms1_isotopes_intensity": pl.List(pl.Float64),
            "formula_array": formula_dtype,
        },
    )


def test_configuration_from_arguments_converts_cli_values() -> None:
    parser = isotopic_calibration.build_argument_parser()
    arguments = parser.parse_args(
        [
            "--chromatogram-dir",
            "/tmp/chromatograms",
            "--output-file",
            "/tmp/calibration/results.txt",
            "--ground-truth-csv",
            "/tmp/ground-truth.csv",
            "--elements",
            "C",
            "Cl",
            "--save-model",
            "/tmp/calibration/model.json",
        ]
    )

    config = isotopic_calibration.configuration_from_arguments(arguments)

    assert config.chromatogram_dir == Path("/tmp/chromatograms")
    assert config.output_file == Path("/tmp/calibration/results.txt")
    assert config.use_mass_list_for_ground_truth is True
    assert config.use_library_search_for_ground_truth is False
    assert config.use_carbon is True
    assert config.use_chlorine is True
    assert config.save_model_path == Path("/tmp/calibration/model.json")


@pytest.mark.parametrize("formula_dtype", [pl.List(pl.Int64), pl.Array(pl.Int64, 12)])
def test_fit_accepts_list_and_array_formula_storage(
    tmp_path: Path, formula_dtype: pl.DataType
) -> None:
    model = isotopic_calibration.fit_isotopic_tolerance_parameters(
        _fit_fixture(formula_dtype),
        tmp_path / "array-list-results.txt",
        formula_array_column="formula_array",
        minimum_isotopic_peak_intensity=1.0,
        plot_percentile_relative_error=False,
        plot_percentile_carbon_count_error=False,
    )

    assert model.alpha > 0.0
    assert (tmp_path / "array-list-results.txt").exists()


def test_fit_filters_only_library_rows_missing_inchikey(tmp_path: Path) -> None:
    fixture = _fit_fixture().with_columns(
        pl.Series(
            "is_library_ground_truth",
            [True, False, False, False, False, False],
            dtype=pl.Boolean,
        ),
        pl.Series("inchikey", [None, None, None, None, None, None], dtype=pl.String),
        pl.Series("true_carbon_count", [10, 10, 10, 10, 10, 10], dtype=pl.Int64),
        pl.Series("true_sulfur_count", [0] * 6, dtype=pl.Int64),
        pl.Series("true_chlorine_count", [0] * 6, dtype=pl.Int64),
        pl.Series("true_bromine_count", [0] * 6, dtype=pl.Int64),
    )

    model = isotopic_calibration.fit_isotopic_tolerance_parameters(
        fixture,
        tmp_path / "provenance-results.txt",
        provenance_column="is_library_ground_truth",
        minimum_isotopic_peak_intensity=1.0,
        plot_percentile_relative_error=False,
        plot_percentile_carbon_count_error=False,
    )

    assert model.success_rate > 0.0


def _main_chromatogram_fixture() -> pl.DataFrame:
    """Return one small MS-DIAL-like feature for main() orchestration tests."""
    return pl.DataFrame(
        {
            "Peak ID": [1],
            "RT (min)": [1.0],
            "Precursor_mz_MSDIAL": [100.0],
            "Height": [1_000_000.0],
            "Isotope": [0],
            "ms1_isotopes_m/z": [[100.0, 101.003355]],
            "ms1_isotopes_intensity": [[1_000_000.0, 100_000.0]],
            "msms_m/z": [[50.0, 75.0]],
            "msms_intensity": [[10.0, 5.0]],
        },
        schema={
            "Peak ID": pl.Int64,
            "RT (min)": pl.Float64,
            "Precursor_mz_MSDIAL": pl.Float64,
            "Height": pl.Float64,
            "Isotope": pl.Int64,
            "ms1_isotopes_m/z": pl.List(pl.Float64),
            "ms1_isotopes_intensity": pl.List(pl.Float64),
            "msms_m/z": pl.List(pl.Float64),
            "msms_intensity": pl.List(pl.Float64),
        },
    )


def _main_library_fixture() -> pl.DataFrame:
    formula = [0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return pl.DataFrame(
        {
            "ion_mode": ["P"],
            "precursor_mz": [100.0],
            "cleaned_normalized_mz": [[50.0, 75.0]],
            "cleaned_normalized_intensity": [[10.0, 5.0]],
            "precursor_formula_array": [formula],
            "name": ["fixture compound"],
            "inchikey": ["FIXTURE-INCHIKEY"],
            "smiles": ["CC"],
        },
        schema={
            "ion_mode": pl.String,
            "precursor_mz": pl.Float64,
            "cleaned_normalized_mz": pl.List(pl.Float64),
            "cleaned_normalized_intensity": pl.List(pl.Float64),
            "precursor_formula_array": pl.Array(pl.Int64, 12),
            "name": pl.String,
            "inchikey": pl.String,
            "smiles": pl.String,
        },
    )


class _MainFakeMassDecomposition:
    """Replace expensive Rust expression evaluation while retaining main()'s lazy graph."""

    def clean_and_normalize_spectrum(
        self, *_args: object, **_kwargs: object
    ) -> pl.Expr:
        return pl.struct(
            pl.lit([50.0, 75.0]).alias("normalized_masses"),
            pl.lit([10.0, 5.0]).alias("intensities"),
            pl.lit([[0, 1]]).alias("formulas"),
        )


class _MainFakeSpectralInfo:
    def spectral_info_score(self, *_args: object, **_kwargs: object) -> pl.Expr:
        return pl.lit(1.0)


class _MainFakeSpectralSimilarity:
    def dotprod_similarity(self, *_args: object, **_kwargs: object) -> pl.Expr:
        return pl.lit(1.0)


def _run_main_with_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    use_library: bool,
    use_csv: bool,
    save_model: bool = False,
    library_ion_mode: str = "P",
    mixed_polarities: bool = False,
) -> pl.DataFrame:
    chromatogram_path = tmp_path / "chromatograms" / "sample.mdpeak"
    chromatogram_path.parent.mkdir()
    chromatogram_path.touch()
    library_path = tmp_path / "library.parquet"
    library_path.touch()
    csv_path = tmp_path / "ground_truth.csv"
    csv_path.write_text("Molecular Formula,Monoisotopic Mass\nC6H12O6,100.0\n")

    captured: dict[str, object] = {}

    def fake_fit(library_hits: pl.DataFrame, **_kwargs: object) -> object:
        captured["optimization_data"] = library_hits
        if save_model:
            return isotopic_calibration.IsotopicToleranceModel(
                alpha=1.0,
                beta=1.0,
                offset=0.0,
                band_lower_width=1.0,
                band_upper_width=1.0,
                hetero_lower_a=1.0,
                hetero_lower_b=1.0,
                hetero_upper_a=1.0,
                hetero_upper_b=1.0,
                success_rate=1.0,
            )
        return "fixture model"

    monkeypatch.setattr(
        isotopic_calibration,
        "get_chromatogram",
        lambda _path: _main_chromatogram_fixture(),
    )
    monkeypatch.setattr(
        isotopic_calibration.pl,
        "scan_parquet",
        lambda _path: (
            pl.concat([
                _main_library_fixture(),
                _main_library_fixture().with_columns(pl.lit("N").alias("ion_mode"), pl.lit("NEGATIVE-INCHIKEY").alias("inchikey")),
            ]) if mixed_polarities else _main_library_fixture()
        ).lazy(),
    )
    monkeypatch.setattr(
        isotopic_calibration.pl.Expr,
        "mass_decomposition",
        _MainFakeMassDecomposition(),
    )
    monkeypatch.setattr(
        isotopic_calibration.pl.Expr,
        "spectral_info",
        _MainFakeSpectralInfo(),
    )
    monkeypatch.setattr(
        isotopic_calibration.pl.Expr,
        "spectral_similarity",
        _MainFakeSpectralSimilarity(),
    )
    monkeypatch.setattr(
        isotopic_calibration, "fit_isotopic_tolerance_parameters", fake_fit
    )

    config = isotopic_calibration.isotopic_config(
        chromatogram_dir=chromatogram_path.parent,
        output_file=tmp_path / "results.txt",
        use_library_search_for_ground_truth=use_library,
        use_mass_list_for_ground_truth=use_csv,
        spectral_library_paths=(library_path,) if use_library else (),
        ground_truth_csv_path=csv_path if use_csv else None,
        save_model_path=tmp_path / "saved-model.json" if save_model else None,
        library_ion_mode=library_ion_mode,
        info_score_threshold=0.0,
        dot_product_threshold=0.0,
    )

    result = isotopic_calibration.main(config)
    if save_model:
        assert isinstance(result, isotopic_calibration.IsotopicToleranceModel)
        assert (tmp_path / "saved-model.json").exists()
        assert (
            load_calibrated_isotopic_model(tmp_path / "saved-model.json").alpha == 1.0
        )
    else:
        assert result == "fixture model"
    optimization_data = captured["optimization_data"]
    assert isinstance(optimization_data, pl.DataFrame)
    return optimization_data


def test_main_can_save_runtime_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run_main_with_fixtures(
        monkeypatch, tmp_path, use_library=True, use_csv=False, save_model=True
    )


def test_main_library_only_passes_library_provenance_to_fitting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    optimization_data = _run_main_with_fixtures(
        monkeypatch, tmp_path, use_library=True, use_csv=False
    )

    assert optimization_data["is_library_ground_truth"].to_list() == [True]
    assert optimization_data["inchikey"].to_list() == ["FIXTURE-INCHIKEY"]
    assert optimization_data.schema["formula_array"] == pl.List(pl.Int64)


def test_main_csv_only_passes_csv_provenance_to_fitting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    optimization_data = _run_main_with_fixtures(
        monkeypatch, tmp_path, use_library=False, use_csv=True
    )

    assert optimization_data["is_library_ground_truth"].to_list() == [False]
    assert optimization_data["true_carbon_count"].to_list() == [6]
    assert optimization_data["inchikey"].to_list() == [None]
    assert optimization_data.schema["formula_array"] == pl.List(pl.Int64)


def test_main_mixed_sources_concatenate_shared_schema_and_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    optimization_data = _run_main_with_fixtures(
        monkeypatch, tmp_path, use_library=True, use_csv=True
    )

    assert optimization_data.height == 2
    assert optimization_data["is_library_ground_truth"].to_list().count(True) == 1
    assert optimization_data["is_library_ground_truth"].to_list().count(False) == 1
    assert optimization_data.schema["formula_array"] == pl.List(pl.Int64)
    csv_row = optimization_data.filter(~pl.col("is_library_ground_truth"))
    assert csv_row["name"].to_list() == [None]
    assert csv_row["true_carbon_count"].to_list() == [6]


@pytest.mark.parametrize("mode,key", [("P", "FIXTURE-INCHIKEY"), ("N", "NEGATIVE-INCHIKEY")])
def test_main_calibration_filters_mixed_library_polarities(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, key: str) -> None:
    data = _run_main_with_fixtures(monkeypatch, tmp_path, use_library=True, use_csv=False, library_ion_mode=mode, mixed_polarities=True)
    assert data["inchikey"].to_list() == [key]


def test_standalone_library_mode_cli_default_and_override() -> None:
    parser = isotopic_calibration.build_argument_parser()
    args = ["--chromatogram-dir", "data", "--output-file", "results.txt"]
    assert isotopic_calibration.configuration_from_arguments(parser.parse_args(args)).library_ion_mode == "P"
    assert isotopic_calibration.configuration_from_arguments(parser.parse_args(args + ["--library-ion-mode", "N"])).library_ion_mode == "N"
