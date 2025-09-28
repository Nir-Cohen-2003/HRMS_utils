import polars as pl
import numpy as np
from hrms_utils.formula_annotation import (
    NUM_ELEMENTS,
    clean_and_normalize_spectra_known_precursor_verbose,
    clean_spectra_known_precursor_verbose,
    decompose_mass_per_bounds_verbose,
    decompose_mass_verbose,
    decompose_spectra_known_precursor_verbose,
)


def test_decompose_mass_verbose_empty_schema() -> None:
    mass_series = pl.Series(name="mass", values=[], dtype=pl.Float64)
    bounds_min = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    bounds_max = np.full(NUM_ELEMENTS, 5, dtype=np.int32)

    formulas, formula_strings = decompose_mass_verbose(
        mass_series=mass_series,
        min_bounds=bounds_min,
        max_bounds=bounds_max,
        tolerance_ppm=5.0,
        max_results=32,
    )

    assert formulas.len() == 0
    expected_dtype = pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))
    assert formulas.dtype == expected_dtype
    assert formula_strings.len() == 0
    assert formula_strings.dtype == pl.List(pl.Utf8)


def test_decompose_mass_per_bounds_verbose_empty_schema() -> None:
    mass_series = pl.Series(name="mass", values=[], dtype=pl.Float64)
    bounds_dtype = pl.Array(pl.Int32, NUM_ELEMENTS)
    min_bounds = pl.Series(name="min_bounds", values=[], dtype=bounds_dtype)
    max_bounds = pl.Series(name="max_bounds", values=[], dtype=bounds_dtype)

    formulas, formula_strings = decompose_mass_per_bounds_verbose(
        mass_series=mass_series,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        tolerance_ppm=5.0,
        max_results=16,
    )

    assert formulas.len() == 0
    assert formulas.dtype == pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))
    assert formula_strings.len() == 0
    assert formula_strings.dtype == pl.List(pl.Utf8)


def test_decompose_spectra_known_precursor_verbose_empty_schema() -> None:
    precursor_series = pl.Series(name="precursor", values=[], dtype=pl.Array(pl.Int32, NUM_ELEMENTS))
    fragments = pl.Series(name="fragments", values=[], dtype=pl.List(pl.Float64))

    formulas, formula_strings = decompose_spectra_known_precursor_verbose(
        precursor_formula_series=precursor_series,
        fragment_masses_series=fragments,
        tolerance_ppm=5.0,
    )

    assert formulas.len() == 0
    assert formulas.dtype == pl.List(pl.List(pl.Array(pl.Int32, NUM_ELEMENTS)))
    assert formula_strings.len() == 0
    assert formula_strings.dtype == pl.List(pl.List(pl.Utf8))


def test_clean_spectra_known_precursor_verbose_empty_struct() -> None:
    precursor_series = pl.Series(name="precursor", values=[], dtype=pl.Array(pl.Int32, NUM_ELEMENTS))
    masses_series = pl.Series(name="masses", values=[], dtype=pl.List(pl.Float64))
    intensities_series = pl.Series(name="intensities", values=[], dtype=pl.List(pl.Float64))

    struct_series = clean_spectra_known_precursor_verbose(
        precursor_formula_series=precursor_series,
        fragment_masses_series=masses_series,
        fragment_intensities_series=intensities_series,
        tolerance_ppm=5.0,
    )

    assert struct_series.len() == 0
    field_dtype = struct_series.struct.field("fragment_formulas_str").dtype
    assert field_dtype == pl.List(pl.List(pl.Utf8))


def test_clean_and_normalize_verbose_empty_struct() -> None:
    precursor_series = pl.Series(name="precursor", values=[], dtype=pl.Array(pl.Int32, NUM_ELEMENTS))
    precursor_masses = pl.Series(name="precursor_mass", values=[], dtype=pl.Float64)
    masses_series = pl.Series(name="masses", values=[], dtype=pl.List(pl.Float64))
    intensities_series = pl.Series(name="intensities", values=[], dtype=pl.List(pl.Float64))

    struct_series = clean_and_normalize_spectra_known_precursor_verbose(
        precursor_formula_series=precursor_series,
        precursor_masses_series=precursor_masses,
        fragment_masses_series=masses_series,
        fragment_intensities_series=intensities_series,
        tolerance_ppm=5.0,
        max_allowed_normalized_mass_error_ppm=3.0,
    )

    assert struct_series.len() == 0
    field_dtype = struct_series.struct.field("fragment_formulas_str").dtype
    assert field_dtype == pl.List(pl.Utf8)
    assert struct_series.struct.field("masses_normalized").dtype == pl.List(pl.Float64)
