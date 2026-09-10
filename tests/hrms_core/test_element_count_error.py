"""Exercise count margins through the actual installed Rust/Polars plugin."""

from __future__ import annotations

import inspect
import math
from dataclasses import replace
from typing import Any

import polars as pl
import pytest
from polars.plugins import register_plugin_function

from hrms_utils.element_count_error import validate_element_count_errors
from hrms_utils.formats.msdial import annotate_chromatogram_with_formulas
from hrms_utils.hrms_core import LIB, CalibratedIsotopicModel, MassDecomposerUtils
from hrms_utils.msdial_annotation import _annotation_model

# Exact constants from mass-decomposition 0.1.0 ELEMENT_ISOTOPES.
ISOTOPES = {
    "C": (1, 1.003355, 0.9893, 0.0107),
    "S": (7, 1.995796, 0.9493, 0.0429),
    "Cl": (8, 1.99705, 0.7578, 0.2422),
    "Br": (10, 1.99795, 0.5069, 0.4931),
}
MODEL = CalibratedIsotopicModel(1.0, 1.0, 0.0, 0.2, 0.0, 0.6, 0.0)


def _spectrum(element: str = "C", count: float = 10.0) -> pl.DataFrame:
    _, difference, probability_zero, probability_one = ISOTOPES[element]
    return pl.DataFrame(
        {
            "precursor": [100.0],
            "masses": [[100.0, 100.0 + difference]],
            "intensities": [[1e6, count * (probability_one * 1e6) / probability_zero]],
        }
    )


def _bounds(frame: pl.DataFrame, **kwargs: Any) -> list[int]:
    result = frame.select(
        pl.col("precursor").mass_decomposition.deduce_isotopic_pattern(
            "masses",
            "intensities",
            minimum_intensity=1.0,
            isotopic_mass_tolerance_ppm=0.1,
            model=kwargs.pop("model", MODEL),
            **kwargs,
        )
    )
    assert result.dtypes == [pl.Array(pl.Int32, 24)]
    return result.to_series()[0].to_list()


@pytest.mark.parametrize("element", list(ISOTOPES))
@pytest.mark.parametrize("minimum,maximum", [(2, 2), (3, None), (None, 2), (0, 0)])
def test_detected_count_margins_all_elements(
    element: str, minimum: int | None, maximum: int | None
) -> None:
    frame = _spectrum(element, 10.1)
    index = ISOTOPES[element][0]
    baseline = _bounds(frame)
    lower_error = max(0, 10 - baseline[index])
    upper_error = max(0, baseline[index + 12] - 10)
    if minimum is not None:
        lower_error, upper_error = max(minimum, lower_error), max(minimum, upper_error)
    if maximum is not None:
        lower_error, upper_error = min(maximum, lower_error), min(maximum, upper_error)
    result = _bounds(
        frame,
        minimum_element_count_error=None if minimum is None else {element: minimum},
        maximum_element_count_error=None if maximum is None else {element: maximum},
        max_bounds={element: 100},
    )
    assert (result[index], result[index + 12]) == (10 - lower_error, 10 + upper_error)
    assert [
        value
        for position, value in enumerate(result)
        if position not in (index, index + 12)
    ] == [
        value
        for position, value in enumerate(baseline)
        if position not in (index, index + 12)
    ]


def test_asymmetric_bounds_do_not_use_midpoint() -> None:
    frame = _spectrum(count=10.1)
    baseline = _bounds(frame)
    assert (baseline[1], baseline[13]) == (9, 18)
    assert _bounds(
        frame,
        minimum_element_count_error={"C": 2},
        maximum_element_count_error={"C": 2},
    )[1:14:12] == [8, 12]


@pytest.mark.parametrize("count,center", [(10.499999, 10), (10.5, 11), (10.500001, 11)])
def test_half_integer_rounding_and_narrow_empty_interval(
    count: float, center: int
) -> None:
    frame = _spectrum(count=count)
    model = replace(MODEL, hetero_lower_a=0.0, hetero_upper_a=0.0)
    baseline = _bounds(frame, model=model)
    assert baseline[1] > baseline[13]  # ceil/floor excludes all integers
    result = _bounds(
        frame,
        model=model,
        minimum_element_count_error={"C": 0},
        maximum_element_count_error={"C": 0},
    )
    assert result[1] == result[13] == center


def test_scale_before_count_clamps() -> None:
    frame = _spectrum(count=10.1)
    for scale in (0.5, 1.0, 2.0):
        effective = _annotation_model(MODEL, scale)
        baseline = _bounds(frame, model=effective)
        adjusted = _bounds(
            frame,
            model=effective,
            minimum_element_count_error={"C": 2},
            maximum_element_count_error={"C": 4},
        )
        assert adjusted[1] == 10 - min(4, max(2, 10 - baseline[1]))
        assert adjusted[13] == 10 + min(4, max(2, baseline[13] - 10))


def test_physical_and_global_caps_and_int32_overflow() -> None:
    assert _bounds(
        _spectrum(count=1.1),
        minimum_element_count_error={"C": 2},
        maximum_element_count_error={"C": 2},
    )[1:14:12] == [0, 3]
    assert _bounds(
        _spectrum(count=10.1),
        minimum_element_count_error={"C": 2},
        maximum_element_count_error={"C": 2},
        min_bounds={"C": 9},
        max_bounds={"C": 11},
    )[1:14:12] == [9, 11]
    # A center outside the cap yields an empty interval, not an invented cap candidate.
    assert _bounds(_spectrum(count=60.1), maximum_element_count_error={"C": 2})[1] > 50
    assert _bounds(
        _spectrum(count=10.1),
        minimum_element_count_error={"C": 2**31 - 1},
        max_bounds={"C": 2**31 - 1},
    )[1:14:12] == [0, 2**31 - 1]


@pytest.mark.parametrize(
    "masses,intensities",
    [([99.0], [1e6]), ([100.0], [1e6]), ([100.0, 101.003355], [1e6, 0.5])],
)
def test_missing_or_below_threshold_has_no_clamp(
    masses: list[float], intensities: list[float]
) -> None:
    frame = pl.DataFrame(
        {"precursor": [100.0], "masses": [masses], "intensities": [intensities]}
    )
    assert _bounds(frame) == _bounds(
        frame, minimum_element_count_error={"C": 100, "S": 100, "Cl": 100, "Br": 100}
    )


def test_exact_matching_max_intensity_and_tolerance_floor() -> None:
    # The highest matching precursor, not the first/nearest, anchors isotope mass.
    intensity = 10.1 * 0.0107 * 1e6 / 0.9893
    frame = pl.DataFrame(
        {
            "precursor": [100.0],
            "masses": [[100.0, 100.0009, 101.004255, 101.00426]],
            "intensities": [[1e5, 1e6, intensity / 2, intensity]],
        }
    )
    assert _bounds(
        frame,
        minimum_element_count_error={"C": 2},
        maximum_element_count_error={"C": 2},
    )[1:14:12] == [8, 12]


def test_both_none_and_empty_maps_match_original_serialized_native_path() -> None:
    for frame in (
        _spectrum(),
        _spectrum("S"),
        pl.DataFrame(
            {"precursor": [100.0], "masses": [[99.0]], "intensities": [[1e6]]}
        ),
    ):
        original = (
            frame.select(
                register_plugin_function(
                    args=[pl.col("precursor"), pl.col("masses"), pl.col("intensities")],
                    plugin_path=LIB,
                    function_name="deduce_isotopic_pattern",
                    is_elementwise=True,
                    kwargs={
                        "ms1_mass_tolerance_ppm": 5.0,
                        "isotopic_mass_tolerance_ppm": 0.1,
                        "minimum_intensity": 1.0,
                        "model": MODEL,
                        "min_bounds": None,
                        "max_bounds": {"C": 1, "S": 0},
                    },
                )
            )
            .to_series()[0]
            .to_list()
        )
        assert original == _bounds(
            frame,
            max_bounds={"C": 1, "S": 0},
            minimum_element_count_error=None,
            maximum_element_count_error=None,
        )
        assert original == _bounds(
            frame,
            max_bounds={"C": 1, "S": 0},
            minimum_element_count_error={},
            maximum_element_count_error={},
        )


@pytest.mark.parametrize(
    "invalid",
    [
        {"N": 2},
        {"c": 2},
        {"C": True},
        {"C": -1},
        {"C": 2.0},
        {"C": math.nan},
        {"C": math.inf},
        {"C": 2**31},
        [],
        "C=2",
    ],
)
def test_invalid_maps_rejected_before_native_serialization(invalid: Any) -> None:
    with pytest.raises(ValueError):
        _bounds(_spectrum(), minimum_element_count_error=invalid)
    with pytest.raises(ValueError):
        _bounds(_spectrum(), maximum_element_count_error=invalid)


def test_minimum_cannot_exceed_maximum() -> None:
    with pytest.raises(ValueError, match="<="):
        validate_element_count_errors({"C": 3}, {"C": 2})


def test_direct_function_arguments_are_keyword_only_and_defaults_unchanged() -> None:
    for function in (
        annotate_chromatogram_with_formulas,
        MassDecomposerUtils.deduce_isotopic_pattern,
    ):
        for name in ("minimum_element_count_error", "maximum_element_count_error"):
            parameter = inspect.signature(function).parameters[name]
            assert parameter.default is None
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    signature = inspect.signature(annotate_chromatogram_with_formulas)
    assert [
        signature.parameters[name].default
        for name in (
            "precursor_mass_accuracy_ppm",
            "fragment_mass_accuracy_ppm",
            "normalized_fragment_mass_accuracy_ppm",
        )
    ] == [3.0, 5.0, 4.0]


@pytest.mark.parametrize(
    "maps",
    [
        {"minimum_element_count_error": {"N": 2}},
        {"minimum_element_count_error": {"C": -1}},
        {
            "minimum_element_count_error": {"C": 3},
            "maximum_element_count_error": {"C": 2},
        },
        {"maximum_element_count_error": {"C": 2**31}},
    ],
)
def test_native_validation_cannot_be_bypassed(maps: dict[str, Any]) -> None:
    with pytest.raises(pl.exceptions.ComputeError):
        _spectrum().select(
            register_plugin_function(
                args=[pl.col("precursor"), pl.col("masses"), pl.col("intensities")],
                plugin_path=LIB,
                function_name="deduce_isotopic_pattern",
                is_elementwise=True,
                kwargs={
                    "ms1_mass_tolerance_ppm": 5.0,
                    "isotopic_mass_tolerance_ppm": 0.1,
                    "minimum_intensity": 1.0,
                    "model": MODEL,
                    "min_bounds": None,
                    "max_bounds": None,
                    **maps,
                },
            )
        )


def test_chromatogram_annotation_forwards_count_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class StopAfterBinding(Exception):
        pass

    def capture(self: Any, **kwargs: Any) -> pl.Expr:
        captured.update(kwargs)
        raise StopAfterBinding

    monkeypatch.setattr(MassDecomposerUtils, "deduce_isotopic_pattern", capture)
    with pytest.raises(StopAfterBinding):
        annotate_chromatogram_with_formulas(
            pl.DataFrame(),
            isotopic_model=MODEL,
            minimum_element_count_error={"C": 3},
            maximum_element_count_error={"S": 1},
        )
    assert captured["minimum_element_count_error"] == {"C": 3}
    assert captured["maximum_element_count_error"] == {"S": 1}
    assert captured["model"] is MODEL


def test_nonrepresentable_adjusted_center_keeps_empty_integer_interval() -> None:
    result = _bounds(
        _spectrum(count=1e100),
        maximum_element_count_error={"C": 2},
        max_bounds={"C": 2**31 - 1},
    )
    assert result[1] > result[13]
