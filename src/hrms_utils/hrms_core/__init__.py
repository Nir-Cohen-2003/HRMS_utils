# packages/hrms_core/hrms_core/__init__.py

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

from .._internal import (
    NUM_ELEMENTS,  # type: ignore
    read_mzml_files,  # type: ignore
    read_thermo_files,  # type: ignore
)
from .._internal import __version__ as __version__  # type: ignore

if TYPE_CHECKING:
    from .typing import IntoExprColumn

LIB: Path = Path(__file__).parent.parent


# Placeholder for the Polars expression namespace
@pl.api.register_expr_namespace("mass_decomposition")
class MassDecomposerUtils:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def decompose_mass(
        self,
        tolerance_ppm: float = 5.0,
        min_dbe: float = 0.0,
        max_dbe: float = 40.0,
        dbe_mode: str = "integer",
        min_bounds: list[int] | None = None,
        max_bounds: list[int] | None = None,
    ) -> pl.Expr:
        """
        Decompose a mass into possible chemical formulas.

        Input expression:
            pl.Float64

        Args:
            tolerance_ppm: The mass tolerance in ppm.
            min_dbe: The minimum degree of unsaturation.
            max_dbe: The maximum degree of unsaturation.
            dbe_mode: The DBE mode, one of 'integer', 'half_integer', 'any'.
            min_bounds: The minimum elemental counts for the formula.
            max_bounds: The maximum elemental counts for the formula.

        Returns:
            A Polars expression with the decomposition results, of the type:
            pl.Struct({
                "formulas": pl.List(pl.Array(pl.Int32, 15)),
                "formulas_str": pl.List(pl.String),
                "errors_ppm": pl.List(pl.Float64),
            })
        """
        if min_bounds is None:
            min_bounds = [0] * NUM_ELEMENTS
        if max_bounds is None:
            max_bounds = [100] * NUM_ELEMENTS

        kwargs = {
            "tolerance_ppm": tolerance_ppm,
            "min_dbe": min_dbe,
            "max_dbe": max_dbe,
            "dbe_mode": dbe_mode,
            "min_bounds": min_bounds,
            "max_bounds": max_bounds,
        }

        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="mass_decomposition",
            is_elementwise=True,
            kwargs=kwargs,
        )

    def decompose_mass_with_bounds(
        self,
        tolerance_ppm: float = 5.0,
        min_dbe: float = -0.5,
        max_dbe: float = 40.0,
        dbe_mode: str = "integer",
    ) -> pl.Expr:
        """
        Decompose a mass into possible chemical formulas, with per-mass bounds.

        Input expression:
            pl.Struct({
                "mass": pl.Float64,
                "min_bounds": pl.Array(pl.Int32),
                "max_bounds": pl.Array(pl.Int32),
            })

        Args:
            tolerance_ppm: The mass tolerance in ppm.
            min_dbe: The minimum degree of unsaturation.
            max_dbe: The maximum degree of unsaturation.
            dbe_mode: The DBE mode, one of 'integer', 'half_integer', 'any'.

        Returns:
            A Polars expression with the decomposition results, of the type:
            pl.Struct({
                "formulas": pl.List(pl.Array(pl.Int32, 15)),
                "formulas_str": pl.List(pl.String),
                "errors_ppm": pl.List(pl.Float64),
            })
        """
        kwargs = {
            "tolerance_ppm": tolerance_ppm,
            "min_dbe": min_dbe,
            "max_dbe": max_dbe,
            "dbe_mode": dbe_mode,
        }

        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="mass_decomposition_with_bounds",
            is_elementwise=True,
            kwargs=kwargs,
        )

    def clean_and_normalize_spectrum(
        self,
        raw_fragment_tolerance_ppm: float = 5.0,
        normalized_fragment_tolerance_ppm: float = 2.0,
        min_dbe: float = -10.0,
        max_dbe: float = 100.0,
        dbe_mode: str = "any",
        water_absorption: bool = False,
    ) -> pl.Expr:
        """
        Clean and normalize a spectrum with a known precursor.
        Input expression:
            pl.Struct({
                "mz": List[float64],
                "intensities": List[float64],
                "precursor_formula": Array[int, 15],
            })
        Args:
            raw_fragment_tolerance_ppm: The mass tolerance in ppm for the initial decomposition.
            normalized_fragment_tolerance_ppm: The maximum allowed mass error in ppm after normalization.
            min_dbe: The minimum degree of unsaturation.
            max_dbe: The maximum degree of unsaturation.
            dbe_mode: The DBE mode, one of 'integer', 'half_integer', 'any'.
            water_absorption: Whether to allow for water absorption.
        Returns:
            A Polars expression with the cleaned and normalized spectrum, of the type:
            pl.Struct({
                "formulas": pl.List(pl.Array(pl.Int32, 15)),
                "formulas_str": pl.List(pl.String),
                "normalized_masses": pl.List(pl.Float64),
                "intensities": pl.List(pl.Float64),
                "errors_ppm": pl.List(pl.Float64),
            })
        """
        kwargs = {
            "raw_fragment_tolerance_ppm": raw_fragment_tolerance_ppm,
            "normalized_fragment_tolerance_ppm": normalized_fragment_tolerance_ppm,
            "min_dbe": min_dbe,
            "max_dbe": max_dbe,
            "dbe_mode": dbe_mode,
            "water_absorption": water_absorption,
        }

        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="spectrum_decomposition_normalized",
            is_elementwise=True,
            kwargs=kwargs,
        )

    def deduce_isotopic_pattern(
        self,
        ms1_mzs: IntoExprColumn,
        ms1_intensities: IntoExprColumn,
        ms1_mass_tolerance_ppm: float = 5.0,
        isotopic_mass_tolerance_ppm: float = 3.0,
        minimum_intensity: float = 5e4,
        intensity_absolute_tolerance: float = 5e4,
        intensity_relative_tolerance: float = 0.05,
        min_bounds: dict[str, int] | None = None,
        max_bounds: dict[str, int] | None = None,
    ) -> pl.Expr:
        """
        Deduce the isotopic pattern from the given precursor and MS1 data.

        Input expression (self): Precursor m/z (Float64)

        Args:
            ms1_mzs: Expression/column for MS1 m/z values (List[Float64])
            ms1_intensities: Expression/column for MS1 intensities (List[Float64])
            ms1_mass_tolerance_ppm: Tolerance for matching precursor in MS1
            isotopic_mass_tolerance_ppm: Tolerance for matching isotopic peaks
            minimum_intensity: Minimum intensity threshold
            intensity_absolute_tolerance: Absolute tolerance for intensity matching
            intensity_relative_tolerance: Relative tolerance for intensity matching
            min_bounds: Optional dictionary of minimum element counts
            max_bounds: Optional dictionary of maximum element counts

        Returns:
            pl.Expr: Series of Array(Int32, 24) containing [min_counts..., max_counts...]
        """
        kwargs = {
            "ms1_mass_tolerance_ppm": ms1_mass_tolerance_ppm,
            "isotopic_mass_tolerance_ppm": isotopic_mass_tolerance_ppm,
            "minimum_intensity": minimum_intensity,
            "intensity_absolute_tolerance": intensity_absolute_tolerance,
            "intensity_relative_tolerance": intensity_relative_tolerance,
            "min_bounds": min_bounds,
            "max_bounds": max_bounds,
        }

        return register_plugin_function(
            args=[self._expr, ms1_mzs, ms1_intensities],
            plugin_path=LIB,
            function_name="deduce_isotopic_pattern",
            is_elementwise=True,
            kwargs=kwargs,
        )


@pl.api.register_expr_namespace("spectral_info")
class SpectralInfoNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def spectral_info_score(
        self,
        *,
        distance_metric: str = "l2",
        ignore_hydrogens: bool = True,
    ) -> pl.Expr:
        """
        Calculates a tree-based spectral information score for each spectrum in a Polars DataFrame.

        The algorithm models relationships between fragments (and precursor) as a tree,
        where edges connect sub-formulas to super-formulas. The score is derived from
        the entropy of the minimum distances between connected nodes in a normalized space.

        Args:
            A struct with the following fields:
            "precursor_formula": Array(Int32, NUM_ELEMENTS),
            "fragment_formulas": List(Array(Int32, NUM_ELEMENTS)),
            and with keywords:

            distance_metric: The distance metric for comparing normalized formulas.
                            One of 'l1', 'l2', or 'cosine'. Defaults to 'l2'.
            ignore_hydrogens: If True, ignores the first element (hydrogen) when comparing formulas.
                            This can cause fragments to become identical. Defaults to True.

        Returns:
            A Polars Series of Float64 scores, one for each input spectrum.
        """
        kwargs = {
            "distance_metric": distance_metric,
            "ignore_hydrogens": ignore_hydrogens,
        }

        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="tree_spectral_info_score",
            is_elementwise=True,
            kwargs=kwargs,
        )


    def spectral_info_score_per_fragment(
        self,
        *,
        distance_metric: Literal["l1", "l2", "cosine"] = "l2",
        ignore_hydrogens: bool = True,
    ) -> pl.Expr:
        """
        Calculate spectral information score for each fragment in a spectrum based on a tree structure.

        Input expression requirements:
        - The expression must evaluate to a Struct column with the following fields:
          - precursor_formula: Array(Int32, NUM_ELEMENTS)
          - fragment_formulas: List(Array(Int32, NUM_ELEMENTS))

        Returns:
            A Polars Series of List(Float64) scores, one score for each fragment in the input spectrum.
        """
        kwargs = {
            "distance_metric": distance_metric,
            "ignore_hydrogens": ignore_hydrogens,
        }

        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="tree_spectral_info_score_per_fragment",
            is_elementwise=True,
            kwargs=kwargs,
        )

@pl.api.register_expr_namespace("spectral_similarity")
class SpectralUtils:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def entropy_similarity(
        self,
        ms2_tolerance_in_ppm: float,
        clean_spectra_first: bool = True,
        noise_threshold: float = 0.001,
        ignore_precursor: bool = False,
    ) -> pl.Expr:
        """
        Calculate spectral similarity between two spectra using spectral entropy.

        Input expression requirements:
        - The expression must evaluate to a Struct column with the following fields:
          - mz1: List[Float64]            # m/z values for spectrum 1 (polars List series)
          - intensities1: List[Float64]   # intensities for spectrum 1
          - mz2: List[Float64]            # m/z values for spectrum 2
          - intensities2: List[Float64]   # intensities for spectrum 2
          - precursor_mz1: Float64        # precursor m/z for spectrum 1
          - precursor_mz2: Float64        # precursor m/z for spectrum 2
          - weights1: List[Float64]       # (Optional) explicit weights for spectrum 1
          - weights2: List[Float64]       # (Optional) explicit weights for spectrum 2

        Return:
        - A pl.Expr that evaluates elementwise to a Float64 similarity score (nullable).
          Rows where required fields are missing will produce null.
        """
        kwargs = {
            "ms2_tolerance_in_ppm": ms2_tolerance_in_ppm,
            "clean_spectra_first": clean_spectra_first,
            "noise_threshold": noise_threshold,
            "ignore_precursor": ignore_precursor,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="calculate_similarity_struct",
            is_elementwise=True,
            kwargs=kwargs,
        )

    def general_cosine_similarity(
        self,
        intensity_power: float,
        mass_power: float,
        ms2_tolerance_in_ppm: float,
        clean_spectra_first: bool = True,
        noise_threshold: float = 0.001,
        ignore_precursor: bool = False,
    ) -> pl.Expr:
        """
        Calculate a general weighted cosine similarity between two spectra.

        Input expression requirements:
        - Same struct as entropy_similarity:
          mz1, intensities1, mz2, intensities2 (List[Float64]) and
          precursor_mz1, precursor_mz2 (Float64).

        Return:
        - A pl.Expr that evaluates elementwise to a Float64 similarity score (nullable).
        - The score is computed using the provided intensity_power and mass_power.
        """
        kwargs = {
            "intensity_power": intensity_power,
            "mass_power": mass_power,
            "ms2_tolerance_in_ppm": ms2_tolerance_in_ppm,
            "clean_spectra_first": clean_spectra_first,
            "noise_threshold": noise_threshold,
            "ignore_precursor": ignore_precursor,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="cosine_similarity_struct",
            is_elementwise=True,
            kwargs=kwargs,
        )

    def mass_weighted_cosine_similarity(
        self,
        ms2_tolerance_in_ppm: float,
        clean_spectra_first: bool = True,
        noise_threshold: float = 0.001,
        ignore_precursor: bool = False,
    ) -> pl.Expr:
        """
        Mass-weighted cosine similarity (intensity^0.5, mass^2).

        Input/Return: same struct input and Float64 (nullable) output as general_cosine_similarity.
        """
        return self.general_cosine_similarity(
            intensity_power=0.5,
            mass_power=2.0,
            ms2_tolerance_in_ppm=ms2_tolerance_in_ppm,
            clean_spectra_first=clean_spectra_first,
            noise_threshold=noise_threshold,
            ignore_precursor=ignore_precursor,
        )

    def dotprod_similarity(
        self,
        ms2_tolerance_in_ppm: float,
        clean_spectra_first: bool = True,
        noise_threshold: float = 0.001,
        ignore_precursor: bool = False,
    ) -> pl.Expr:
        """
        NIST-like dot product similarity (intensity^0.5, mass^0.0).

        Input/Return: same struct input and Float64 (nullable) output as general_cosine_similarity.
        """
        return self.general_cosine_similarity(
            intensity_power=0.5,
            mass_power=0.0,
            ms2_tolerance_in_ppm=ms2_tolerance_in_ppm,
            clean_spectra_first=clean_spectra_first,
            noise_threshold=noise_threshold,
            ignore_precursor=ignore_precursor,
        )

    def explained_intensity(
        self,
        ms2_tolerance_in_ppm: float,
        intensity_power: float = 1.0,
        mass_power: float = 0.0,
        clean_spectra_first: bool = False,
        noise_threshold: float = 0.001,
        ignore_precursor: bool = False,
        permissive: bool = False,
    ) -> pl.Expr:
        """
        Calculate explained intensity. Assumes spectrum 1 is a subset of spectrum 2.

        Input expression requirements:
        - Same struct as other similarity functions:
          mz1, intensities1, mz2, intensities2 (List[Float64]) and
          precursor_mz1, precursor_mz2 (Float64).

        Return:
        - A pl.Expr that evaluates elementwise to a Float64 explained-intensity score (nullable).
        - Rows with missing required fields will produce null.
        """
        kwargs = {
            "intensity_power": intensity_power,
            "mass_power": mass_power,
            "ms2_tolerance_in_ppm": ms2_tolerance_in_ppm,
            "clean_spectra_first": clean_spectra_first,
            "noise_threshold": noise_threshold,
            "ignore_precursor": ignore_precursor,
            "permissive": permissive,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="explained_intensity_struct",
            is_elementwise=True,
            kwargs=kwargs,
        )


def read_mzml(paths: list[str]) -> list[pl.DataFrame]:
    """
    Read multiple mzML files into Polars DataFrames using the Rust backend.

    Args:
        paths: List of file paths to read.

    Returns:
        List of Polars DataFrames, one for each file.
    """
    return read_mzml_files(paths)
