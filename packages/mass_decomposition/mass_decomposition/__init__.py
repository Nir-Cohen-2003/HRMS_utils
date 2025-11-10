# packages/mass_decomposition/mass_decomposition/__init__.py

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from ._internal import __version__ as __version__

if TYPE_CHECKING:
    from .typing import IntoExprColumn

LIB = Path(__file__).parent

# Placeholder for the Polars expression namespace
@pl.api.register_expr_namespace("mass_decomposer")
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

        Args:
            tolerance_ppm: The mass tolerance in ppm.
            min_dbe: The minimum degree of unsaturation.
            max_dbe: The maximum degree of unsaturation.
            dbe_mode: The DBE mode, one of 'integer', 'half_integer', 'any'.
            min_bounds: The minimum elemental counts for the formula.
            max_bounds: The maximum elemental counts for the formula.
        """
        from ._internal import NUM_ELEMENTS

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
        min_dbe: float = 0.0,
        max_dbe: float = 40.0,
        dbe_mode: str = "integer",
    ) -> pl.Expr:
        """
        Decompose a mass into possible chemical formulas, with per-mass bounds.

        The input expression is expected to be a struct with the following fields:
        - 'mass': float, the mass to decompose
        - 'min_bounds': list[int], the minimum elemental counts for the formula
        - 'max_bounds': list[int], the maximum elemental counts for the formula

        Args:
            tolerance_ppm: The mass tolerance in ppm.
            min_dbe: The minimum degree of unsaturation.
            max_dbe: The maximum degree of unsaturation.
            dbe_mode: The DBE mode, one of 'integer', 'half_integer', 'any'.
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
        tolerance_ppm: float = 5.0,
        max_allowed_normalized_mass_error_ppm: float = 2.0,
        min_dbe: float = -10.0,
        max_dbe: float = 100.0,
        dbe_mode: str = "any",
        water_absorption: bool = False,
    ) -> pl.Expr:
        """
        Clean and normalize a spectrum with a known precursor.
        takes polars expression fo the type:
        pl.strcut({
            "mz": List[float64],
            "intensities": List[float64],
            "precursor_formula": Array[int, 15],
        })
        Args:
            tolerance_ppm: The mass tolerance in ppm for the initial decomposition.
            max_allowed_normalized_mass_error_ppm: The maximum allowed mass error in ppm after normalization.
            min_dbe: The minimum degree of unsaturation.
            max_dbe: The maximum degree of unsaturation.
            dbe_mode: The DBE mode, one of 'integer', 'half_integer', 'any'.
            water_absorption: Whether to allow for water absorption.
        Returns:
            A Polars expression with the cleaned and normalized spectrum, of the type:
            pl.Struct({
                "masses_normalized": pl.List(pl.Float64),
                "cleaned_intensities": pl.List(pl.Float64),
                "fragment_formulas": pl.List(pl.Array(inner=pl.Int32,shape=(num_elements,))),
                "fragment_formulas_str": pl.List(pl.String),
                "fragment_errors_ppm": pl.List(pl.Float64),
            })
        """
        kwargs = {
            "tolerance_ppm": tolerance_ppm,
            "max_allowed_normalized_mass_error_ppm": max_allowed_normalized_mass_error_ppm,
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


