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
            # A default that is not too restrictive
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
            function_name="decompose_mass",
            is_elementwise=True,
            kwargs=kwargs,
        )
