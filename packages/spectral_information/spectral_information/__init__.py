from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function
from ._internal_spectral_information import NUM_ELEMENTS
from ._internal_spectral_information import __version__ as __version__


if TYPE_CHECKING:
    from .typing import IntoExprColumn

LIB = Path(__file__).parent
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
