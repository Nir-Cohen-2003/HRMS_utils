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

    # This will be filled with actual plugin functions later
    def placeholder_function(self) -> pl.Expr:
        return self._expr
