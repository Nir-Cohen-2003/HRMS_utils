from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from ._internal import __version__ as __version__

if TYPE_CHECKING:
    from .typing import IntoExprColumn

LIB = Path(__file__).parent

def calculate_similarity(spectra: pl.Expr) -> pl.Expr:
    """
    Calculate spectral similarity between two spectra.
    """
    return pl.plugins.register_plugin_function(
        args=[spectra],
        plugin_path=LIB,
        function_name="calculate_similarity_struct",
        is_elementwise=True,
    )
