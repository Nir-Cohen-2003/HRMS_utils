from __future__ import annotations
from pathlib import Path
import polars as pl

def calculate_similarity(spectra: pl.Expr) -> pl.Expr:
    """
    Calculate spectral similarity between two spectra.
    """
    return pl.plugins.register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="calculate_similarity_struct",
        args=[spectra],
        is_elementwise=False,
    )
