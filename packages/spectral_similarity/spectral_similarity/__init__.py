from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from ._internal import __version__ as __version__

if TYPE_CHECKING:
    from .typing import IntoExprColumn

LIB = Path(__file__).parent

def entropy_similarity(
    spectra: pl.Expr,
    ms2_tolerance_in_ppm: float | None = None,
    clean_spectra_first: bool | None = None,
    noise_threshold: float | None = 0.001,
) -> pl.Expr:
    """
    Calculate spectral similarity between two spectra.
    takes a struct with fields:
    - mz1: List[Float32]
    - intensities1: List[Float32]
    - mz2: List[Float32]
    - intensities2: List[Float32]
    """
    kwargs = {
        "ms2_tolerance_in_ppm": ms2_tolerance_in_ppm,
        "clean_spectra_first": clean_spectra_first,
        "noise_threshold": noise_threshold,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    return register_plugin_function(
        args=[spectra],
        plugin_path=LIB,
        function_name="calculate_similarity_struct",
        is_elementwise=True,
        kwargs=kwargs,
    )
