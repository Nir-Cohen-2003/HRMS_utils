from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from ._internal import __version__ as __version__

if TYPE_CHECKING:
    from .typing import IntoExprColumn

LIB = Path(__file__).parent
@pl.api.register_expr_namespace("spectral_similarity")
class SpectralUtils:
    def __init__(self, expr : pl.Expr):
        self._expr = expr

    def entropy_similarity(
        self,
        ms2_tolerance_in_ppm: float ,
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
