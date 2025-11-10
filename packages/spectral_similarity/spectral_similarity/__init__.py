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
        takes a struct with fields:
        - mz1: List[Float64]
        - intensities1: List[Float64]
        - mz2: List[Float64]
        - intensities2: List[Float64]
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

        Args:
            intensity_power: The power to raise the intensity to.
            mass_power: The power to raise the mass to.
            ms2_tolerance_in_ppm: The tolerance for matching peaks in ppm.
            clean_spectra_first: Whether to clean the spectra before calculating similarity.
            noise_threshold: The noise threshold for cleaning.
            precursor_mz: The precursor m/z to filter fragments.
            ignore_precursor: Whether to ignore the precursor peak.
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
        Calculate mass weighted cosine similarity (intensity^0.5, mass^2).
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
        Calculate NIST-like dot product similarity (intensity^0.5, mass^0.0).
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
        clean_spectra_first: bool = True,
        noise_threshold: float = 0.001,
        ignore_precursor: bool = False,
        permissive: bool = False,
    ) -> pl.Expr:
        """
        Calculate explained intensity.
        Assumes spectrum 1 is a subset of spectrum 2.
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
