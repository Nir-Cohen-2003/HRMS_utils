import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, Iterable, List, Literal, Tuple, TypeVar, cast

import numpy as np
import polars as pl

from ..formula_annotation.element_table import ELEMENT_INDEX, ELEMENT_MASSES
from ..hrms_core import *

T = TypeVar("T", pl.DataFrame, pl.LazyFrame)


MSDIAL_columns_to_read = {
    "Peak ID": pl.Int64,
    "Scan": pl.Int64,
    "RT left(min)": pl.Float64,
    "RT (min)": pl.Float64,
    "RT right (min)": pl.Float64,
    "Precursor m/z": pl.Float64,
    "Height": pl.Float64,
    "Adduct": pl.String,
    "Isotope": pl.Int32,
    "MSMS spectrum": pl.String,  # will be converted to 2 lists, m/z and intensity
    "MS1 isotopes": pl.String,  # will be converted to 2 lists, m/z and intensity
}

MSDIAL_other_columns = [
    "Estimated noise",
    "S/N",
    "Sharpness",
    "Gaussian similarity",
    "Ideal slope",
    "Symmetry",
    "MS1 isotopes",  #'S/N', (second S/N has the same values as the first one.)
]


MSDIAL_columns_to_output = [
    "Peak ID",
    "RT (min)",
    "Precursor_mz_MSDIAL",
    "Height",
    "Precursor_type_MSDIAL",
    "Isotope",
    "msms_m/z",
    "msms_intensity",
    "isobars",
    "msms_m/z_cleaned",
    "msms_intensity_cleaned",
    "energy_is_too_low",
    "energy_is_too_high",
    "ms1_isotopes_m/z",
    "ms1_isotopes_intensity",
]


@dataclass
class blank_config:
    ms1_mass_tolerance: float = 3e-6
    dRT_min: float = 0.1
    ratio: float | int = 5
    use_ms2: bool = False
    dRT_min_with_ms2: float = 0.3
    ms2_fit: float = 0.85
    ms2_mass_tolerance: float = 5e-6  # new field
    noise_threshold: float = 0.005  # new field

    def __post_init__(self):
        if (
            self.ms1_mass_tolerance > 0.0001
        ):  # if the value is more than 0.0001, its a ppm value and we multiply by 1e-6
            self.ms1_mass_tolerance = self.ms1_mass_tolerance * 1e-6
        if self.use_ms2:
            # Ensure ms2_mass_tolerance and noise_threshold are set if use_ms2 is True
            if (
                not hasattr(self, "ms2_mass_tolerance")
                or self.ms2_mass_tolerance is None
            ):
                self.ms2_mass_tolerance = 5e-6
            if not hasattr(self, "noise_threshold") or self.noise_threshold is None:
                self.noise_threshold = 0.005

    def to_dict(self) -> dict:
        return {
            "ms1_mass_tolerance": self.ms1_mass_tolerance,
            "dRT_min": self.dRT_min,
            "ratio": self.ratio,
            "use_ms2": self.use_ms2,
            "dRT_min_with_ms2": self.dRT_min_with_ms2,
            "ms2_fit": self.ms2_fit,
            "ms2_mass_tolerance": self.ms2_mass_tolerance,
            "noise_threshold": self.noise_threshold,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "blank_config":
        return cls(**config_dict)


def get_chromatogram(path: str | Path) -> pl.DataFrame:
    """Reads the .txt output of a complete chromatogram from MSDIAL (note- use the "trim content fo excel option), and returns a polars dataframe with the following schema:
    Peak ID: pl.Int64
    RT (min): pl.Float64
    Precursor_mz_MSDIAL: pl.Float64
    Height: pl.Float64
    Precursor_type_MSDIAL: pl.String
    msms_m/z: pl.List(pl.Float64)
    msms_intensity: pl.List(pl.Float64)
    isobars: pl.List(pl.Int64)
    msms_m/z_cleaned: pl.List(pl.Float64)
    msms_intensity_cleaned: pl.List(pl.Float64)
    energy_is_too_low: pl.Boolean
    energy_is_too_high: pl.Boolean
    ms1_isotopes_m/z: pl.List(pl.Float64)
    ms1_isotopes_intensity: pl.List(pl.Float64)
    """
    chromatogram = _get_chromatogram_basic(path=path)
    chromatogram = _annotate_isobars_and_clean_spectrum(chromatogram=chromatogram)
    chromatogram = _add_energy_annotation(chromatogram=chromatogram)
    chromatogram = chromatogram.select(MSDIAL_columns_to_output)
    if not isinstance(chromatogram, pl.DataFrame):
        raise Exception("failed getting chromatogram from the file: " + str(path))

    return chromatogram


def subtract_blank_frame(
    sample_df: pl.DataFrame, blank_df: pl.DataFrame, config: blank_config
) -> pl.DataFrame:
    """subtracts a blank chromatogram, using ms1, ms2 and RT.
    in absense of ms2 for either the blank or the sample compound, a stricter rt threshold is used.
    keep dRT_min_with_ms2 > dRT_min, or the logic gets wrong."""

    if not config.use_ms2:  # so when both sample and blank spectra has msms, we require a 0.85 fit on ms2, but lower fit on rt. if any of them lacks ms2, we just use strict rt.
        sample_lf = sample_df.select(
            [
                "Peak ID",
                "RT (min)",
                "Precursor_mz_MSDIAL",
                "Height",
            ]
        ).lazy()
        blank_lf = blank_df.select(
            [
                "RT (min)",
                "Precursor_mz_MSDIAL",
                "Height",
            ]
        ).lazy()

        subtract_df = sample_lf.join_where(
            blank_lf,
            pl.col("RT (min)") < pl.col("RT (min)_blank") + config.dRT_min,
            pl.col("RT (min)") > pl.col("RT (min)_blank") - config.dRT_min,
            (
                pl.col("Precursor_mz_MSDIAL").truediv(
                    pl.col("Precursor_mz_MSDIAL_blank")
                )
                - 1.0
            )
            .abs()
            .le(config.ms1_mass_tolerance),
            pl.col("Height") < pl.col("Height_blank") * config.ratio,
            suffix="_blank",
        ).collect(engine="streaming")
    else:  # so we just use strict rt
        sample_lf = sample_df.select(
            [
                "Peak ID",
                "RT (min)",
                "Precursor_mz_MSDIAL",
                "Height",
                "msms_m/z",
                "msms_intensity",
            ]
        ).lazy()
        blank_lf = blank_df.select(
            ["RT (min)", "Precursor_mz_MSDIAL", "Height", "msms_m/z", "msms_intensity"]
        ).lazy()
        subtract_lf = sample_lf.join_where(
            blank_lf,
            pl.col("RT (min)") < pl.col("RT (min)_blank") + config.dRT_min_with_ms2,
            pl.col("RT (min)") > pl.col("RT (min)_blank") - config.dRT_min_with_ms2,
            (
                pl.col("Precursor_mz_MSDIAL").truediv(
                    pl.col("Precursor_mz_MSDIAL_blank")
                )
                - 1
            )
            .abs()
            .le(config.ms1_mass_tolerance),
            pl.col("Height") < pl.col("Height_blank") * config.ratio,
            suffix="_blank",
        )
        subtract_lf_rt_strict = subtract_lf.filter(
            pl.col("msms_m/z").is_null() | pl.col("msms_m/z_blank").is_null()
        )
        subtract_lf_rt_strict = subtract_lf_rt_strict.filter(
            pl.col("RT (min)") < pl.col("RT (min)_blank") + config.dRT_min,
            pl.col("RT (min)") > pl.col("RT (min)_blank") - config.dRT_min,
        )

        subtract_df_ms2 = subtract_lf.filter(
            pl.col("msms_m/z").is_not_null(), pl.col("msms_m/z_blank").is_not_null()
        ).collect(engine="streaming")

        subtract_df_ms2 = subtract_df_ms2.filter(
            pl.struct(  # type: ignore[missing-attribute]
                pl.col("msms_intensity").alias("intensities1"),
                pl.col("msms_m/z").alias("mz1"),
                pl.col("msms_intensity_blank").alias("intensities2"),
                pl.col("msms_m/z_blank").alias("mz2"),
                pl.col("Precursor_mz_MSDIAL").alias("precursor_mz1"),
                pl.col("Precursor_mz_MSDIAL_blank").alias("precursor_mz2"),
            )
            .spectral_similarity.dotprod_similarity(
                ms2_tolerance_in_ppm=config.ms2_mass_tolerance,
                clean_spectra_first=True,
                noise_threshold=0.001,
                ignore_precursor=True,
            )
            .ge(config.ms2_fit)
        )

        subtract_df = pl.concat(
            [subtract_df_ms2, subtract_lf_rt_strict.collect(engine="streaming")]
        )

    cleaned_sample_df = sample_df.join(subtract_df, on="Peak ID", how="anti")
    return cleaned_sample_df


def annotate_chromatogram_with_formulas(
    chromatogram: pl.DataFrame,
    max_bounds: dict | None = None,
    precursor_mass_accuracy_ppm: float = 3.0,
    fragment_mass_accuracy_ppm: float = 5.0,
    normalized_fragment_mass_accuracy_ppm: float = 4.0,
    isotopic_mass_accuracy_ppm: float = 2.0,
    isotopic_minimum_intensity: float = 5e4,
    isotopic_intensity_absolute_tolerance: float = 2e5,
    isotopic_intensity_relative_tolerance: float = 0.1,
) -> pl.DataFrame:
    """
    Annotate an MSDIAL chromatogram with isotopic patterns, candidate elemental formulas
    and cleaned/normalized MS/MS fragments.

    What the function does
    - Deduce an isotopic pattern (element count bounds) for each precursor using the
      observed MS1 isotopes.
    - Compute candidate elemental decompositions for the (non-ionized) precursor mass
      using the deduced bounds and the provided precursor mass tolerance.
    - For each candidate formula, shift MS/MS fragment m/z values to the non-ionized
      frame (subtracting addcut_mass), then clean and normalize the fragment spectrum
      against the candidate formula (matching fragment masses with tolerance and
      filtering/noise handling).
    - Explode the chromatogram so each output row corresponds to one candidate precursor
      formula (i.e., one decomposition), with accompanying cleaned MS/MS results.

    Arguments
    - chromatogram: pl.DataFrame
        Input chromatogram. Required input columns (types expected):
        - "Precursor_mz_MSDIAL": Float (precursor m/z measured by MSDIAL)
        - "ms1_isotopes_m/z": List(Float) (observed MS1 isotope m/z values)
        - "ms1_isotopes_intensity": List(Float) (absolute intensities for MS1 isotopes)
        - "msms_m/z": List(Float) (observed MS/MS fragment m/z)
        - "msms_intensity": List(Float) (observed MS/MS fragment intensities)
        The function fails fast if required columns are missing.

    - addcut_mass: float
        Mass of the adduct or proton to subtract from observed m/z to obtain the
        neutral (non-ionized) mass. Default is PROTON_MASS.

    - max_bounds: dict | None
        Optional user-specified upper bounds for element counts used by the isotopic
        pattern deduction. If None, bounds are deduced from the data.

    - precursor_mass_accuracy_ppm: float
        Tolerance in ppm used for precursor mass matching (isotope deduction and
        mass decomposition). Units: ppm.

    - fragment_mass_accuracy_ppm: float
        Tolerance in ppm used when matching observed fragment m/z to theoretical
        fragment masses during cleaning. Units: ppm.

    - normalized_fragment_mass_accuracy_ppm: float
        Maximum allowed normalized mass error (in ppm) for fragment mass normalization
        checks performed during cleaning/normalization.

    - isotopic_mass_accuracy_ppm: float
        Tolerance in ppm used when matching isotopic peaks to the expected isotope
        positions during isotopic pattern deduction.

    - isotopic_minimum_intensity: float
        Minimum absolute intensity to consider an isotope peak when deducing the
        isotopic pattern.

    - isotopic_intensity_absolute_tolerance: float
        Absolute intensity tolerance used when comparing expected vs observed isotope
        intensities.

    - isotopic_intensity_relative_tolerance: float
        Relative intensity tolerance (fraction) used when comparing expected vs
        observed isotope intensities.

    Returned DataFrame (columns added / meaning)
    The returned polars DataFrame contains the original chromatogram columns plus the
    following annotations (types indicated informally):

    - min_bounds: Array(Int32) length NUM_ELEMENTS
        Per-element minimum counts inferred for the precursor formula (from isotopic pattern).
    - max_bounds: Array(Int32) length NUM_ELEMENTS
        Per-element maximum counts inferred for the precursor formula (from isotopic pattern).
    - non_ionized_mass: Float
        Precursor neutral mass = Precursor_mz_MSDIAL - addcut_mass.
    - precursor_formula: Array(Int32, shape=(NUM_ELEMENTS,))
        Candidate elemental formula for the precursor (each is an integer vector of
        element counts). The function explodes this column so each output row contains
        exactly one candidate formula (a single Array(Int32,...)).
    - non_ionized_msms_m/z: List(Float)
        MS/MS fragment m/z values shifted to the neutral frame (each mz - addcut_mass).
    - cleaned_msms_mz: List(Float)
        Cleaned and (internally) normalized fragment masses that were retained after
        matching against the candidate formula and applying mass tolerances.
    - cleaned_msms_intensity: List(Float)
        Corresponding intensities for cleaned_msms_mz (normalized/filtered by the
        cleaning routine).
    - cleaned_spectrum_formulas: List(Array(Int32, shape=(NUM_ELEMENTS,)))
        Per-fragment candidate elemental formulas (for fragments that were assigned a
        formula by the cleaning routine). Each fragment formula is represented as an
        element-count array aligned with NUM_ELEMENTS.
    - cleaned_fragment_errors_ppm: List(Float)
        Mass errors (in ppm) for the matched fragments after cleaning/normalization.

    Notes and behavior
    - The function relies on domain utilities: deduce_isotopic_pattern,
      and the mass_decomposition polars plugin. Any
      change in those APIs must be propagated here.
    - One input precursor row may expand into multiple output rows (one per candidate
      decomposition) because of the explosion of "decomposed_formulas".
    - Mass tolerances are expressed in ppm; callers should pass values appropriate
      for their instrument and data quality.
    - The function performs no downstream filtering of candidate formulas; downstream
      ranking/selection is the caller's responsibility.
    """
    # Isotopic pattern deduction
    chromatogram = (
        chromatogram.with_columns(
            pl.col("Precursor_mz_MSDIAL")  # type: ignore[missing-attribute]
            .mass_decomposition.deduce_isotopic_pattern(
                ms1_mzs=pl.col("ms1_isotopes_m/z"),
                ms1_intensities=pl.col("ms1_isotopes_intensity"),
                ms1_mass_tolerance_ppm=precursor_mass_accuracy_ppm,
                isotopic_mass_tolerance_ppm=isotopic_mass_accuracy_ppm,
                minimum_intensity=isotopic_minimum_intensity,
                intensity_absolute_tolerance=isotopic_intensity_absolute_tolerance,
                intensity_relative_tolerance=isotopic_intensity_relative_tolerance,
                max_bounds=max_bounds,
            )
            .alias("bounds")
        )
        .filter(
            pl.col("bounds")
            .arr.min()
            .ge(
                0
            )  # Filter out rows where pattern deduction failed, which is signaled by negative bounds
        )
        .with_columns(
            pl.col("bounds")
            .arr.slice(0, length=NUM_ELEMENTS)
            .list.to_array(width=NUM_ELEMENTS)
            .alias("min_bounds"),
            pl.col("bounds")
            .arr.slice(NUM_ELEMENTS, length=NUM_ELEMENTS)
            .list.to_array(width=NUM_ELEMENTS)
            .alias("max_bounds"),
        )
    )
    # Mass decomposition
    chromatogram = (
        chromatogram.with_columns(
            pl.struct(  # type: ignore[missing-attribute]
                pl.col("Precursor_mz_MSDIAL").alias("mass"),
                pl.col("min_bounds"),
                pl.col("max_bounds"),
            )
            .mass_decomposition.decompose_mass_with_bounds(
                tolerance_ppm=precursor_mass_accuracy_ppm,
                dbe_mode="half_integer",
            )
            .alias("decomposed_formulas_struct")
        )
        .with_columns(
            pl.col("decomposed_formulas_struct")
            .struct.field("formulas")
            .alias("precursor_formula"),
            pl.col("decomposed_formulas_struct")
            .struct.field("formulas_str")
            .alias("precursor_formula_str"),
            pl.col("decomposed_formulas_struct")
            .struct.field("errors_ppm")
            .alias("precursor_errors_ppm"),
        )
        .drop(["bounds", "decomposed_formulas_struct"])
    )
    chromatogram = chromatogram.explode(
        ["precursor_formula", "precursor_formula_str", "precursor_errors_ppm"]
    )

    # Cleaning + normalization
    chromatogram = (
        chromatogram.rechunk()
        .with_columns(
            pl.struct(  # type: ignore[missing-attribute]
                pl.col("msms_m/z").alias("mz"),
                pl.col("msms_intensity").alias("intensities"),
                pl.col("precursor_formula"),
            )
            .mass_decomposition.clean_and_normalize_spectrum(
                raw_fragment_tolerance_ppm=fragment_mass_accuracy_ppm,
                normalized_fragment_tolerance_ppm=normalized_fragment_mass_accuracy_ppm,
                min_dbe=-0.5,
                max_dbe=30.0,
                dbe_mode="half_integer",
                water_absorption=False,
            )
            .alias("cleaned_spectra")
        )
        .with_columns(
            pl.col("cleaned_spectra")
            .struct.field("normalized_masses")
            .alias("cleaned_msms_mz"),
            pl.col("cleaned_spectra")
            .struct.field("intensities")
            .alias("cleaned_msms_intensity"),
            pl.col("cleaned_spectra")
            .struct.field("formulas")
            .alias("cleaned_spectrum_formulas"),
            pl.col("cleaned_spectra")
            .struct.field("formulas_str")
            .alias("cleaned_spectrum_formulas_str"),
            pl.col("cleaned_spectra")
            .struct.field("errors_ppm")
            .alias("cleaned_fragment_errors_ppm"),
        )
        .drop("cleaned_spectra")
    )

    return chromatogram


def _get_chromatogram_basic(path: str | Path) -> pl.DataFrame:
    chromatogram = pl.read_csv(
        source=path,
        has_header=True,
        skip_rows=0,
        separator="	",
        null_values="null",
        columns=list(MSDIAL_columns_to_read.keys()),  # type: ignore[no-matching-overload]
        schema_overrides=MSDIAL_columns_to_read,
    )
    # chromatogram = chromatogram.select(MSDIAL_columns_to_read.keys())
    chromatogram = _convert_spectra_to_list(chromatogram).drop(
        ["MSMS spectrum", "MS1 isotopes"]
    )
    chromatogram = chromatogram.with_columns(
        pl.col("RT right (min)").sub(pl.col("RT left(min)")).alias("peak_width_min"),
        pl.col("Precursor m/z").round(0).cast(pl.Int64).alias("nominal_mass"),
        pl.col("RT (min)").mul(60).round(0).cast(pl.Int64).alias("RT_(sec)"),
        pl.col("Precursor m/z").round(4).alias("Precursor m/z"),
    ).rename(
        {
            "Precursor m/z": "Precursor_mz_MSDIAL",
            "Adduct": "Precursor_type_MSDIAL",
        }
    )

    return chromatogram


def _add_energy_annotation(chromatogram: pl.DataFrame) -> pl.DataFrame:
    chromatogram_with_msms = chromatogram.filter(pl.col("msms_m/z").is_not_null())
    chromatogram_with_msms = chromatogram_with_msms.with_columns(  # get the index of the molecular ion, if it even exists
        molecular_ion_index=(pl.col("msms_m/z") - pl.col("Precursor_mz_MSDIAL"))
        .list.eval(pl.element().abs())
        .list.arg_min()
    )  # this will return an index even if there is no molecular ion.
    chromatogram_with_msms = chromatogram_with_msms.with_columns(
        molecular_ion_intensity=pl.when(
            (
                pl.col("msms_m/z").list.get(pl.col("molecular_ion_index"))
                - pl.col("Precursor_mz_MSDIAL")
            )
            < 0.003  # 3 mDa as the tolerance
        )
        .then(pl.col("msms_intensity").list.get(pl.col("molecular_ion_index")))
        .otherwise(pl.lit(0)),
        second_highest_intensity=pl.col("msms_intensity")
        .list.sort(descending=True, nulls_last=True)
        .list.get(1, null_on_oob=True)
        .fill_null(
            pl.lit(0)
        ),  # for cases where there is only one peak, we fill this value with 0
    )
    chromatogram_with_msms = chromatogram_with_msms.with_columns(
        pl.col("molecular_ion_intensity").le(0.1).alias("energy_is_too_high"),
        (
            pl.col("molecular_ion_intensity").eq(1)
            & pl.col("second_highest_intensity").le(0.2)
        ).alias("energy_is_too_low"),
    ).select(["Peak ID", "energy_is_too_high", "energy_is_too_low"])
    return chromatogram.join(other=chromatogram_with_msms, on="Peak ID", how="left")


def _convert_spectra_to_list(chromatogram: T) -> T:
    return cast(
        T,
        chromatogram.with_columns(
            pl.col("MSMS spectrum")
            .str.extract_all(pattern=r"(\d+\.\d+)")
            .list.eval(pl.element().cast(pl.Float64))
            .alias("msms_m/z"),
            pl.col("MSMS spectrum")
            .str.extract_all(pattern=r"(\d+)\s|(\d+$)")
            .list.eval(
                pl.element().str.extract(pattern=r"(\d+)").cast(pl.Float64).round(4)
            )
            .alias("msms_intensity"),
            pl.col("MS1 isotopes")
            .str.extract_all(pattern=r"(\d+\.\d+)")
            .list.eval(pl.element().cast(pl.Float64))
            .alias("ms1_isotopes_m/z"),
            pl.col("MS1 isotopes")
            .str.extract_all(pattern=r"(\d+)\s|(\d+$)")
            .list.eval(
                pl.element().str.extract(pattern=r"(\d+)").cast(pl.Float64).round(4)
            )
            .alias("ms1_isotopes_intensity"),
        ).with_columns(
            pl.col("msms_intensity").truediv(pl.col("msms_intensity").list.max())
        ),
    )


def _annotate_isobars_and_clean_spectrum(chromatogram: T) -> pl.DataFrame:
    chromatogram_lf = chromatogram.lazy()
    chromatogram_with_msms = chromatogram_lf.filter(
        pl.col("msms_intensity").is_not_null()
    )  # why? cause otherwise we don't know how to subtract spectrum

    isobars = chromatogram_with_msms.join_where(
        chromatogram_with_msms,
        # pl.col('Precursor_mz_MSDIAL').round(decimals=0).eq(pl.col('Precursor_mz_MSDIAL_isobar').round(decimals=0)),
        # pl.col('Precursor_mz_MSDIAL').round(decimals=0).cast(pl.UInt16).eq(pl.col('Precursor_mz_MSDIAL_isobar').round(decimals=0).cast(pl.UInt16)),
        pl.col("nominal_mass").eq(pl.col("nominal_mass_isobar")),
        pl.col("RT_(sec)")
        .sub(pl.col("RT_(sec)_isobar"))
        .abs()
        .le(pl.lit(6, dtype=pl.Int64)),  # less than 6 seconds of difference
        pl.col("Height")
        .truediv(pl.col("Height_isobar"))
        .le(pl.lit(3, dtype=pl.Int64)),  # the contaminant is at least a third as high
        pl.col("Peak ID").ne(
            pl.col("Peak ID_isobar")
        ),  # to prevent compunds from being the isobars of themselves
        suffix="_isobar",
    )

    isobars = isobars.group_by("Peak ID").all()
    isobars = isobars.with_columns(pl.col("Peak ID_isobar").alias("isobars"))
    isobars = isobars.select(["Peak ID", "isobars"])

    chromatogram_lf = chromatogram_lf.join(isobars, on="Peak ID", how="left")
    chromatogram_df = chromatogram_lf.collect()

    only_with_isobars = chromatogram_df.filter(pl.col("isobars").is_not_null())

    # ugly workaround. didn't find a better way.
    only_with_isobars_rows = only_with_isobars.select(
        ["Peak ID", "msms_m/z", "msms_intensity", "RT (min)", "isobars", "Height"]
    ).rows_by_key(key=["Peak ID"], named=True, unique=True)
    chromatogram_rows = chromatogram_df.rows_by_key(
        key=["Peak ID"], named=True, unique=True
    )
    for compound in only_with_isobars_rows:
        isobars = only_with_isobars_rows[compound]["isobars"]
        for isobar in isobars:
            (
                only_with_isobars_rows[compound]["msms_m/z"],
                only_with_isobars_rows[compound]["msms_intensity"],
            ) = _subtract_isobar_spectra(  # subtracts the second from the first
                only_with_isobars_rows[compound]["msms_m/z"],
                only_with_isobars_rows[compound]["msms_intensity"],
                only_with_isobars_rows[compound]["RT (min)"],
                only_with_isobars_rows[compound]["Height"],
                chromatogram_rows[isobar]["msms_m/z"],
                chromatogram_rows[isobar]["msms_intensity"],
                chromatogram_rows[isobar]["RT (min)"],
                chromatogram_rows[isobar]["Height"],
            )

    # this block just rearanges the data to a dict of {"Peak ID" : [the IDs], "data1":[the data] etc}
    cleaned_rows = []
    for ID, labels in only_with_isobars_rows.items():
        new_row = {"Peak ID": ID}
        new_row.update(labels)
        cleaned_rows.append(new_row)
    result_dict = {}
    for row in cleaned_rows:
        for key, value in row.items():
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)  # type: ignore[missing-attribute]

    chromatogram3 = pl.DataFrame(
        result_dict,
        schema_overrides={
            "Peak ID": pl.Int64,
            "msms_m/z": pl.List(pl.Float64),
            "msms_intensity": pl.List(pl.Float64),
        },
    )
    if (
        chromatogram3.is_empty()
    ):  # so if there are no isobars, we still have a dataframe
        chromatogram3 = pl.DataFrame(
            {"Peak ID": [], "msms_m/z": [], "msms_intensity": []},
            schema_overrides={
                "Peak ID": pl.Int64,
                "msms_m/z": pl.List(pl.Float64),
                "msms_intensity": pl.List(pl.Float64),
            },
        )
    chromatogram3 = chromatogram3.select(["Peak ID", "msms_m/z", "msms_intensity"])

    chromatogram_df = chromatogram_df.join(
        chromatogram3, on="Peak ID", how="left", suffix="_cleaned"
    )
    chromatogram_df = chromatogram_df.with_columns(  # converts empty lists to null
        pl.when(pl.col("msms_m/z_cleaned").list.len().gt(0)).then(
            pl.col("msms_m/z_cleaned")
        ),
        pl.when(pl.col("msms_intensity_cleaned").list.len().gt(0)).then(
            pl.col("msms_intensity_cleaned")
        ),
    )

    return chromatogram_df


def _subtract_isobar_spectra(
    compound_msms_mz,
    compound_msms_intensity,
    compound_RT,
    compound_height,
    isobar_msms_mz,
    isobar_msms_intensity,
    isobar_RT,
    isobar_height,
):
    rt_diff = compound_RT - isobar_RT
    coeff = np.exp(-np.power(rt_diff, 2) * 10) * (isobar_height / compound_height)
    coeff = np.full_like(isobar_msms_intensity, fill_value=coeff)
    adj_isobar_msms_intensity = np.multiply(coeff, isobar_msms_intensity)

    compound_spectra_dict = dict(zip(compound_msms_mz, compound_msms_intensity))
    isobar_spectra_dict = dict(zip(isobar_msms_mz, adj_isobar_msms_intensity))
    compound_spectra_dict = {
        mz: (compound_spectra_dict[mz] - isobar_spectra_dict.get(mz, 0))
        for mz in compound_spectra_dict.keys()
    }

    compound_spectra_dict = {
        mz: intensity
        for mz, intensity in compound_spectra_dict.items()
        if intensity > 0
    }

    compound_msms_mz = np.array(list(compound_spectra_dict.keys()), dtype=np.float64)
    compound_msms_intensity = np.array(
        list(compound_spectra_dict.values()), dtype=np.float64
    )

    return compound_msms_mz, compound_msms_intensity


# =============================================================================
# MS-DIAL Console App CLI Integration
# =============================================================================

# Base template for MS-DIAL parameters (without ion mode and adduct list)
_MSDIAL_PARAMS_TEMPLATE = """MS1 data type: Centroid
MS2 data type: Centroid
Ion mode: {ion_mode}
Target omics: Metabolomics
Ionization: ESI
Machine category: LCMS
Instrument type:
Instrument:
Authors:
License:
Comment:
Msp file path:
Lbm file path:
Text DB file path:
Isotope text DB file path:
Compounds library file path for target detection:
Compounds library file path for RT correction:

#Adduct ion setting
adduct list: {adduct_list}

# Export
Export spectra file format: msp
Export spectra type: deconvoluted
Mat file export folder path:
Export folder path:
Height matrix export: True
Normalized height matrix export: False
Representative spectra export: False
Peak ID matrix export: False
Retention time matrix export: False
Mass matrix export: False
MSMS included matrix export: False
Unique mass matrix export: False
Peak area matrix export: False
Parameter export: False
GNPS export: False
Molecular networking export: False
SN matrix export: False
Export as mztabM format: True


# Process parameters
Process option: All
Number of threads: {threads}


# Feature detection parameters
Smoothing method: LinearWeightedMovingAverage
Smoothing level: 3
Minimum peak height: {minimum_peak_height}
Minimum peak width: 5
Average peak width: 30
Mass slice width: 0.005
Retention time begin: 0
Retention time end: 100
MS1 mass range begin: 0
MS1 mass range end: 2000
MS2 mass range begin: 0
MS2 mass range end: 2000
MS1 tolerance for centroid: 0.01
MS2 tolerance for centroid: 0.025
Accuracy type: IsAccurate
Max charge number: 2
Considering Br and Cl for isotopes: True
Exclude mass list:
Max isotopes detected in ms1 spectrum: 3


# Deconvolution
Sigma window value: 0.5
Amplitude cut off: 1000
Keep isotope range: 5
Exclude after precursor: True
Keep original precursor isotopes: False
Is do andromeda ms2 deconvolution: False
Andromeda delta: 100
Andromeda max peaks: 12
Target CE: 0


# Retention time correction
Execute RT correction: False


# CorrDec settings
CorrDec execute: False
"""

_NEGATIVE_ADDUCTS = "[M-H]-,[M-H2O-H]-,[M+Na-2H]-,[M+Cl]-,[M+K-2H]-,[M+HCOO]-,[M+CH3COO]-,[M+C2H3N+Na-2H]-,[M+Br]-,[M+TFA-H]-,[M-C6H10O4-H]-,[M-C6H10O5-H]-,[M-C6H8O6-H]-,[M+CH3COONa-H]-,[2M-H]-,[2M+FA-H]-,[2M+Hac-H]-,[3M-H]-,[M-2H]2-,[M-3H]3-"

_POSITIVE_ADDUCTS = "[M+H]+,[M-H2O+H]+,[M+Na]+,[M+K]+,[M+NH4]+,[M+HCOO+Na]+,[M+CH3COO+Na]+,[M+CH3COO+K]+,[M+CH3COO+NH4]+,[M+HCOO+K]+,[M+HCOO+NH4]+,[2M+H]+,[2M+Na]+,[2M+K]+,[2M+NH4]+,[3M+H]+,[M+2H]2+,[M+3H]3+"


@dataclass
class MSDialRunnerConfig:
    """Configuration for running MS-DIAL Console App.

    Attributes
    ----------
    msdial_path : Path | None
        Path to the MS-DIAL executable. If None, will attempt auto-detection.
    threads : int
        Number of threads to use for processing (default: 20).
    minimum_peak_height : int
        Minimum peak height for feature detection (default: 100000).
    """

    msdial_path: Path | None = None
    threads: int = 20
    minimum_peak_height: int = 100000


def _get_project_root() -> Path | None:
    """Find the project root directory containing packages/msdial.
    
    Tries multiple strategies:
    1. Search upward from current file location
    2. Search upward from current working directory
    3. Check environment variable
    
    Returns
    -------
    Path | None
        Project root path if found, None otherwise.
    """
    # Strategy 1: Search upward from current file
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "packages" / "msdial").exists():
            return parent
    
    # Strategy 2: Search upward from current working directory
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "packages" / "msdial").exists():
            return parent
    
    # Strategy 3: Check environment variable
    env_root = Path(os.environ.get("HRMS_UTILS_ROOT", ""))
    if env_root.exists() and (env_root / "packages" / "msdial").exists():
        return env_root
    
    return None


def _find_msdial_executable() -> Path:
    """Find MS-DIAL Console App executable on the system.

    Searches common installation directories for both Windows and Linux.
    Handles permission errors gracefully by skipping inaccessible directories.

    Returns
    -------
    Path
        Path to the MS-DIAL executable.

    Raises
    ------
    FileNotFoundError
        If MS-DIAL executable cannot be found in any common location.
    """
    system = platform.system()
    
    # Try to find project root
    project_root = _get_project_root()

    if system == "Windows":
        executable_names = ["MsdialConsoleApp.exe", "MsdialConsoleApp.exe"]
        search_paths = [
            Path("C:/Program Files/MSDIAL"),
            Path("C:/Program Files (x86)/MSDIAL"),
            Path.home() / "AppData/Local/MSDIAL",
            Path.home() / "Documents/MSDIAL",
            Path.home() / "Downloads/MSDIAL",
            Path("./"),
        ]
    else:  # Linux and other Unix-like systems
        executable_names = ["MsdialConsoleApp", "MsdialConsoleApp.exe", "MSDIALCUI"]
        search_paths = [
            Path("/usr/local/bin"),
            Path("/opt/msdial"),
            Path("/opt/MSDIAL"),
            Path.home() / ".local/bin",
            Path.home() / "bin",
            Path("./"),
        ]
    
    # Insert project root path at the beginning if found
    if project_root is not None:
        msdial_package_dir = project_root / "packages" / "msdial"
        search_paths.insert(0, msdial_package_dir)

    for search_path in search_paths:
        try:
            if not search_path.exists():
                continue

            for exec_name in executable_names:
                candidate = search_path / exec_name
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()

            # Also search recursively one level deep in some directories
            if search_path in [Path("/opt"), Path.home() / "Downloads"]:
                try:
                    for subdir in search_path.iterdir():
                        if subdir.is_dir():
                            for exec_name in executable_names:
                                candidate = subdir / exec_name
                                if candidate.exists() and candidate.is_file():
                                    return candidate.resolve()
                except PermissionError:
                    continue

        except PermissionError:
            continue

    raise FileNotFoundError(
        f"MS-DIAL Console App executable not found. "
        f"Please install MS-DIAL or provide the path explicitly via MSDialRunnerConfig.msdial_path. "
        f"Searched in: {[str(p) for p in search_paths]}"
    )


def _generate_params_file(
    polarity: Literal["positive", "negative"],
    output_path: Path,
    threads: int = 20,
    minimum_peak_height: int = 100000,
) -> Path:
    """Generate MS-DIAL parameters file for the specified polarity.

    Parameters
    ----------
    polarity : Literal["positive", "negative"]
        Ionization polarity for the run.
    output_path : Path
        Directory where the parameters file will be saved.
    threads : int
        Number of threads for processing.
    minimum_peak_height : int
        Minimum peak height for feature detection.

    Returns
    -------
    Path
        Path to the generated parameters file.
    """
    if polarity == "positive":
        ion_mode = "Positive"
        adduct_list = _POSITIVE_ADDUCTS
    else:
        ion_mode = "Negative"
        adduct_list = _NEGATIVE_ADDUCTS

    params_content = _MSDIAL_PARAMS_TEMPLATE.format(
        ion_mode=ion_mode,
        adduct_list=adduct_list,
        threads=threads,
        minimum_peak_height=minimum_peak_height,
    )

    params_file = output_path / f"Msdial-lcms-dda-{polarity}-params.txt"
    params_file.write_text(params_content)

    return params_file


def _parse_msdial_errors(stdout: str, stderr: str) -> List[str]:
    """Parse MS-DIAL output for error patterns.

    Parameters
    ----------
    stdout : str
        Standard output from MS-DIAL process.
    stderr : str
        Standard error from MS-DIAL process.

    Returns
    -------
    List[str]
        List of detected error messages.
    """
    errors = []

    # Skip if this looks like help text
    combined_output_lower = (stdout + "\n" + stderr).lower()
    if "requires the following args" in combined_output_lower or "usage:" in combined_output_lower:
        return errors

    # Known error patterns in MS-DIAL output
    error_patterns = [
        "error",
        "exception",
        "failed",
        "failure",
        "cannot",
        "unable to",
        "not found",
        "invalid",
        "crash",
        "abort",
    ]

    combined_output = combined_output_lower

    for pattern in error_patterns:
        if pattern in combined_output:
            # Extract lines containing the error pattern
            for line in (stdout + "\n" + stderr).split("\n"):
                if pattern in line.lower() and line.strip():
                    errors.append(line.strip())

    # Remove duplicates while preserving order
    seen = set()
    unique_errors = []
    for error in errors:
        if error not in seen:
            seen.add(error)
            unique_errors.append(error)

    return unique_errors


def run_msdial_lcmsdda(
    input_dir: Path | str,
    output_dir: Path | str,
    polarity: Literal["positive", "negative"],
    params_file: Path | str | None = None,
    config: MSDialRunnerConfig | None = None,
) -> subprocess.CompletedProcess:
    """Run MS-DIAL LCMS-DDA analysis via command line interface.

    Parameters
    ----------
    input_dir : Path | str
        Directory containing input raw data files.
    output_dir : Path | str
        Directory where results will be saved.
    polarity : Literal["positive", "negative"]
        Ionization polarity for the run.
    params_file : Path | str | None
        Path to custom MS-DIAL parameters file. If None, a parameters file
        will be generated based on polarity and saved to output_dir.
    config : MSDialRunnerConfig | None
        Configuration object. If None, uses default configuration with
        auto-detection of MS-DIAL executable.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process object containing return code and output.

    Raises
    ------
    FileNotFoundError
        If MS-DIAL executable cannot be found.
    RuntimeError
        If MS-DIAL execution fails or reports errors.
    ValueError
        If invalid polarity is specified.

    Examples
    --------
    >>> result = run_msdial_lcmsdda(
    ...     input_dir="/path/to/raw/data",
    ...     output_dir="/path/to/output",
    ...     polarity="negative",
    ... )
    >>> print(result.returncode)
    0

    >>> config = MSDialRunnerConfig(
    ...     msdial_path=Path("/custom/path/MsdialConsoleApp.exe"),
    ...     threads=10,
    ... )
    >>> result = run_msdial_lcmsdda(
    ...     input_dir="/path/to/raw/data",
    ...     output_dir="/path/to/output",
    ...     polarity="positive",
    ...     config=config,
    ... )
    """
    if polarity not in ("positive", "negative"):
        raise ValueError(f"polarity must be 'positive' or 'negative', got '{polarity}'")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine MS-DIAL executable path
    if config is None:
        config = MSDialRunnerConfig()

    if config.msdial_path is None:
        msdial_executable = _find_msdial_executable()
    else:
        msdial_executable = Path(config.msdial_path)
        if not msdial_executable.exists():
            raise FileNotFoundError(
                f"Specified MS-DIAL executable not found: {msdial_executable}"
            )

    # Generate or use provided parameters file
    if params_file is None:
        params_path = _generate_params_file(
            polarity=polarity,
            output_path=output_dir,
            threads=config.threads,
            minimum_peak_height=config.minimum_peak_height,
        )
    else:
        params_path = Path(params_file)
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_path}")

    # Construct command
    cmd = [
        str(msdial_executable),
        "lcms",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-m",
        str(params_path),
    ]

    # Run MS-DIAL
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    # Parse for errors
    detected_errors = _parse_msdial_errors(result.stdout, result.stderr)

    if result.returncode != 0 or detected_errors:
        error_msg = f"MS-DIAL execution failed with return code {result.returncode}"
        if detected_errors:
            error_msg += f". Detected errors: {'; '.join(detected_errors[:5])}"
        raise RuntimeError(error_msg)

    return result


if __name__ == "__main__":
    start = time()
    pl.Config(tbl_rows=20, tbl_cols=15)
    path = Path(r"/home/analytit_admin/Data/iibr_data/250515_006.txt")
    blank_path = Path(r"/home/analytit_admin/Data/iibr_data/250515_003.txt")
    chromatogram = get_chromatogram(path=path)
    blank = get_chromatogram(path=blank_path)

    # if isinstance(chromatogram,pl.LazyFrame):
    #     print(chromatogram.collect_schema())
    #     print(chromatogram.collect())
    # elif isinstance(chromatogram,pl.DataFrame):
    #     print(chromatogram.schema)
    #     print(chromatogram)
    # else:
    #     print("wrong output! this must be either a polars lazyframe or dataframe")
    #     print(type(chromatogram))
    chromatogram = subtract_blank_frame(
        sample_df=chromatogram, blank_df=blank, config=blank_config()
    )
    print(chromatogram.head(10))

    print(time() - start)
