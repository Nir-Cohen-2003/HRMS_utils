import polars as pl
import numpy as np
from pathlib import Path
from time import time
from ms_entropy import calculate_spectral_entropy, calculate_entropy_similarity
from dataclasses import dataclass
from typing import List, Tuple, Dict
from numba import  jit
from ..formula_annotation.isotopic_pattern import deduce_isotopic_pattern
from ..formula_annotation.mass_decomposition import decompose_mass_per_bounds_verbose, clean_and_normalize_spectra_known_precursor_verbose, NUM_ELEMENTS
from ..formula_annotation.element_table import ELEMENT_INDEX, ELEMENT_MASSES

PROTON_MASS = ELEMENT_MASSES[ELEMENT_INDEX['H']]


MSDIAL_columns_to_read = {
    'Peak ID': pl.Int64,
    'Scan': pl.Int64,
    'RT left(min)': pl.Float64, 
    'RT (min)': pl.Float64, 
    'RT right (min)': pl.Float64,
    'Precursor m/z': pl.Float64,
    'Height': pl.Float64, 
    'Adduct': pl.String,
    'Isotope': pl.String, 
    'MSMS spectrum': pl.String, # will be converted to 2 lists, m/z and intensity
    'MS1 isotopes': pl.String, # will be converted to 2 lists, m/z and intensity
}

MSDIAL_other_columns = [
    'Estimated noise', 'S/N',
    'Sharpness', 'Gaussian similarity', 'Ideal slope', 'Symmetry', 'MS1 isotopes' #'S/N', (second S/N has the same values as the first one.)
]


MSDIAL_columns_to_output= [
    'Peak ID',
    'RT (min)',
    'Precursor_mz_MSDIAL',
    'Height', 
    'Precursor_type_MSDIAL', 
    'msms_m/z', 'msms_intensity', 
    'isobars',
    'msms_m/z_cleaned', 'msms_intensity_cleaned',
    'spectral_entropy',
    'energy_is_too_low', 'energy_is_too_high',
    'ms1_isotopes_m/z', 'ms1_isotopes_intensity'
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
    noise_threshold: float = 0.005    # new field

    def __post_init__(self):
        if self.ms1_mass_tolerance > 0.0001:  # if the value is more than 0.0001, its a ppm value and we multiply by 1e-6
            self.ms1_mass_tolerance = self.ms1_mass_tolerance * 1e-6
        if self.use_ms2:
            # Ensure ms2_mass_tolerance and noise_threshold are set if use_ms2 is True
            if not hasattr(self, 'ms2_mass_tolerance') or self.ms2_mass_tolerance is None:
                self.ms2_mass_tolerance = 5e-6
            if not hasattr(self, 'noise_threshold') or self.noise_threshold is None:
                self.noise_threshold = 0.005

    def to_dict(self) -> dict:
        return {
            'ms1_mass_tolerance': self.ms1_mass_tolerance,
            'dRT_min': self.dRT_min,
            'ratio': self.ratio,
            'use_ms2': self.use_ms2,
            'dRT_min_with_ms2': self.dRT_min_with_ms2,
            'ms2_fit': self.ms2_fit,
            'ms2_mass_tolerance': self.ms2_mass_tolerance,
            'noise_threshold': self.noise_threshold
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'blank_config':
        kwargs = {}
        for field_ in cls.__dataclass_fields__: # why complicate it and not just use cls(**config_dict)? because we want to avoid keys that are not in the dataclass
            if field_ in config_dict and config_dict[field_] is not None:
                kwargs[field_] = config_dict[field_]
        return cls(**kwargs)



def get_chromatogram(path: str | Path)-> pl.DataFrame :
    '''Reads the .txt output of a complete chromatogram from MSDIAL (note- use the "trim content fo excel option), and returns a polars dataframe with the following schema:
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
        spectral_entropy: pl.Float32
        energy_is_too_low: pl.Boolean
        energy_is_too_high: pl.Boolean
        ms1_isotopes_m/z: pl.List(pl.Float64)
        ms1_isotopes_intensity: pl.List(pl.Float64)
    '''
    chromatogram = _get_chromatogram_basic(path=path)
    chromatogram = _annotate_isobars_and_clean_spectrum(chromatogram=chromatogram)
    chromatogram = _add_energy_annotation(chromatogram=chromatogram)
    chromatogram = _add_entropy(chromatogram=chromatogram)
    chromatogram = chromatogram.select(MSDIAL_columns_to_output)
    if not isinstance(chromatogram,pl.DataFrame):
        raise Exception("failed getting chromatogram from the file: " + str(path))
    
    return chromatogram


def subtract_blank_frame(
        sample_df: pl.DataFrame, 
        blank_df: pl.DataFrame, 
        config:blank_config) -> pl.DataFrame:
    '''subtracts a blank chromatogram, using ms1, ms2 and RT. 
    in absense of ms2 for either the blank or the sample compound, a stricter rt threshold is used. 
    keep dRT_min_with_ms2 > dRT_min, or the logic gets wrong.'''

    if not config.use_ms2: #so when both sample and blank spectra has msms, we require a 0.85 fit on ms2, but lower fit on rt. if any of them lacks ms2, we just use strict rt.
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
            pl.col("RT (min)") > pl.col("RT (min)_blank") -  config.dRT_min,
            (pl.col("Precursor_mz_MSDIAL").truediv(pl.col("Precursor_mz_MSDIAL_blank"))-1.0).abs().le(config.ms1_mass_tolerance),
            pl.col("Height") <  pl.col("Height_blank") * config.ratio,
            suffix="_blank"
        ).collect(engine="streaming")
    else: # so we just use strict rt
        sample_lf = sample_df.select(
            [
                "Peak ID",
                "RT (min)",
                "Precursor_mz_MSDIAL",
                "Height",
                'msms_m/z',
                'msms_intensity'
            ]
        ).lazy()
        blank_lf = blank_df.select(
            [
                "RT (min)",
                "Precursor_mz_MSDIAL",
                "Height",
                'msms_m/z',
                'msms_intensity'
            ]
        ).lazy()
        subtract_lf = sample_lf.join_where(
            blank_lf,
            pl.col("RT (min)") < pl.col("RT (min)_blank") + config.dRT_min_with_ms2,
            pl.col("RT (min)") > pl.col("RT (min)_blank") - config.dRT_min_with_ms2,
            (pl.col("Precursor_mz_MSDIAL").truediv(pl.col("Precursor_mz_MSDIAL_blank"))-1).abs().le(config.ms1_mass_tolerance),
            pl.col("Height") <  pl.col("Height_blank") * config.ratio,
            suffix="_blank"
        )
        subtract_lf_rt_strict = subtract_lf.filter(
            pl.col('msms_m/z').is_null() | 
            pl.col('msms_m/z_blank').is_null()
        )
        subtract_lf_rt_strict = subtract_lf_rt_strict.filter(
            pl.col("RT (min)") < pl.col("RT (min)_blank") + config.dRT_min,
            pl.col("RT (min)") > pl.col("RT (min)_blank") - config.dRT_min
        )

        subtract_df_ms2 = subtract_lf.filter(
            pl.col('msms_m/z').is_not_null(),
            pl.col('msms_m/z_blank').is_not_null()
        ).collect(engine="streaming")
        
        subtract_df_ms2 = subtract_df_ms2.filter(
            pl.struct(
                pl.col('msms_intensity'),
                pl.col('msms_m/z'),
                pl.col('msms_intensity_blank'),
                pl.col('msms_m/z_blank')
            ).map_batches(
                lambda spectra: _entropy_score_batch(
                    spectra.struct.field('msms_m/z').to_numpy(),
                    spectra.struct.field('msms_intensity').to_numpy(),
                    spectra.struct.field('msms_m/z_blank').to_numpy(),
                    spectra.struct.field('msms_intensity_blank').to_numpy(),
                    config
                    ),
                return_dtype=pl.Float64,
                is_elementwise=True
            ).ge(config.ms2_fit))
        
        subtract_df = pl.concat([subtract_df_ms2,subtract_lf_rt_strict.collect(engine="streaming")])

    cleaned_sample_df = sample_df.join(subtract_df,on="Peak ID", how='anti')
    return cleaned_sample_df


PROTON_MASS = ELEMENT_MASSES[ELEMENT_INDEX['H']]

def annotate_chromatogram_with_formulas(
    chromatogram: pl.DataFrame,
    addcut_mass: float = PROTON_MASS,
    max_bounds: dict|None = None,
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
    - decomposed_formulas: List(Array(Int32, shape=(NUM_ELEMENTS,)))
        Candidate elemental formula(s) for the precursor (each is an integer vector of
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
      decompose_mass_per_bounds and clean_and_normalize_spectra_known_precursor. Any
      change in those APIs must be propagated here.
    - One input precursor row may expand into multiple output rows (one per candidate
      decomposition) because of the explosion of "decomposed_formulas".
    - Mass tolerances are expressed in ppm; callers should pass values appropriate
      for their instrument and data quality.
    - The function performs no downstream filtering of candidate formulas; downstream
      ranking/selection is the caller's responsibility.
    """
    # Isotopic pattern deduction
    chromatogram = chromatogram.with_columns(
        pl.struct(
            ["Precursor_mz_MSDIAL", "ms1_isotopes_m/z", "ms1_isotopes_intensity"]
        ).map_batches(
            lambda batch: deduce_isotopic_pattern(
                batch.struct.field("Precursor_mz_MSDIAL"),
                batch.struct.field("ms1_isotopes_m/z"),
                batch.struct.field("ms1_isotopes_intensity"),
                ms1_mass_tolerance_ppm=precursor_mass_accuracy_ppm,
                isotopic_mass_tolerance_ppm=isotopic_mass_accuracy_ppm,
                minimum_intensity=isotopic_minimum_intensity,
                intensity_absolute_tolerance=isotopic_intensity_absolute_tolerance,
                intensity_relative_tolerance=isotopic_intensity_relative_tolerance,
                max_bounds=max_bounds,
            ),
            return_dtype=pl.Array(inner=pl.Int32, shape=(2*NUM_ELEMENTS,))
        ).alias("bounds")
    ).with_columns(
        pl.col("bounds").arr.slice(0, length=NUM_ELEMENTS).list.to_array(width=NUM_ELEMENTS).alias("min_bounds"),
        pl.col("bounds").arr.slice(NUM_ELEMENTS, length=NUM_ELEMENTS).list.to_array(width=NUM_ELEMENTS).alias("max_bounds")
    )

    # Mass decomposition
    chromatogram = chromatogram.with_columns(
        non_ionized_mass = pl.col("Precursor_mz_MSDIAL") - addcut_mass
    ).with_columns(
        pl.struct(
            ["non_ionized_mass", "min_bounds", "max_bounds"]
        ).map_batches(
            lambda batch: decompose_mass_per_bounds_verbose(
                batch.struct.field("non_ionized_mass"),
                batch.struct.field("min_bounds"),
                batch.struct.field("max_bounds"),
                tolerance_ppm=precursor_mass_accuracy_ppm,
            ),
            return_dtype=pl.Struct({
                "decomposed_formulas": pl.List(pl.Array(inner=pl.Int32, shape=(NUM_ELEMENTS,))),
                "decomposed_formulas_str": pl.List(pl.String),
            })
        ).alias("decomposed_formulas_struct")).with_columns(
            pl.col("decomposed_formulas_struct").struct.unnest()
        ).drop(["bounds", "decomposed_formulas_struct"])

    chromatogram = chromatogram.with_columns(pl.col("msms_m/z").sub(addcut_mass).alias("non_ionized_msms_m/z"))
    chromatogram = chromatogram.explode(["decomposed_formulas", "decomposed_formulas_str"])
    
    # Cleaning + normalization (updated API requires observed precursor mass series)
    chromatogram = chromatogram.with_columns(
        pl.struct(["decomposed_formulas", "non_ionized_mass", "non_ionized_msms_m/z", "msms_intensity"]).map_batches(
            lambda batch: clean_and_normalize_spectra_known_precursor_verbose(
                precursor_formula_series=batch.struct.field("decomposed_formulas"),
                precursor_masses_series=batch.struct.field("non_ionized_mass"),
                fragment_masses_series=batch.struct.field("non_ionized_msms_m/z"),
                fragment_intensities_series=batch.struct.field("msms_intensity"),
                tolerance_ppm=fragment_mass_accuracy_ppm,
                max_allowed_normalized_mass_error_ppm=normalized_fragment_mass_accuracy_ppm,
            ),
            return_dtype=pl.Struct({
                "masses_normalized": pl.List(pl.Float64),
                "cleaned_intensities": pl.List(pl.Float64),
                "fragment_formulas": pl.List(pl.Array(inner=pl.Int32, shape=(NUM_ELEMENTS,))),
                "fragment_formulas_str": pl.List(pl.String),
                "fragment_errors_ppm": pl.List(pl.Float64),
            }),
        ).alias("cleaned_spectra")
    ).with_columns(
        pl.col("cleaned_spectra").struct.unnest()
    ).rename(
        {
            "masses_normalized": "cleaned_msms_mz",
            "cleaned_intensities": "cleaned_msms_intensity",
            "fragment_formulas": "cleaned_spectrum_formulas",
            "fragment_formulas_str": "cleaned_spectrum_formulas_str",
            "fragment_errors_ppm": "cleaned_fragment_errors_ppm",
        }
    ).drop(
        "cleaned_spectra"
    )
    return chromatogram

def _entropy_score(
        spec1_mz:np.ndarray, spec1_intensity:np.ndarray,
        spec2_mz:np.ndarray, spec2_intensity:np.ndarray) -> np.float64:
    if any(x is None for x in [spec1_mz,spec2_mz,spec1_intensity,spec2_intensity]):
        return -1
    spec1 = np.column_stack((spec1_mz,spec1_intensity))
    spec1 = np.array(spec1,dtype=np.float32)
    spec2 = np.column_stack((spec2_mz,spec2_intensity))
    spec2 = np.array(spec2,dtype=np.float32)
    score = calculate_entropy_similarity(
        spec1,spec2,
        ms2_tolerance_in_ppm=config.ms2_mass_tolerance*10e6,
        clean_spectra=True,
        noise_threshold=config.noise_threshold)
    score = np.float64(score)
    return score
_entropy_score_batch=np.vectorize(_entropy_score)


def _get_chromatogram_basic(path: str | Path)-> pl.LazyFrame :
    chromatogram = pl.read_csv(
        source=path,has_header=True,skip_rows=0,separator="	", null_values='null',
        columns=list(MSDIAL_columns_to_read.keys()),
        schema_overrides=MSDIAL_columns_to_read)
    # chromatogram = chromatogram.select(MSDIAL_columns_to_read.keys())
    chromatogram = _convert_MSMS_to_list(chromatogram).drop('MSMS spectrum')
    chromatogram = _convert_MS1_to_list(chromatogram).drop('MS1 isotopes')
    chromatogram=chromatogram.with_columns(
        pl.col('RT right (min)').sub(pl.col('RT left(min)')).alias('peak_width_min'),
        pl.col('Precursor m/z').round(0).cast(pl.Int64).alias('nominal_mass'),
        pl.col('RT (min)').mul(60).round(0).cast(pl.Int64).alias('RT_(sec)'),
        pl.col('Precursor m/z').round(4).alias('Precursor m/z'),
        
    ).rename(
        {
        'Precursor m/z':'Precursor_mz_MSDIAL',
        'Adduct':'Precursor_type_MSDIAL', 
        }
    )
    
    return chromatogram

def _add_energy_annotation(chromatogram:pl.DataFrame) -> pl.DataFrame:
    chromatogram_with_msms = chromatogram.filter(pl.col('msms_m/z').is_not_null())
    chromatogram_with_msms = chromatogram_with_msms.with_columns( # get the index of the molecular ion, if it even exists
        molecular_ion_index=(pl.col('msms_m/z')-pl.col('Precursor_mz_MSDIAL')).list.eval(pl.element().abs()).list.arg_min()
    ) #this will return an index even if there is no molecular ion.
    chromatogram_with_msms = chromatogram_with_msms.with_columns(
        molecular_ion_intensity=pl.when(
            (pl.col('msms_m/z').list.get(pl.col('molecular_ion_index')) - pl.col('Precursor_mz_MSDIAL'))<0.003 # 3 mDa as the tolerance
        ).then(pl.col('msms_intensity').list.get(pl.col('molecular_ion_index'))).otherwise(pl.lit(0)),
        second_highest_intensity=pl.col('msms_intensity').list.sort(descending=True,nulls_last=True).list.get(1,null_on_oob=True).fill_null(pl.lit(0))#for cases where there is only one peak, we fill this value with 0 
    )
    chromatogram_with_msms = chromatogram_with_msms.with_columns(
        pl.col('molecular_ion_intensity').le(0.1).alias('energy_is_too_high'),
        (pl.col('molecular_ion_intensity').eq(1)&pl.col('second_highest_intensity').le(0.2)).alias('energy_is_too_low')
    ).select(['Peak ID','energy_is_too_high','energy_is_too_low'])
    return chromatogram.join(other=chromatogram_with_msms,on='Peak ID',how='left')

def _add_entropy(chromatogram:pl.DataFrame)-> pl.DataFrame:
    chromatogram = chromatogram.with_columns(
        pl.struct(
            pl.col('msms_m/z'),
            pl.col('msms_intensity')
        ).map_batches(
            lambda spectra: _calculate_spectral_entropy_wrapper_batch(
                spectra.struct.field('msms_m/z').to_numpy(),
                spectra.struct.field('msms_intensity').to_numpy()),
            return_dtype=pl.Float32,
            is_elementwise=True
        ).alias('spectral_entropy')
    )
    return chromatogram

def _calculate_spectral_entropy_wrapper(mz,intesity):
    spectrum = np.column_stack((mz,intesity))
    spectrum = np.array(spectrum,dtype=np.float32)
    return calculate_spectral_entropy(spectrum)
_calculate_spectral_entropy_wrapper_batch=np.vectorize(_calculate_spectral_entropy_wrapper)

def _convert_MSMS_to_list(chromatogram: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:

    chromatogram = chromatogram.with_columns(
        pl.col('MSMS spectrum').str.extract_all(
            pattern=r'(\d+\.\d+)'
        ).list.eval(pl.element().cast(pl.Float64)).alias('msms_m/z'),
        pl.col('MSMS spectrum').str.extract_all(
            pattern=r'(\d+)\s|(\d+$)'
        ).list.eval(pl.element().str.extract( pattern=r'(\d+)').cast(pl.Float64).round(4)).alias('msms_intensity')
        #).alias('msms_intensity')
    )
    chromatogram = chromatogram.with_columns(
        pl.col('msms_intensity').truediv(pl.col('msms_intensity').list.max())
    )

    return chromatogram

def _convert_MS1_to_list(chromatogram: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:

    chromatogram = chromatogram.with_columns(
        pl.col('MS1 isotopes').str.extract_all(
            pattern=r'(\d+\.\d+)'
        ).list.eval(pl.element().cast(pl.Float64)).alias('ms1_isotopes_m/z'),
        pl.col('MS1 isotopes').str.extract_all(
            pattern=r'(\d+)\s|(\d+$)'
        ).list.eval(pl.element().str.extract( pattern=r'(\d+)').cast(pl.Float64).round(4)).alias('ms1_isotopes_intensity')
    )
    # removed because we need to know the actual intensity of each, not only the relative.
    # chromatogram = chromatogram.with_columns(
    #     pl.col('ms1_isotopes_intensity').truediv(
    #         pl.col('ms1_isotopes_intensity').list.get(
    #             pl.col('ms1_isotopes_m/z').sub(pl.col('Precursor m/z')).list.eval(pl.element().abs()).list.arg_min()
    #         )
    #     )
    # )
    return chromatogram


def _annotate_isobars_and_clean_spectrum(chromatogram: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    chromatogram = chromatogram.lazy()
    chromatogram_with_msms = chromatogram.filter(pl.col('msms_intensity').is_not_null()) #why? cause otherwise we don't know how to subtract spectrum

    
    isobars = chromatogram_with_msms.join_where(
        chromatogram_with_msms,
        # pl.col('Precursor_mz_MSDIAL').round(decimals=0).eq(pl.col('Precursor_mz_MSDIAL_isobar').round(decimals=0)),
        # pl.col('Precursor_mz_MSDIAL').round(decimals=0).cast(pl.UInt16).eq(pl.col('Precursor_mz_MSDIAL_isobar').round(decimals=0).cast(pl.UInt16)),
        pl.col('nominal_mass').eq(pl.col('nominal_mass_isobar')),
        pl.col('RT_(sec)').sub(pl.col('RT_(sec)_isobar')).abs().le(pl.lit(6,dtype=pl.Int64)), #less than 6 seconds of difference
        pl.col('Height').truediv(pl.col('Height_isobar')).le(pl.lit(3,dtype=pl.Int64)), #the contaminant is at least a third as high
        pl.col('Peak ID').ne(pl.col('Peak ID_isobar')) # to prevent compunds from being the isobars of themselves
        ,suffix='_isobar'
        )

    isobars = isobars.group_by('Peak ID').all()
    isobars = isobars.with_columns(pl.col('Peak ID_isobar').alias('isobars'))
    isobars = isobars.select(['Peak ID','isobars'])

    chromatogram = chromatogram.join(isobars,on='Peak ID',how='left')
    chromatogram = chromatogram.collect()
    
    only_with_isobars = chromatogram.filter(pl.col('isobars').is_not_null())

    # ugly workaround. didn't find a better way.
    only_with_isobars_rows = only_with_isobars.select(['Peak ID','msms_m/z','msms_intensity','RT (min)','isobars','Height']).rows_by_key(key=['Peak ID'],named=True,unique=True)
    chromatogram_rows = chromatogram.rows_by_key(key=['Peak ID'],named=True,unique=True)

    for compound in only_with_isobars_rows:
        isobars = only_with_isobars_rows[compound]['isobars']
        for isobar in isobars:
            only_with_isobars_rows[compound]['msms_m/z'], only_with_isobars_rows[compound]['msms_intensity'] = _subtract_isobar_spectra( # subtracts the second from the first
                only_with_isobars_rows[compound]['msms_m/z'], 
                only_with_isobars_rows[compound]['msms_intensity'],
                only_with_isobars_rows[compound]['RT (min)'], 
                only_with_isobars_rows[compound]['Height'],
                chromatogram_rows[isobar]['msms_m/z'],
                chromatogram_rows[isobar]['msms_intensity'],
                chromatogram_rows[isobar]['RT (min)'],
                chromatogram_rows[isobar]['Height']
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
            result_dict[key].append(value)

    chromatogram3 = pl.DataFrame(
        result_dict,
        schema_overrides={
            'Peak ID': pl.Int64,
            'msms_m/z': pl.List(pl.Float64),
            'msms_intensity': pl.List(pl.Float64),
        })
    if chromatogram3.is_empty(): # so if there are no isobars, we still have a dataframe
        chromatogram3 = pl.DataFrame(
            {'Peak ID': [], 'msms_m/z': [], 'msms_intensity': []},
            schema_overrides={
                'Peak ID': pl.Int64,
                'msms_m/z': pl.List(pl.Float64),
                'msms_intensity': pl.List(pl.Float64),
            }
        )
    chromatogram3 = chromatogram3.select(['Peak ID','msms_m/z','msms_intensity'])

    chromatogram=chromatogram.join(chromatogram3, on="Peak ID",how="left", suffix="_cleaned")
    chromatogram = chromatogram.with_columns( #converts empty lists to null
        pl.when(
        pl.col('msms_m/z_cleaned').list.len().gt(0)
        ).then(pl.col('msms_m/z_cleaned')),
        pl.when(
        pl.col('msms_intensity_cleaned').list.len().gt(0)
        ).then(pl.col('msms_intensity_cleaned'))) 
    

    return chromatogram

def _subtract_isobar_spectra(
        compound_msms_mz,compound_msms_intensity,
        compound_RT, compound_height,
        isobar_msms_mz,isobar_msms_intensity,
        isobar_RT,isobar_height):

    rt_diff= compound_RT - isobar_RT
    coeff = np.exp(-np.power(rt_diff,2)*10) *(isobar_height/compound_height)
    coeff = np.full_like(isobar_msms_intensity,fill_value=coeff)
    adj_isobar_msms_intensity = np.multiply(coeff,isobar_msms_intensity)

    compound_spectra_dict = dict(zip(compound_msms_mz,compound_msms_intensity))
    isobar_spectra_dict = dict(zip(isobar_msms_mz,adj_isobar_msms_intensity))
    compound_spectra_dict = {mz: (compound_spectra_dict[mz] - isobar_spectra_dict.get(mz,0)) for mz in compound_spectra_dict.keys()}
    
    compound_spectra_dict = {mz: intensity for mz, intensity in compound_spectra_dict.items() if intensity > 0 }

    compound_msms_mz = np.array(list(compound_spectra_dict.keys()),dtype=np.float64)
    compound_msms_intensity = np.array(list(compound_spectra_dict.values()),dtype=np.float64)

    return compound_msms_mz,compound_msms_intensity
    



if __name__ == '__main__':
    start = time()
    pl.Config(
    tbl_rows=20,
    tbl_cols=15)
    path = Path(r'/home/analytit_admin/Data/iibr_data/250515_006.txt')
    blank_path = Path(r'/home/analytit_admin/Data/iibr_data/250515_003.txt')
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
        sample_df=chromatogram,
        blank_df=blank,
        config=blank_config()
    )
    print(chromatogram.head(10))

    print(time()-start)
