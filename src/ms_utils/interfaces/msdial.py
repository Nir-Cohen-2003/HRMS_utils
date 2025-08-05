import polars as pl
import numpy as np
from pathlib import Path
from time import time
from ms_entropy import calculate_spectral_entropy
from dataclasses import dataclass
from ..pyscreen.spectral_search import entropy_score_batch # used to compare chromatograms against each other, which is a case of spectral search

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
    ms1_mass_tolerance: float=3e-6
    dRT_min : float=0.1
    ratio : float | int=5
    use_ms2:bool=False
    dRT_min_with_ms2:float=0.3
    ms2_fit:float=0.85
    def __post_init__(self):
        if self.ms1_mass_tolerance > 0.0001: # if the value is more than 0.0001, its a ppm value and we multiply by 1e-6
            self.ms1_mass_tolerance = self.ms1_mass_tolerance * 1e-6
            
    def to_dict(self) -> dict:
        return {
            'ms1_mass_tolerance': self.ms1_mass_tolerance,
            'dRT_min': self.dRT_min,
            'ratio': self.ratio,
            'use_ms2': self.use_ms2,
            'dRT_min_with_ms2': self.dRT_min_with_ms2,
            'ms2_fit': self.ms2_fit
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'blank_config':
        kwargs = {}
        for field_ in cls.__dataclass_fields__: # why complicate it and not just use cls(**config_dict)? because we want to avoid keys that are not in the dataclass
            if field_ in config_dict and config_dict[field_] is not None:
                kwargs[field_] = config_dict[field_]
        return cls(**kwargs)



def get_chromatogram(path: str | Path)-> pl.DataFrame :
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
                lambda spectra: entropy_score_batch(
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
            lambda spectra: calculate_spectral_entropy_wrapper_batch(
                spectra.struct.field('msms_m/z').to_numpy(),
                spectra.struct.field('msms_intensity').to_numpy()),
            return_dtype=pl.Float32,
            is_elementwise=True
        ).alias('spectral_entropy')
    )
    return chromatogram

def calculate_spectral_entropy_wrapper(mz,intesity):
    spectrum = np.column_stack((mz,intesity))
    spectrum = np.array(spectrum,dtype=np.float32)
    return calculate_spectral_entropy(spectrum)
calculate_spectral_entropy_wrapper_batch=np.vectorize(calculate_spectral_entropy_wrapper)

def _convert_MSMS_to_list(chromatogram: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:

    chromatogram = chromatogram.with_columns(
        pl.col('MSMS spectrum').str.extract_all(
            pattern=r'(\d+\.\d+)'
        ).list.eval(pl.element().str.to_decimal().cast(pl.Float64)).alias('msms_m/z'),
        pl.col('MSMS spectrum').str.extract_all(
            pattern=r'(\d+)\s|(\d+$)'
        ).list.eval(pl.element().str.extract( pattern=r'(\d+)').str.to_decimal().cast(pl.Float64).round(4)).alias('msms_intensity')
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
        ).list.eval(pl.element().str.to_decimal().cast(pl.Float64)).alias('ms1_isotopes_m/z'),
        pl.col('MS1 isotopes').str.extract_all(
            pattern=r'(\d+)\s|(\d+$)'
        ).list.eval(pl.element().str.extract( pattern=r'(\d+)').str.to_decimal().cast(pl.Float64).round(4)).alias('ms1_isotopes_intensity')
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
