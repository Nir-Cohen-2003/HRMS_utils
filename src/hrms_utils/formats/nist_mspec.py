import re
import polars as pl
# import numpy as np
import polars.selectors as plcs
from typing import TypeVar, cast, Dict,Iterable
from ..formula_annotation.utils import formula_fits_mass, format_formula_string_to_array,  get_precursor_ion_formula_array, num_elements
from ..formula_annotation.element_table import ADDUCT_MASSES
import mass_decomposition
import spectral_similarity 
from pathlib import Path 
from scipy.stats import linregress

T = TypeVar('T', pl.DataFrame, pl.LazyFrame)


def create_nist_dataframe(named_file_list: list[tuple[str|Path, str]]) -> pl.DataFrame:
    '''takes a list of tuples with the first element being the path to the file and the second being the to write as "DB_Name" column, and returns a polars DataFrame with the data from all files'''
    for file_path, db_name in named_file_list:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        if not file_path.is_file():
            raise ValueError(f"Path {file_path} is not a file.")
        # make sure the file is a MSPEC, mspec, MSP or msp file
        if file_path.suffix.lower() not in ['.mspec', '.msp']:
            raise ValueError(f"File {file_path} is not a MSPEC or MSP file.")
    dataframes = []
    for file_path, db_name in named_file_list:
        df = read_MSPEC_file(file_path)
        df = df.with_columns(pl.lit(db_name).alias('DB_Name'))
        dataframes.append(df)
    combined_df = pl.concat(dataframes, how='vertical')
    return combined_df

def read_MSPEC_file(path: Path | str, raw_fragment_tolerance_ppm: float = 10.0, normalized_fragment_tolerance_ppm: float = 5.0, molecular_ion_tolerance_ppm: float = 5.0) -> pl.DataFrame:
    with open(path, 'r') as file:
        file_contents = file.read()
    
    data = _read_file(file_contents)
    data = _annotate_and_filter_metadata(data)
    data = _annotate_spectra(
        data, 
        raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm, 
        normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm)

    # data = _add_num_clean_peaks(data)
    data = _add_precursor_type_indicators(data)

    data = _add_base_peak_mz_fraction_and_diff(data)
    data = _add_molecular_ion_info(data, molecular_ion_tolerance_ppm)

    return data.collect(engine='streaming')



def _read_file(file_contents: str):
    mz_intensity_pattern = r'(\d+\.\d+)\s(\d+(\.\d+)?)'
    # Collision_energy_ev_pattern = r'(\d+)e*V*v*$'

    entries = _split_entries(file_contents)
    data = pl.DataFrame(entries, schema={'raw': pl.String}).lazy()
    data = data.with_columns(
        pl.col('raw').str.extract(pattern=r'(?i)Name: (.+)',group_index=1).alias('Name'),
        pl.col('raw').str.extract(pattern=r'(?i)NIST#: (\d+)',group_index=1).alias('NIST_ID'),
        pl.col('raw').str.extract(pattern=r'(?i)DB#: (\d+)',group_index=1).alias('DB_ID'),
        pl.col('raw').str.extract(pattern=r'(?i)Instrument_?type: (.+)',group_index=1).alias('Instrument_type'),
        pl.col('raw').str.extract(pattern=r'(?i)Instrument: (.+)',group_index=1).alias('Instrument'),
        pl.col('raw').str.extract(pattern=r'(?i)(?:Spectrum_type|MSLEVEL): (?:MS)?(\d+)', group_index=1).str.to_integer().alias('MSLEVEL'), # extract the numeric MS level
        pl.col('raw').str.extract(pattern=r'(?i)Collision_gas: (.+)',group_index=1).alias('Collision_gas'),
        pl.col('raw').str.extract(pattern=r'(?i)Collision_?energy: (.+)',group_index=1).alias('Collision_energy_raw'),
        pl.col('raw').str.extract(pattern=r'(?i)Ionization: (.+)',group_index=1).alias('Ionization'),
        pl.col('raw').str.extract(pattern=r'(?i)Ion_?mode: (p|n)',group_index=1).alias('Ion_mode'), # works for P,N, and negative/postivie in any capitalization
        pl.col('raw').str.extract(pattern=r'(?i)Precursor_?type: (.+)',group_index=1).alias('Precursor_type'),
        pl.col('raw').str.extract(pattern=r'(?i)PrecursorMZ: (\d+\.?\d*)',group_index=1).alias('PrecursorMZ'),
        pl.col('raw').str.extract(pattern=r'(?i)MW: (\d+)',group_index=1).alias('MW'),
        pl.col('raw').str.extract(pattern=r'(?i)Formula: (.+)',group_index=1).alias('Formula'),
        pl.col('raw').str.extract(pattern=r'(?i)Num Peaks: (\d+)',group_index=1).alias('Num_Peaks'),
        pl.col('raw').str.extract(pattern=r'(?i)\nCAS#: ([0-9,-]+)',group_index=1).alias('CAS'),
        pl.col('raw').str.extract(pattern=r'(?i)\nRelated_CAS#: ([0-9,-]+)',group_index=1).alias('Related_CAS'),
        pl.col('raw').str.extract(pattern=r'(?i)\nInChIKey: (.+)',group_index=1).alias('InChIKey'),
        pl.col('raw').str.extract(pattern=r'(?i)\nExactMass: (\d+\.\d+)',group_index=1).alias('ExactMass'),
        pl.col('raw').str.extract(pattern=r'(?i)[Mm]z_diff=(-?\d+\.\d+)',group_index=1).alias('mz_diff'),
        pl.col('raw').str.extract_all(pattern=r'(?i)Synon: (.+)')
        .list.eval(pl.element().str.extract(pattern=r'(?i)Synon: (.+)',group_index=1))
        .alias('Synonyms'),
        pl.col('raw').str.extract(pattern=r'(?i)Peptide_sequence: (.+)').alias('Peptide_sequence'),
        pl.col('raw').str.extract(pattern=r'(?i)Peptide_mods: (.+)').alias('Peptide_mods'),
        pl.col('raw').str.extract(pattern=r'(?i)InChI: (.+)').alias('inchi'),
        pl.col('raw').str.extract(pattern=r'(?i)SMILES: (.+)').alias('smiles'),
        pl.col('raw').str.extract_all(pattern=mz_intensity_pattern).alias('mz_intensity')
    ).drop(
        'raw'
    ).with_columns(
        pl.col('InChIKey').str.extract(r'(.+?)-').alias('base_InChIKey'),
        pl.col("NIST_ID").str.to_integer(),
        pl.col("DB_ID").str.to_integer(),
        pl.col("MW").str.to_integer(),
        pl.col("Ion_mode").str.to_uppercase(),
        pl.col("Num_Peaks").str.to_integer(),
        pl.col("PrecursorMZ").cast(pl.Float64),
        pl.col('mz_diff').cast(pl.Float64),
        # extract numeric collision energies allowing integers or floats
        pl.col('Collision_energy_raw').str.extract(r'(?i)NCE=([0-9]+(?:\.[0-9]+)?)', group_index=1).cast(pl.Float64).alias('Collision_energy_NCE'),
        pl.col('Collision_energy_raw').str.extract(r'([0-9]+(?:\.[0-9]+)?)e*V*v*$', group_index=1).cast(pl.Float64).alias('Collision_energy_ev'),
        pl.col('Collision_energy_raw').str.strip_chars("[]").str.split(by=",").list.eval(
            pl.element().str.extract(pattern=r'([0-9]+(?:\.[0-9]+)?)')
        ).cast(
            pl.List(pl.Float64)
        ).alias("collision_energy_list"),
        pl.col('Formula').map_elements(format_formula_string_to_array,return_dtype=pl.List(pl.Int32)).list.to_array(width=num_elements).alias('Formula_array'),
        pl.col('mz_intensity').list.eval(pl.element().str.split(by=' ').list.get(index=0).cast(pl.Float64)).alias('raw_spectrum_mz'),
        pl.col('mz_intensity').list.eval(pl.element().str.split(by=' ').list.get(index=1).cast(pl.Float64)).alias('raw_spectrum_intensity'),
        pl.col('Precursor_type').str.replace(r'\[(M.*)\][+\\-]?\\d*', r'$1').str.replace('M', pl.col('Formula')).alias('precursor_formula')
    ).with_columns(
        pl.col('precursor_formula').map_elements(format_formula_string_to_array,return_dtype=pl.List(pl.Int32)).list.to_array(width=num_elements).alias('precursor_formula_array'),
        pl.col("collision_energy_list").list.len().ge(1).alias("multiple_collision_energies"),
        pl.col("collision_energy_list").list.mean().alias("collision_energy_mean")
    )
    return data

def _annotate_and_filter_metadata(data:T)-> T:
    '''filters out entries with missing or invalid metadata, low resolution spectra etc'''
    instrument_data_columns= plcs.by_name(['Instrument', 'Instrument_type',  'Ionization'])

    data = cast(T,data.filter(
       pl.all_horizontal(instrument_data_columns.str.contains(r'(?i)QQ').not_()), # ioniuzation migth incldue instrument type info, and we don't want triple quads since they are low res. sometimes triple quad is also written with only 2 Qs, but nothign else should be havign 2 Qs next to each other
        
    ).with_columns(
        pl.any_horizontal(instrument_data_columns.str.contains(r'(?i)LC')).alias("is_LC"), # 
        pl.any_horizontal(
        instrument_data_columns.str.contains(r'(?i)orbi(?:trap)?|HCD') |
        instrument_data_columns.str.contains(r'(?i)thermo') |
            (
                instrument_data_columns.str.contains(r'(?i)FT') &
                instrument_data_columns.str.contains(r'(?i)ICR').not_() & 
                instrument_data_columns.str.contains(r'(?i)TOF').not_()
            )

        ).alias("is_orbitrap"),
        pl.any_horizontal(instrument_data_columns.str.contains(r'(?i)TOF')).alias("is_TOF"),
        pl.any_horizontal(instrument_data_columns.str.contains(r'(?i)ESI|LC')).alias("is_ESI"), # LC is usually coupled with ESI
    )
    )
    
    

    return data


def _annotate_spectra(data: T, raw_fragment_tolerance_ppm: float, normalized_fragment_tolerance_ppm: float) -> T:
    # Determine adduct_mass based on Precursor_type
    adduct_mapping = pl.Series(name="Precursor_type", values=list(ADDUCT_MASSES.keys()), dtype=pl.String)
    adduct_masses = pl.Series(name="adduct_mass", values=list(ADDUCT_MASSES.values()), dtype=pl.Float64)
    adduct_df = pl.DataFrame({"Precursor_type": adduct_mapping, "adduct_mass": adduct_masses})

    if isinstance(data, pl.LazyFrame):
        adduct_lf = adduct_df.lazy()
        data_lf = data.join(adduct_lf, on="Precursor_type", how="left")
        data_frame = cast(T,data_lf)
    elif isinstance(data, pl.DataFrame):
        data_df = data.join(adduct_df, on="Precursor_type", how="left")
        data_frame = cast(T,data_df)
    else:
        raise TypeError(f"In function '_annotate_spectra', data must be a Polars DataFrame or LazyFrame, got {type(data)}")

    return cast(T,
    data_frame.with_columns(
        pl.struct([
            pl.col("precursor_formula_array").alias("precursor_formula"),
            pl.col("raw_spectrum_mz").alias("mz"),
            pl.col("raw_spectrum_intensity").alias("intensities")
        ]).mass_decomposition.clean_and_normalize_spectrum(
                raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
                normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
                min_dbe= -0.5,
                max_dbe= 40,
                dbe_mode="half_integer",
                water_absorption=True,
        ).alias("cleaned_normalized_spectra")
    ).with_columns( # Extract results and add adduct_mass back to normalized masses
        pl.col("cleaned_normalized_spectra").struct.field("normalized_masses").alias("cleaned_normalized_mz"),
        pl.col("cleaned_normalized_spectra").struct.field("intensities").alias("cleaned_normalized_intensity"),
        pl.col("cleaned_normalized_spectra").struct.field("formulas").alias("cleaned_fragment_formulas"),
        pl.col("cleaned_normalized_spectra").struct.field("formulas_str").alias("cleaned_fragment_formulas_str"),
        pl.col("cleaned_normalized_spectra").struct.field("errors_ppm").alias("cleaned_fragment_errors_ppm"),
    ).drop(
        "cleaned_normalized_spectra"
    ).with_columns(
         pl.struct([
            pl.col("cleaned_normalized_mz").alias("mz1"),
            pl.col("cleaned_normalized_intensity").alias("intensities1"),
            pl.col("raw_spectrum_mz").alias("mz2"),
            pl.col("raw_spectrum_intensity").alias("intensities2")
        ]).spectral_similarity.explained_intensity(permissive=True,ms2_tolerance_in_ppm=15.0).alias("explained_intensity"))
    )

def _add_precursor_type_indicators(data: T) -> T:
    fragment_pattern = r'-\d*'+  r'((H(\d+|[A-Z]|[a-z]))|([A-G]|[I-Z])[a-z]?\d*)'+ r'(([A-Z][a-z]?\d*))*'

    return cast(T,data.with_columns(
        pl.col('Precursor_type').str.contains('i').alias('Isotope'),
        pl.col('Precursor_type').str.contains('Cat').alias('Cation'),
        pl.col('Precursor_type').str.contains('[0-9]M').alias('Multimer'),
        pl.col('Precursor_type').str.contains('][0-9]').alias('MultiCharge'),
        pl.col('Precursor_type').str.contains(fragment_pattern).alias('Fragment')
        ).with_columns(
        (pl.col('Isotope') | pl.col('Cation') | pl.col('Multimer') | pl.col('MultiCharge') | pl.col('Fragment') |
         pl.col('Precursor_type').str.contains('M').not_() # there are some that are [123.1234]+, all of the m with single occurance, which are probably not clean
         ).not_().alias('clean_precursor'))
    )


def _add_molecular_ion_info(NIST: T, tolerance_ppm: float = 10.0) -> T:
    lazy_frame = NIST.lazy()
    lazy_frame = lazy_frame.with_columns(
        molecular_ion_intensity=pl.when(
            pl.col('cleaned_normalized_mz').list.last().is_close(
            pl.col('PrecursorMZ'), 
            rel_tol=tolerance_ppm*1e-6, abs_tol=200.*tolerance_ppm*1e-6)
            )
        .then(pl.col('cleaned_normalized_intensity').list.last())
        .otherwise(None)
        )


    if isinstance(NIST, pl.LazyFrame):
        return cast(T, lazy_frame)
    elif isinstance(NIST, pl.DataFrame):
        return cast(T, lazy_frame.collect(engine="streaming"))
    else:
        raise TypeError(f"In function '_add_molecular_ion_info', NIST must be a Polars DataFrame or LazyFrame, got {type(NIST)}")


def _add_base_peak_mz_fraction_and_diff(NIST: T) -> T:
    return cast(T, NIST.with_columns(
        pl.col('raw_spectrum_mz').list.get(pl.col('raw_spectrum_intensity').list.arg_max()).alias('base_peak_mz')
        ).with_columns(   
        pl.col('base_peak_mz').truediv(pl.col('PrecursorMZ')).round(3).alias('base_peak_div_precursor_mz'),
        pl.col('PrecursorMZ').sub(pl.col('base_peak_mz')).round(3).alias('precursor_minus_base_peak_mz')
    ))



def _find_missing_pattern_sections(file_contents, pattern):
    sections = _split_entries(file_contents)
    for section in sections:
        if pattern not in section:
            print('Missing '+ pattern+ ' in section:', section)
            break


def _split_entries(file_contents: str) -> list:
    entries = re.split(r'\n\s*\n', file_contents)
    if entries[len(entries)-1] == '':
        entries.pop()
    return entries


def _add_inchi_SMILES_from_pubchem(NIST: T, pubchem_path: str | Path) -> T:
    if 'InChI' in NIST.columns:
        raise Warning("InChI column already exists in NIST schema")

    NIST_lf = NIST.select(['NIST_ID','InChIKey']).lazy()
    
    pubchem = pl.scan_parquet(pubchem_path,low_memory=True).rename(
        {'CID':'pubchem_CID',
         'SMILES:':'CanonicalSMILES',}
         ,strict=False)
    combined = NIST_lf.join(pubchem, left_on='InChIKey', right_on='InChIKey', how='left')
    combined = combined.unique(subset=['NIST_ID'],keep='any')
    
    # combined_df = combined.collect(streaming=True)
    
    # if 'InChI' not in combined_df.columns:
    #     raise ValueError("InChI column was not written for some reason")
        
    result = combined.drop('InChIKey').join(NIST_lf, on='NIST_ID', how='right', coalesce=True)

    if isinstance(NIST, pl.DataFrame):
        return cast(T, result.collect(engine="streaming"))
    elif isinstance(NIST, pl.LazyFrame):
        return cast(T, result)
    else:
        raise TypeError(f"In function '_add_inchi_SMILES_from_pubchem', NIST must be a Polars DataFrame or LazyFrame, got {type(NIST)}")

if __name__ == "__main__":
    # Example usage
    from time import perf_counter
    start_time = perf_counter()
    nist= pl.read_parquet(r"D:\Nir\pyscreen_test\NIST23.parquet")
    # replace the DB_Name with the correct one: 
    # hr_msms -> hr_msms_nist
    # NIST_hr_msms2 -> nist_hr_msms#2
    nist = nist.with_columns(
        pl.when(pl.col('DB_Name').eq('hr_msms')).then(pl.lit('hr_msms_nist'))
        .when(pl.col('DB_Name').eq('NIST_hr_msms2')).then(pl.lit('nist_hr_msms#2'))
        .otherwise(pl.col('DB_Name')).alias('DB_Name'))
    nist.write_parquet(r"D:\Nir\pyscreen_test\NIST23_fixed.parquet")

    # #### creation of NIST23 dataframe
    # file_dir = Path('/home/analytit_admin/Data/NIST_hr_msms/')
    # # now the names and DB_name of the files:
    # file_names = [
    #     ('hr_msms_1.MSPEC', 'hr_msms_nist'),
    #     ('hr_msms_2.MSPEC', 'hr_msms_nist'),
    #     ('hr_msms_3.MSPEC', 'hr_msms_nist'),
    #     ('hr_msms_4.MSPEC', 'hr_msms_nist'),
    #     ('hr_msms_5.MSPEC', 'hr_msms_nist'),
    #     ('hr_msms_6.MSPEC', 'hr_msms_nist'),
    #     ('NIST_hr_msms2_1.MSPEC', 'nist_hr_msms#2'),
    #     ('NIST_hr_msms2_2.MSPEC', 'nist_hr_msms#2'),
    #     ('NIST_hr_msms2_3.MSPEC', 'nist_hr_msms#2'),
    #     ('NIST_hr_msms2_4.MSPEC', 'nist_hr_msms#2'),
    #     ('NIST_hr_msms2_5.MSPEC', 'nist_hr_msms#2'),
    # ]
    # file_list = [(file_dir / file_name, db_name) for file_name, db_name in file_names]
    # nist_df = create_nist_dataframe(file_list)
    # end_create_time = perf_counter()
    # print(f"Time taken to create NIST23 DataFrame: {end_create_time - start_time:.2f} seconds")
    # nist_df.write_parquet(file_dir / 'NIST23.parquet')
    # print("NIST23 DataFrame created and saved to NIST23.parquet")
    # end_write_time = perf_counter()
    # print(f"Time taken to write NIST23 DataFrame: {end_write_time - end_create_time:.2f} seconds")
    # print(f"Total time taken: {end_write_time - start_time:.2f} seconds")