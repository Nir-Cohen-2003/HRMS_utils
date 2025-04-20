import re
import polars as pl
import numpy as np
from MS_utils.formula import formula_fits_mass, format_formula_string_to_array,  get_precursor_ion_formula_array, num_elements
from pathlib import Path 
from scipy.stats import linregress



def read_MSPEC_file(path: Path | str) -> pl.DataFrame:
    with open(path, 'r') as file:
        file_contents = file.read()
    
    data = get_non_spectrum_data(file_contents)
    spectra = get_spectra(file_contents)

    data = pl.concat([data, spectra], how='horizontal')
    data = add_num_clean_peaks(data)
    data = add_precursor_type_indicators(data)

    data = add_base_peak_mz_fraction_and_diff(data)
    data = add_Mol_intensity_to_clean_spectrum(data)

    return data



def get_spectra(file_contents: str) -> pl.DataFrame:
    ''' gets spectra via regex and some numpy operations'''
    entries = split_entries(file_contents)
    results = []
    for entry in entries:
        results.append(get_entry_spectrum_and_formula(entry))
    #note - when stroing and reading the data, we get a flat list.
    try:
        results = list(zip(*results))
        spectra = pl.DataFrame(results,schema={
            'raw_spectrum_mz': pl.List(pl.Float64),
            'raw_spectrum_intensity': pl.List(pl.Float64),
            'normalized_spectrum_mz': pl.List(pl.Float64),
            'clean_spectrum_mz': pl.List(pl.Float64),
            'clean_spectrum_intensity': pl.List(pl.Float64),
            'clean_spectrum_formula':pl.List(pl.String),
            'clean_spectrum_formula_array':pl.List(pl.Array(pl.Int64,num_elements))})

        return spectra
    except Exception as e:
        print('Error in writing the data. The error is:', e)
        return

def get_entry_spectrum_and_formula(entry: str) -> tuple:
    '''
    extracts teh spcetrum and the formula from the entry, if it is singly charged and fits the mass.
    does not return the formulas or formula array if the formula doesn't fit the mass or the entry is multiply charged.
    '''
    entry_raw_mz = []
    entry_raw_intensity = []
    possible_formulas = []

    entry_clean_mz = []
    entry_clean_intensity = []
    entry_clean_formulas = []
    entry_clean_formulas_array = []

    mz_intensity_pattern = r'(\d+\.\d+)\s(\d+(\.\d+)?)'
    formula_pattern = r'(\s"(.*?)((([A-Z][a-z]?\d*)|[+-]|\d)+)=?p[/+-])?'
    unknown_pattern = r'(\s"\?)?'
    precursor_pattern = r'(\s"p/)?'
    pattern = mz_intensity_pattern + formula_pattern + unknown_pattern + precursor_pattern
    #pattern = r'(\d+\.\d+)\s(\d+(\.\d+)?)'+r'(\s"(.*?)((([A-Z][a-z]?\d*)|[+-]|\d)+)=?p[/+-])?'+r'(\s"\?)?'+r'(\s"p/)?'
    entry_raw_fragments = re.findall(pattern, entry)
    for i in range(len(entry_raw_fragments)):
        entry_raw_mz.append(entry_raw_fragments[i][0])
        entry_raw_intensity.append(entry_raw_fragments[i][1])
        possible_formulas.append(entry_raw_fragments[i][5])
    
    entry_raw_mz = np.array(entry_raw_mz, dtype=np.float64).flatten()
    entry_raw_intensity = np.array(entry_raw_intensity, dtype=np.float64).flatten()

    possible_mz_diff = re.search(r'[Mm]z_diff=(-?\d+\.\d+)', entry)
    if possible_mz_diff is not None:
        mz_normalization_coefficient= 1.0 + float(possible_mz_diff.group(1))*1e-6
    else:   
        mz_normalization_coefficient = 1
    entry_normalized_mz = np.round(np.divide(entry_raw_mz, mz_normalization_coefficient), 4)
    
    for i in range(len(entry_raw_fragments)):
        if 'p' in entry_raw_fragments[i][9]: # if it's the molecular ion
            entry_clean_mz.append(entry_normalized_mz[i])
            entry_clean_intensity.append(entry_raw_intensity[i])
            entry_clean_formulas.append('Mol')
            entry_clean_formulas_array.append(get_precursor_ion_formula_array(entry))
        elif formula_fits_mass(possible_formulas[i], entry_normalized_mz[i]): # this will probably give false for any multiply charged ion
            entry_clean_mz.append(entry_normalized_mz[i])
            entry_clean_intensity.append(entry_raw_intensity[i])
            entry_clean_formulas.append(possible_formulas[i])
            formula_array = format_formula_string_to_array(possible_formulas[i])
            entry_clean_formulas_array.append(formula_array)
    return (entry_raw_mz, entry_raw_intensity, entry_normalized_mz, entry_clean_mz, entry_clean_intensity, entry_clean_formulas, entry_clean_formulas_array)




def get_spectra_via_UDF(file_contents: str) -> pl.DataFrame:
    '''
    uses a udf and retunring a nested struct do get the spectrum.
    this doesn't work becuase the udf can't return a nested struct to that extent. 
    (some lists and a list of arrays)
    '''
    spectrum_struct = pl.Struct({
        'raw_spectrum_mz': pl.List(pl.Float64),
        'raw_spectrum_intensity': pl.List(pl.Float64),
        'normalized_spectrum_mz': pl.List(pl.Float64),
        'clean_spectrum_mz': pl.List(pl.Float64),
        'clean_spectrum_intensity': pl.List(pl.Float64),
        'clean_spectrum_formula':pl.List(pl.String),
        'clean_spectrum_formula_array':pl.List(pl.Array(pl.Int64,num_elements))
    })
    entries = split_entries(file_contents)

    spectra = pl.DataFrame({'raw': entries})
    spectra = spectra.with_columns(
        pl.col('raw').map_elements(get_entry_spectrum_and_formula_UDF,return_dtype=spectrum_struct).alias('spectra'))
    spectra = spectra.with_columns(pl.col('spectra').struct.unnest())
    spectra = spectra.drop(['raw','spectra'])
    return spectra
    
def get_entry_spectrum_and_formula_UDF(entry: str) -> dict:
    entry_raw_mz = []
    entry_raw_intensity = []
    possible_formulas = []

    entry_clean_mz = []
    entry_clean_intensity = []
    entry_clean_formulas = []
    entry_clean_formulas_array = []

    mz_intensity_pattern = r'(\d+\.\d+)\s(\d+(\.\d+)?)'
    formula_pattern = r'(\s"(.*?)((([A-Z][a-z]?\d*)|[+-]|\d)+)=?p[/+-])?'
    unknown_pattern = r'(\s"\?)?'
    precursor_pattern = r'(\s"p/)?'
    pattern = mz_intensity_pattern + formula_pattern + unknown_pattern + precursor_pattern
    #pattern = r'(\d+\.\d+)\s(\d+(\.\d+)?)'+r'(\s"(.*?)((([A-Z][a-z]?\d*)|[+-]|\d)+)=?p[/+-])?'+r'(\s"\?)?'+r'(\s"p/)?'
    entry_raw_fragments = re.findall(pattern, entry)
    for i in range(len(entry_raw_fragments)):
        entry_raw_mz.append(entry_raw_fragments[i][0])
        entry_raw_intensity.append(entry_raw_fragments[i][1])
        possible_formulas.append(entry_raw_fragments[i][5])
    
    entry_raw_mz = np.array(entry_raw_mz, dtype=np.float64).flatten()
    entry_raw_intensity = np.array(entry_raw_intensity, dtype=np.float64).flatten()

    possible_mz_diff = re.search(r'[Mm]z_diff=(-?\d+\.\d+)', entry)
    if possible_mz_diff is not None:
        mz_normalization_coefficient= 1.0 + float(possible_mz_diff.group(1))*1e-6
    else:   
        mz_normalization_coefficient = 1
    entry_normalized_mz = np.round(np.divide(entry_raw_mz, mz_normalization_coefficient), 4)
    
    for i in range(len(entry_raw_fragments)):
        if 'p' in entry_raw_fragments[i][9]: # if it's the molecular ion
            entry_clean_mz.append(entry_normalized_mz[i])
            entry_clean_intensity.append(entry_raw_intensity[i])
            entry_clean_formulas.append('Mol')
            entry_clean_formulas_array.append(get_precursor_ion_formula_array(entry))
        elif formula_fits_mass(possible_formulas[i], entry_normalized_mz[i]): # this will probably give false for any multiply charged ion
            entry_clean_mz.append(entry_normalized_mz[i])
            entry_clean_intensity.append(entry_raw_intensity[i])
            entry_clean_formulas.append(possible_formulas[i])
            formula_array = format_formula_string_to_array(possible_formulas[i])
            entry_clean_formulas_array.append(formula_array)

    spectrum = {
        'raw_spectrum_mz': entry_raw_mz,
        'raw_spectrum_intensity': entry_raw_intensity,
        'normalized_spectrum_mz': entry_normalized_mz,
        'clean_spectrum_mz': entry_clean_mz,
        'clean_spectrum_intensity': entry_clean_intensity,
        'clean_spectrum_formula': entry_clean_formulas,
        'clean_spectrum_formula_array': entry_clean_formulas_array
    }
    return spectrum

def get_non_spectrum_data(file_contents: str) -> pl.DataFrame:
    entries = split_entries(file_contents)
    data = pl.DataFrame(entries, schema={'raw': pl.String})
    data = data.with_columns(
        pl.col('raw').str.extract(pattern=r'Name: (.+)',group_index=1).alias('Name'),
        pl.col('raw').str.extract(pattern=r'NIST#: (\d+)',group_index=1).alias('NIST_ID'),
        pl.col('raw').str.extract(pattern=r'DB#: (\d+)',group_index=1).alias('DB_ID'),
        pl.col('raw').str.extract(pattern=r'Instrument_type: (.+)',group_index=1).alias('Instrument_type'),
        pl.col('raw').str.extract(pattern=r'Instrument: (.+)',group_index=1).alias('Instrument'),
        pl.col('raw').str.extract(pattern=r'Spectrum_type: (.+)',group_index=1).alias('Spectrum_type'),
        pl.col('raw').str.extract(pattern=r'Collision_gas: (.+)',group_index=1).alias('Collision_gas'),
        pl.col('raw').str.extract(pattern=r'Collision_energy: (.+)',group_index=1).alias('Collision_energy_raw'),
        pl.col('raw').str.extract(pattern=r'Ionization: (.+)',group_index=1).alias('Ionization'),
        pl.col('raw').str.extract(pattern=r'Ion_mode: ([P,N])',group_index=1).alias('Ion_mode'),
        pl.col('raw').str.extract(pattern=r'Precursor_type: (.+)',group_index=1).alias('Precursor_type'),
        pl.col('raw').str.extract(pattern=r'PrecursorMZ: (\d+\.?\d*)',group_index=1).alias('PrecursorMZ'),
        pl.col('raw').str.extract(pattern=r'MW: (\d+)',group_index=1).alias('MW'),
        pl.col('raw').str.extract(pattern=r'Formula: (.+)',group_index=1).alias('Formula'),
        pl.col('raw').str.extract(pattern=r'Num Peaks: (\d+)',group_index=1).alias('Num_Peaks'),
        pl.col('raw').str.extract(pattern=r'\nCAS#: ([0-9,-]+)',group_index=1).alias('CAS'),
        pl.col('raw').str.extract(pattern=r'\nRelated_CAS#: ([0-9,-]+)',group_index=1).alias('Related_CAS'),
        pl.col('raw').str.extract(pattern=r'\nInChIKey: (.+)',group_index=1).alias('InChIKey'),
        pl.col('raw').str.extract(pattern=r'\nExactMass: (\d+\.\d+)',group_index=1).alias('ExactMass'),
        pl.col('raw').str.extract(pattern=r'[Mm]z_diff=(-?\d+\.\d+)',group_index=1).alias('mz_diff'),
        pl.col('raw').str.extract_all(pattern=r'Synon: (.+)')
        .list.eval(pl.element().str.extract(pattern=r'Synon: (.+)',group_index=1))
        .alias('Synonyms'),
        pl.col('raw').str.extract(pattern=r'Peptide_sequence: (.+)').alias('Peptide_sequence'),
        pl.col('raw').str.extract(pattern=r'Peptide_mods: (.+)').alias('Peptide_mods'),
        pl.col('raw').str.extract(pattern=r'InChI: (.+)').alias('inchi'),
        pl.col('raw').str.extract(pattern=r'SMILES: (.+)').alias('smiles'),

    )
    data = data.drop('raw')
    Collision_energy_ev_pattern = r'(\d+)e*V*v*$'
    data = data.with_columns(
        pl.col('InChIKey').str.extract(r'(.+?)-').alias('base_InChIKey'),
        pl.col("NIST_ID").str.to_integer(),
        pl.col("DB_ID").str.to_integer(),
        pl.col("MW").str.to_integer(),
        pl.col("Num_Peaks").str.to_integer(),
        pl.col("PrecursorMZ").cast(pl.Float64),
        pl.col('mz_diff').cast(pl.Float64),
        pl.col('Collision_energy_raw').str.extract(r'NCE=(\d+)').str.to_integer().alias('Collision_energy_NCE'),
        pl.col('Collision_energy_raw').str.extract(Collision_energy_ev_pattern).str.to_integer().alias('Collision_energy_ev'),
        pl.col('Formula').map_elements(format_formula_string_to_array,return_dtype=pl.List(pl.Int64)).list.to_array(width=num_elements).alias('Formula_array'))
    return data



def add_num_clean_peaks(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame |  pl.LazyFrame:
    data = data.with_columns(pl.col('clean_spectrum_mz').list.len().alias('num_clean_peaks'))
    return data


def add_precursor_type_indicators(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame |  pl.LazyFrame:
    fragment_pattern = r'-\d*'+  r'((H(\d+|[A-Z]|[a-z]))|([A-G]|[I-Z])[a-z]?\d*)'+ r'(([A-Z][a-z]?\d*))*'

    data = data.with_columns(
        pl.col('Precursor_type').str.contains('i').alias('Isotope'),
        pl.col('Precursor_type').str.contains('Cat').alias('Cation'),
        pl.col('Precursor_type').str.contains('[0-9]M').alias('Multimer'),
        pl.col('Precursor_type').str.contains('][0-9]').alias('MultiCharge'),
        pl.col('Precursor_type').str.contains(fragment_pattern).alias('Fragment'))
    data = data.with_columns(
        (pl.col('Isotope') | pl.col('Cation') | pl.col('Multimer') | pl.col('MultiCharge') | pl.col('Fragment') |
         pl.col('Precursor_type').str.contains('M').not_() # there are some that are [123.1234]+, all of the m with single occurance, which are probably not clean
         ).not_().alias('clean_precursor'))


    return data

# returns the intensity of the molecular ion if found, -1 if not found, -2 if the biggest mz value in not the molecular ion
def add_Mol_intensity_to_clean_spectrum(NIST: pl.DataFrame) -> pl.DataFrame:
    NIST = NIST.with_columns(
        pl.when(pl.col('clean_spectrum_formula').list.contains("Mol"))
        .then
        (
        pl.when(pl.col('clean_spectrum_formula').list.last().eq("Mol"))
        .then(pl.col('clean_spectrum_intensity').list.last())
        .otherwise(-2)
        )
        .otherwise(-1)
        .alias('molecular_ion_intensity')
    )
    return NIST


def add_base_peak_mz_fraction_and_diff(NIST: pl.DataFrame) -> pl.DataFrame:
    NIST = NIST.with_columns(
        pl.col('raw_spectrum_mz').list.get(pl.col('raw_spectrum_intensity').list.arg_max()).alias('base_peak_mz'))
    NIST= NIST.with_columns(   
        pl.col('base_peak_mz').truediv(pl.col('PrecursorMZ')).round(3).alias('base_peak_div_precursor_mz'),
        pl.col('PrecursorMZ').sub(pl.col('base_peak_mz')).round(3).alias('precursor_minus_base_peak_mz')
    )
    return NIST


def find_missing_pattern_sections(file_contents, pattern):
    sections = split_entries(file_contents)
    for section in sections:
        if pattern not in section:
            print('Missing '+ pattern+ ' in section:', section)
            break


def split_entries(file_contents: str) -> list:
    entries = re.split(r'\n\s*\n', file_contents)
    if entries[len(entries)-1] == '':
        entries.pop()
    return entries


def add_inchi_SMILES_from_pubchem(NIST: pl.DataFrame, pubchem_path: str | Path) -> pl.DataFrame:
    if 'InChI' in NIST.schema.names():
        raise Warning("InChI column already exists in NIST schema")
    NIST_lf = NIST.select(['NIST_ID','InChIKey']).lazy()
    pubchem = pl.scan_parquet(pubchem_path,low_memory=True).rename(
        {'CID':'pubchem_CID',
         'SMILES:':'CanonicalSMILES',}
         ,strict=False)
    combined = NIST_lf.join(pubchem, left_on='InChIKey', right_on='InChIKey', how='left')
    combined = combined.unique(subset=['NIST_ID'],keep='any')
    combined = combined.collect(streaming=True)
    
    if 'InChI' not in combined.schema.names():
        raise ValueError("InChI column was not written for some reason")
    combined = combined.drop('InChIKey').join(NIST, on='NIST_ID', how='right',coalesce=True)
    return combined

def add_estimated_ev(NIST: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame :
    NIST_temp = NIST.select(
        ['NIST_ID','Collision_energy_NCE','Collision_energy_ev',
         'PrecursorMZ',
         'Precursor_type', 'Instrument_type','Instrument',]
        )
    
    if isinstance(NIST_temp,pl.LazyFrame):
        NIST_temp = NIST_temp.collect()
    
    NIST_temp = NIST_temp.filter(
        pl.col('Instrument_type').str.contains('HCD')
    )
    
    NIST_with_ev_and_NCE = NIST_temp.filter(
        pl.col('Collision_energy_NCE').is_not_null() & 
        pl.col('Collision_energy_ev').is_not_null())
    
    NIST_with_ev_and_NCE = NIST_with_ev_and_NCE.with_columns(
        pl.col('Collision_energy_NCE').mul(pl.col('PrecursorMZ')).alias('Collision_energy_ev_estimated_no_coefficient')
        ).select(
            ['Collision_energy_ev_estimated_no_coefficient','Collision_energy_ev',
             'Instrument']
            )
    
    instrument_dfs = []
    for instrument in ['Elite','Velos','Fusion']:
        instrument_data = NIST_with_ev_and_NCE.filter(pl.col('Instrument').str.contains(instrument))
        if instrument_data.height == 0:
            print('no data for '+instrument)
            continue
        instrument_relation = get_energy_relation(instrument_data)
        slope , intercept = instrument_relation['slope'],instrument_relation['intercept']
        instrument_data = NIST_temp.filter(pl.col('Instrument').str.contains(instrument))
        instrument_data = add_estimated_ev_per_split(instrument_data,slope=slope,intercept=intercept)
        instrument_dfs.append(instrument_data)

    NIST_ev = pl.concat(instrument_dfs,how='vertical')
    NIST_ev = NIST_ev.select(['NIST_ID','Collision_energy_ev_estimated'])
    NIST = NIST.join(NIST_ev, on='NIST_ID', how='left')

    return NIST
    
def add_estimated_ev_per_split(split:pl.LazyFrame | pl.DataFrame,slope,intercept) -> pl.LazyFrame | pl.DataFrame:
    split = split.with_columns(
        pl.col('Collision_energy_NCE').mul(pl.col('PrecursorMZ')).mul(slope).add(intercept)
        .alias('Collision_energy_ev_estimated')
    )
    return split

def get_energy_relation(NIST_with_ev_and_NCE: pl.DataFrame):
    NIST_with_ev_and_NCE = NIST_with_ev_and_NCE.sort('Collision_energy_ev').with_columns(pl.col('Collision_energy_ev').cast(pl.Float64))
    NIST_energies_x = NIST_with_ev_and_NCE.select(
        ['Collision_energy_ev_estimated_no_coefficient']
        ).to_numpy().flatten()
    NIST_energies_y  = NIST_with_ev_and_NCE.select(
        ['Collision_energy_ev']
        ).to_numpy().flatten()

    result = linregress(NIST_energies_x, NIST_energies_y)
    slope, intercept, r_value,  = result.slope, result.intercept, result.rvalue

    return {'slope':slope, 'intercept':intercept,'r_value': r_value}

