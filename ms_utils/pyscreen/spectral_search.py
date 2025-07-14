import polars as pl
import numpy as np
from pathlib import Path
import os
from typing import List,Literal
from time import sleep
import shutil as sh 
from numba import njit
import matplotlib.pyplot as plt
from ms_entropy import calculate_entropy_similarity
from dataclasses import dataclass, field
@dataclass
class search_config:
    NIST_db_path:Path|str # this is the path to the NIST database, which is a parquet file.
    polarity:str
    ms1_mass_tolerance:float=5e-6
    ms2_mass_tolerance:float=10e-6 # high, but it's because NIST can have pretty high mass error.
    DotProd_threshold:dict[int:int|float]=field(default_factory=lambda:{
                0:650,
                1:700,
                2:800,
                3:900})
    search_engine:Literal['entropy','nir_cosine','cosine']='entropy' # the search engine to use, can be 'entropy', 'nir_cosine' or 'cosine'
    noise_threshold:float=0.005
    def __post_init__(self):
        # make sure the NIST_db_path is a Path object, and that the file exists
        if isinstance(self.NIST_db_path, str):
            self.NIST_db_path = Path(self.NIST_db_path)
        if not isinstance(self.NIST_db_path, Path):
            raise TypeError("NIST_db_path must be a Path object or a string")
        if not self.NIST_db_path.exists():
            raise FileNotFoundError(f"NIST_db_path {self.NIST_db_path} does not exist.")
        # we  want to accept tolerance both in ppm and in absolute values, so we convert them to absolute values. how? the tolerance would never be over 100 ppm, which is 0.0001, so if the value is more than that, its a ppm value and we multiply by 1e-6
        if self.ms1_mass_tolerance > 0.0001:
            self.ms1_mass_tolerance = self.ms1_mass_tolerance * 1e-6
        if self.ms2_mass_tolerance > 0.0001:
            self.ms2_mass_tolerance = self.ms2_mass_tolerance * 1e-6
    # method for getting a dict from this config
    def to_dict(self) -> dict:
        return {
            'NIST_db_path': str(self.NIST_db_path),
            'polarity': self.polarity,
            'ms1_mass_tolerance': self.ms1_mass_tolerance,  
            'ms2_mass_tolerance': self.ms2_mass_tolerance,
            'DotProd_threshold': self.DotProd_threshold,
            'search_engine': self.search_engine,
            'noise_threshold': self.noise_threshold
        }
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'search_config':
        """Create a search_config instance from a dictionary, using defaults for None or missing values.
        Raises ValueError if required fields are missing or None."""
        if isinstance(config_dict, cls):
            return config_dict
        if not isinstance(config_dict, dict):
            raise TypeError("the input to search_config.from_dict() must be a dictionary or a search_config instance.")
        nist_db_path = config_dict.get('NIST_db_path')
        polarity = config_dict.get('polarity')

        if nist_db_path is None:
            raise ValueError("NIST_db_path is required and cannot be None.")
        if polarity is None:
            raise ValueError("polarity is required and cannot be None.")

        # Use the class defaults for any missing or None values
        kwargs = {}
        for field_ in cls.__dataclass_fields__:
            if field_ in config_dict and config_dict[field_] is not None:
                kwargs[field_] = config_dict[field_]
        # Ensure required fields are set
        kwargs['NIST_db_path'] = Path(nist_db_path)
        kwargs['polarity'] = polarity
        return cls(**kwargs)



#### runs at import of this file, to automagically set the NIST_path variable to the NIST23 installation path.

#
NIST_PATH = Path(r"")

# search the following location, in order:
# if its windows, it will search the default NIST installation paths.
if os.name == 'nt':
    possible_paths = [
        Path(r"C:\NIST23\MSSEARCH"),
        Path(r"D:\NIST23\MSSEARCH"),
        Path(r"P:\NIST23\MSSEARCH"),
        Path(r"C:\Program Files\NIST23\MSSEARCH"),
        Path(r"C:\Program Files\NIST\NIST23"),
        Path(r"C:\Program Files\NIST\NIST23\MSSEARCH"),
        Path(r"C:\Program Files (x86)\NIST\NIST23"),
        Path(r"C:\Program Files (x86)\NIST23\MSSEARCH"),
        Path(r"C:\Program Files (x86)\NIST\NIST23\MSSEARCH"),
    ]
else:
    # if its not windows, it won't work
    possible_paths = [
        Path(r"/home/analytit_admin/Data/NIST_hr_msms/NIST23/MSSEARCH"),
        Path(r"/home/analytit_admin/Data/NIST_hr_msms/NIST23/MSSEARCH"),
    ]
# now we go over them, searching for the nist executable: nistms.exe 
for path in possible_paths:
    if path.exists() and (path / 'nistms.exe').exists():
        NIST_PATH = path
        print(f"NIST path found: {NIST_PATH}")
        break

def get_NIST(config:search_config) -> pl.DataFrame:
    NIST = pl.scan_parquet(source=config.NIST_db_path)
    NIST = NIST.select([
    'Name',
    'NIST_ID',
    'DB_Name',
    'DB_ID',
    'Precursor_type',
    'PrecursorMZ',
    'Ion_mode',
    'Instrument_type',
    'Formula',
    'Num_Peaks',
    'CAS',
    'InChIKey',
    'Synonyms',
    'raw_spectrum_intensity',
    'normalized_spectrum_mz',
    'MultiCharge',
    'Collision_energy_raw',
    # 'Collision_energy_NCE'
    # 'InChI',
    # 'CanonicalSMILES'
    ]).rename(
            {'InChIKey':'inchikey_NIST',
             'CAS':'CAS_NIST',
             'Synonyms':'Synonyms_NIST',
             'Name':'Name_NIST',
             'Formula':'Formula_NIST',
             'Precursor_type':'Precursor_type_NIST',
             'Collision_energy_raw':'collision_energy_NIST'
             }
        )
    

    if config.polarity.lower()=="positive":
        mode="P"
    elif config.polarity.lower()=="negative":
        mode="N"

    if mode is not None:
        NIST = NIST.filter(
            pl.col("Ion_mode").eq(mode)
    )
    else:
        print("supply correct operation mode to NIST_presearch_filtering to get better performance")
        NIST = NIST

    NIST = NIST.filter(
        pl.col('Instrument_type').eq('HCD') |
        pl.col('Instrument_type').eq('IT-FT/ion trap with FTMS'),
        pl.col('MultiCharge').not_()
    )
    NIST = NIST.collect()
    return NIST

def custom_search(
        query_df: pl.DataFrame | pl.LazyFrame, 
        NIST:pl.DataFrame | pl.LazyFrame,
        config:search_config,
    )->pl.DataFrame:
    match config.search_engine.lower():
        case 'entropy':
            function=entropy_score_batch
        case 'nir_cosine':
            function=identity_score_NIST_like_batch
        case 'cosine':
            function=identity_score_NIST_like_batch
        case _:
            print(config.search_engine)
            raise Exception("no valid search engine given")
        
    NIST_lf = NIST.select(
        [
            'NIST_ID',
            'PrecursorMZ',
            'raw_spectrum_intensity',
            'normalized_spectrum_mz',
        ]
    ).lazy()
    query_lf = query_df.lazy().select([
            'Peak ID',
            'Precursor_mz_MSDIAL',
            'msms_intensity',
            'msms_m/z'
            ])

    results = NIST_lf.join_where(
        query_lf,
        pl.col('PrecursorMZ').ge(pl.col('Precursor_mz_MSDIAL').mul(1-config.ms1_mass_tolerance)),
        pl.col('PrecursorMZ').le(pl.col('Precursor_mz_MSDIAL').mul(1+config.ms1_mass_tolerance)),
        suffix='_query'
    )
    results = results.collect(streaming=True)
    #using a UDF (User Defined Function) requires materializing in memory anyway, so it's better to collect
    results = results.with_columns(
        pl.struct(
            pl.col('raw_spectrum_intensity'),
            pl.col('normalized_spectrum_mz'),
            pl.col('msms_intensity'),
            pl.col('msms_m/z')
        ).map_batches(
            lambda spectra: function(
                spectra.struct.field('normalized_spectrum_mz').to_numpy(),
                spectra.struct.field('raw_spectrum_intensity').to_numpy(),
                spectra.struct.field('msms_m/z').to_numpy(),
                spectra.struct.field('msms_intensity').to_numpy(),
                config),
            return_dtype=pl.Float64,
            is_elementwise=True
        ).mul(1000.0).alias('DotProd')
    )
    results = results.select([
        'NIST_ID',
        'Peak ID',
        'DotProd'
        ])
    return results




################ functions beyond this point are for search using the NIST software, which is very slow. ##############

def NIST_search_external(
        query_df: pl.DataFrame, 
        MSDIAL_file_path: Path|str,
        NIST:pl.DataFrame| pl.LazyFrame,
        ignore_previous_results:bool=True,
        ) -> pl.DataFrame:
    
    NIST.select(['NIST_ID','DB_ID',"DB_Name"])

    searched = _searched_already(file_path=MSDIAL_file_path)
    if not searched['searched'] or ignore_previous_results:
        query_df = format_MSP(query_df)
        send_frame_to_NIST(query_df)
        _wait_for_NIST_ready()
        results_path = _save_search_results(data_path=MSDIAL_file_path)
        results = _read_NIST_eager(results_path=results_path)
    else:
        results = _read_NIST_eager(results_path=searched['path'])
    results = results.select(['Peak ID','DotProd','DB_Name','DB_ID'])
    results = results.join(NIST,on=['DB_Name','DB_ID'],how='left')
    results = results.drop(['DB_Name','DB_ID'])
    
    
    return results

# # reads the current results file, saves it if needed (meaning no results were saved for the given file.)
# # the path it takes is the path of the MSDIAL file it (presumably) reads the results of.
# # waits for NIST to finish.
# # also note- it first copies and saves the file, then reads from the saved file.
# def read_NIST(file_path : str | Path) -> pl.DataFrame:
#     _wait_for_NIST_ready()
#     results = _read_NIST_eager(MSDIAL_file_path=file_path)
#     return results

# reads the current results file, saves it if needed (if no results were saved already for the given file.)
# the path it takes is the path of the MSDIAL file it (presumably) reads the results of.
# does not wait for NIST to finish.
# also note- it first copies and saves the file, then reads from the sved file.
def _read_NIST_eager(
        results_path: str| Path) -> pl.DataFrame:
    if isinstance(results_path,str):
        results_path = Path(results_path)

    results = _read_SRCRESLT_type_file(results_path)
    # if results.item(0,0) is None:
    #     print("error in file: " + str(MSDIAL_file_path) + "no results found in nist after filtering based on library etc', so probavly wrong search settings in NIST")
    
    return results

# checks if a given MSDIAL export file was searched in NIST already
# and returns teh answer and the possible path of the result file.
def _searched_already(file_path : str | Path ) -> dict:
    '''checks if a given MSDIAL export file was searched in NIST already and returns the answer and the possible path of the result file.'''
    if isinstance(file_path,str):
        file_path = Path(file_path)
    possible_results_path = file_path.parent.joinpath(file_path.stem+'_NIST_results.txt')
    return {'searched':possible_results_path.exists(), 'path':possible_results_path}

def send_frame_to_NIST(chromatogram: pl.LazyFrame | pl.DataFrame):
    
    if isinstance(chromatogram,pl.DataFrame):
        if 'msp' not in chromatogram.columns:
            chromatogram = format_MSP(chromatogram)
        msp = chromatogram.select(['msp']).to_series()
    elif isinstance(chromatogram,pl.LazyFrame):
        if 'msp' not in chromatogram.collect_schema().names():
            chromatogram = format_MSP(chromatogram)
        msp = chromatogram.select(['msp']).collect().to_series()

    
    msp_to_print = msp.str.join(delimiter="\n\n")
    data_file_path = NIST_PATH.joinpath('data2nist.msp')
    msp_file = open(data_file_path,'w')
    msp_file.write(msp_to_print[0])
    msp_file.close()

    #first we need to make sure that autoimp.msd exists, and points to filespec.fil
    autoimp_path = NIST_PATH.joinpath('autoimp.msd')
    if not autoimp_path.exists():
        with open(autoimp_path, 'w') as autoimp_file:
            autoimp_file.write(str(NIST_PATH.joinpath('filespec.fil')))
    # even if it exists, we need to make sure it points to the right file.
    if autoimp_path.read_text().strip() != str(NIST_PATH.joinpath('filespec.fil')):
        with open(autoimp_path, 'w') as autoimp_file:
            autoimp_file.write(str(NIST_PATH.joinpath('filespec.fil')))


    filespec = open(NIST_PATH.joinpath('filespec.fil'),'w')
    filespec.write(str(data_file_path) +' APPEND'+ '\n'+ '10 724')
    filespec.close()
    srcreslt = open(NIST_PATH.joinpath('SRCRESLT.txt'),'w')
    srcreslt.close()
    if os.path.isfile(NIST_PATH.joinpath('SRCREADY.txt')):
        os.remove(NIST_PATH.joinpath('SRCREADY.txt'))
    command = str(NIST_PATH) +r'\NISTMS$.EXE /INSTRUMENT /PAR=2' # 
    # print(command)
    os.system(command)

def _wait_for_NIST_ready():
    '''waits for NIST to finish searching, by checking if the SRCREADY.txt file exists'''
    stp = 0
    while stp == 0:
        try:
            srcready = open(NIST_PATH.joinpath('SRCREADY.txt'),'r')
            srcready.close()
            sleep(2)  # waiting to make sure

            stp = 1
        except FileNotFoundError:
            sleep(10)
            continue
    return

def remove_HLMs():
    '''cleans all the *.HLM files nist search leaves behind'''
    for file in NIST_PATH.glob('*.HLM'):
        file.unlink()

def format_MSP(chromatogram: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    chromatogram = chromatogram.with_columns(
        msp_start=pl.concat_str(
            pl.lit("Name: "),
            pl.col('Peak ID'),
            pl.lit("\n"),
            pl.lit("PRECURSORMZ: "),
            pl.col('Precursor_mz_MSDIAL'),
            pl.lit("\n"),
            pl.lit("Num Peaks: "),
            pl.col('msms_m/z').list.len().cast(pl.String),
            pl.lit("\n"),
            # pl.col('MSMS spectrum') # TODO: remove this, use the proper lists
            ),
        msp_spectrum=pl.struct(
            pl.col('msms_m/z'),
            pl.col('msms_intensity')
            ).map_batches(
                function=lambda spectra: _convert_spectrum_to_text_batch(
                spectra.struct.field('msms_m/z').to_numpy(),
                spectra.struct.field('msms_intensity').to_numpy()
                ),
                return_dtype=pl.String,
                is_elementwise=True)
    ).with_columns(
        msp=pl.concat_str(
            pl.col('msp_start'),
            pl.col('msp_spectrum')
        )
    ).drop(['msp_start','msp_spectrum'])
    return chromatogram

def _convert_spectrum_to_text(mz:np.array,intensity:np.array) -> str:
    mz = mz.astype(np.str_)
    intensity = intensity.astype(np.str_)
    tabs = np.full_like(a=mz,fill_value='   ',dtype=np.str_)
    line_breaks = np.full_like(a=mz,fill_value='\n',dtype=np.str_)
    str_arr = np.stack((mz,tabs,intensity,line_breaks),axis=-1).flatten() # this interleaves them.
    string = ''.join(str_arr)
    return string
_convert_spectrum_to_text_batch = np.vectorize(_convert_spectrum_to_text)




def _read_SRCRESLT_type_file(file_path : str | Path ) -> pl.DataFrame:
    if isinstance(file_path,str):
        file_path = Path(file_path)
    srcrsults = open(file_path, mode='r',encoding='ANSI',errors="ignore")  # ignoring encoding errors
    search_results = srcrsults.read()
    search_results = search_results.split('\nUnknown: ')
    srcrsults.close()
    search_results[0] = search_results[0].strip("Unknown: ")
    NIST_results = pl.DataFrame(search_results,schema={'raw':pl.String})
    NIST_results = NIST_results.with_columns(
        pl.col('raw').str.extract(pattern=r"^(\d+)",group_index=1).str.to_integer().alias('Peak ID'))
    
    
    # NIST_results = NIST_results.select(
    #     pl.col('raw').str.split(by='\n').explode()
    # )
    NIST_results = NIST_results.with_columns(
        pl.col('raw').str.split(by='\n').alias('raw')
    )
    NIST_results = NIST_results.explode('raw')
    # NIST_results.write_csv(file="a",separator= "|")
    NIST_results = NIST_results.with_columns(
        # pl.col('raw').str.extract(pattern=r"^(\d+)",group_index=1).str.to_integer().alias('Peak ID'),
        pl.col('raw').str.extract(pattern=r"; MF:(\s+)(\d+);",group_index=2).cast(pl.Int64)
            .alias('Score'),
        pl.col('raw').str.extract(pattern=r"; RMF:(\s+)(\d+);",group_index=2).cast(pl.Int64)
            .alias('DotProd'), #yes, after testing this was concluded to be the DotProd, not the RevDot.
        pl.col('raw').str.extract(pattern=r"; RMF:(\s+)(\d+);",group_index=2).cast(pl.Int64)
            .alias('revdot'),
        pl.col('raw').str.extract(pattern=r"Id:(\s+)(\d+)\.",group_index=2).cast(pl.Int64)
            .alias('DB_ID'),
        pl.col('raw').str.extract(pattern=r"Lib: <<(.+)>>;",group_index=1)
            .alias('DB_Name'),
        pl.col('raw').str.extract(pattern=r"Hit \d{1}  : <<(.+?)\[",group_index=1)
            .alias('Name_of_NIST_hit'),
        pl.col('raw').str.extract(pattern=r"\s{2}\[(.+?)\]\d?[+-]",group_index=1)
            .alias('precursor_type'),
        pl.col('raw').str.extract(pattern=r"\s{2}\[(.+?)\](\d?[+-])",group_index=2)
            .alias('charge'),    
        pl.col('raw').str.extract(pattern=r"; CAS:(.+?);",group_index=1)
            .alias('CAS'),
        pl.col('raw').str.extract(pattern=r"(QTOF)|(IT\-FT)|(IT\s)|(HCD)|(QQQ)",group_index=0)
            .alias('instrument_type'),
        pl.col('raw').str.extract(pattern=r"P=(\d+(\.\d+)?)>>;",group_index=1).cast(pl.Float64)
            .alias('tag_mass'),
    )

    NIST_results = NIST_results.with_columns(
        precursor_type=pl.concat_str(
            pl.lit("["),
            pl.col('precursor_type'),
            pl.lit("]"),
            pl.col('charge')
        )
    )
    NIST_results = NIST_results.drop(['raw','charge']).sort(by='Peak ID',nulls_last=True)
    # NIST_results.write_csv(file="b",separator= "|")

    NIST_results = filter_NIST_results(NIST_results)
    return NIST_results

def read_NIST_default_location() -> pl.DataFrame:
    NIST_results = _read_SRCRESLT_type_file(file_path=NIST_PATH.joinpath('SRCRESLT.txt'))
    return NIST_results

# filters the results to be only of orbitrap, and from the right libraries
def filter_NIST_results(results:  pl.DataFrame) -> pl.DataFrame:
    filtered = results.filter(
        # pl.col('precursor_type').eq('[M+H]+'),
        pl.col('DB_Name').eq('hr_msms_nist#2') | pl.col('DB_Name').eq('hr_msms_nist') ,
        pl.col('instrument_type').eq('HCD') | pl.col('instrument_type').eq('IT-FT')
    )
    return filtered

#saves the results in a file with added "_NIST_results", and returns its path.
def _save_search_results(data_path: str | Path) -> Path:
    if isinstance(data_path,str):
        data_path = Path(data_path)
    results_path = data_path.parent.joinpath(data_path.stem+'_NIST_results.txt')
    sh.copyfile(src=NIST_PATH.joinpath('SRCRESLT.txt'),dst=results_path)
    return results_path

def plot_result_stats(NIST_results):
    NIST_results =  NIST_results.filter(pl.col('dot').is_not_null())
    hist = NIST_results.select('dot').to_series().to_numpy()
    vals = plt.hist(hist,bins=10,range=(500,1000))
    print(vals)
    plt.show()
    return

def print_result_stats(NIST_results):
    NIST_results =  NIST_results.filter(pl.col('dot').is_not_null())
    hist = NIST_results.select('dot').to_series().to_numpy()
    vals = np.histogram(hist,bins=10,range=(500,1000))
    print(vals)
    return



###### spectral similarity functions for NIST search ######
# yeah this is a bit of a mess, but splitting to multiple functions didn't behave nicely with numba
@njit
def identity_score_NIST_like(
        spec1_mz:np.ndarray, spec1_intensity:np.ndarray,
        spec2_mz:np.ndarray, spec2_intensity:np.ndarray,
        config:search_config
        )-> np.float64:
    """ """
    ms2_mass_tolerance_search = config.ms2_mass_tolerance

    shift=np.float64(0.0)
    mz_power=np.float64(2)
    intensity_power=np.float64(0.5)
    #### find_peak_matches
    lowest_idx = 0
    matches = []
    for i in range(spec1_mz.shape[0]):
        mz1 = spec1_mz[i]
        mz_min = mz1 - ms2_mass_tolerance_search*mz1
        mz_max = mz1 + ms2_mass_tolerance_search*mz1
        for j in range(lowest_idx, spec2_mz.shape[0]):
            mz2 = spec2_mz[j] + shift
            if mz2 > mz_max:
                break
            if mz2 < mz_min:
                lowest_idx = j + 1
            else:

                matches.append((i,j))
    #### end of "find_peak_matches"
    idx1 = np.array([x[0] for x in matches])
    idx2 = np.array([x[1] for x in matches])

    if len(idx1) == 0:
        return 0.0
    score = 0.0
    for i in range(idx1.shape[0]):
        score += (
            np.power(spec1_intensity[idx1[i]]*spec2_intensity[idx2[i]],intensity_power) * 
            np.power((spec1_mz[idx1[i]]*spec2_mz[idx2[i]]),mz_power)
            )

    

    # spec1_norm = cosine_score_self(mz1, spec_intensity1, mz_tol_ppm, mz_power, intensity_power)
    
    # this is "find_peak_matches_self" for spec1
    lowest_idx = 0
    matches = []
    for i in range(spec1_mz.shape[0]):
        mz1 = spec1_mz[i]
        mz_min = mz1 - ms2_mass_tolerance_search*mz1
        mz_max = mz1 + ms2_mass_tolerance_search*mz1
        for j in range(lowest_idx, spec1_mz.shape[0]):
            mz2 = spec1_mz[j]
            if mz2 > mz_max:
                break
            if mz2 < mz_min:
                lowest_idx = j + 1
            else:
                matches.append((i,j))
    idx1 = np.array([x[0] for x in matches])
    idx2 = np.array([x[1] for x in matches])
    spec1_norm = np.float64(0.0)
    for i in range(idx1.shape[0]):
        spec1_norm += (
            np.power(spec1_intensity[idx1[i]]*spec1_intensity[idx2[i]],intensity_power) * 
            np.power((spec1_mz[idx1[i]]*spec1_mz[idx2[i]]),mz_power))

    # this is "find_peak_matches_self" for spec2
    lowest_idx = 0
    matches = []
    for i in range(spec2_mz.shape[0]):
        mz1 = spec2_mz[i]
        mz_min = mz1 - ms2_mass_tolerance_search*mz1
        mz_max = mz1 + ms2_mass_tolerance_search*mz1
        for j in range(lowest_idx, spec2_mz.shape[0]):
            mz2 = spec2_mz[j]
            if mz2 > mz_max:
                break
            if mz2 < mz_min:
                lowest_idx = j + 1
            else:
                matches.append((i,j))
    idx1 = np.array([x[0] for x in matches])
    idx2 = np.array([x[1] for x in matches])
    spec2_norm = np.float64(0.0)
    for i in range(idx1.shape[0]):
        spec2_norm += (
            np.power(spec2_intensity[idx1[i]]*spec2_intensity[idx2[i]],intensity_power) * 
            np.power((spec2_mz[idx1[i]]*spec2_mz[idx2[i]]),mz_power))


    score = score / np.sqrt(spec1_norm*spec2_norm)

    return score
identity_score_NIST_like_batch = np.vectorize(identity_score_NIST_like)



def entropy_score(
        spec1_mz:np.ndarray, spec1_intensity:np.ndarray,
        spec2_mz:np.ndarray, spec2_intensity:np.ndarray,
        config:search_config) -> np.float64:
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
entropy_score_batch=np.vectorize(entropy_score)

if __name__ == "__main__":
    # Example usage
    config = search_config(
        polarity='positive',
        ms1_mass_tolerance=5e-6,
        ms2_mass_tolerance=10e-6,
        search_engine='entropy',
        noise_threshold=0.005,
        NIST_db_path=Path(r"/home/analytit_admin/Data/NIST_hr_msms/NIST_hr_msms.parquet")  # Adjust the path as needed
    )
    try:
        nist = get_NIST(config)
    except:
        nist = pl.scan_parquet(source=config.NIST_db_path).collect_schema()
    print(nist)