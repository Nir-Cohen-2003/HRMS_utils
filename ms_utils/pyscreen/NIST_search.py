import polars as pl
import numpy as np
from pathlib import Path
import os
from spectral_similarity import identity_score_NIST_like_batch, entropy_score_batch
from time import sleep
import shutil as sh 
from pyscreen_config import search_config

NIST_path = Path(r"")

def get_NIST(condig:search_config) -> pl.DataFrame:
    NIST = pl.scan_parquet(source=r"NIST_DB.parquet")
    NIST = NIST.select([
    'Name',
    'NIST_ID',
    'DB_ID',
    'DB_Name',
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
    

    if condig.polarity.lower()=="positive":
        mode="P"
    elif condig.polarity.lower()=="negative":
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

# reads the current results file, saves it if needed (meaning no results were saved for the given file.)
# the path it takes is the path of the MSDIAL file it (presumably) reads the results of.
# waits for NIST to finish.
# also note- it first copies and saves the file, then reads from the sved file.
def read_NIST(file_path : str | Path) -> pl.DataFrame:
    _wait_for_NIST_ready()
    results = _read_NIST_eager(MSDIAL_file_path=file_path)
    return results

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
    data_file_path = NIST_path.joinpath('data2nist.msp')
    msp_file = open(data_file_path,'w')
    msp_file.write(msp_to_print[0])
    msp_file.close()


    filespec = open(NIST_path.joinpath('filespec.fil'),'w')
    filespec.write(str(data_file_path) +' APPEND'+ '\n'+ '10 724')
    filespec.close()
    srcreslt = open(NIST_path.joinpath('SRCRESLT.txt'),'w')
    srcreslt.close()
    if os.path.isfile(NIST_path.joinpath('SRCREADY.txt')):
        os.remove(NIST_path.joinpath('SRCREADY.txt'))
    command = str(NIST_path) +r'\NISTMS$.EXE /INSTRUMENT /PAR=2' # 
    # print(command)
    os.system(command)

def _wait_for_NIST_ready():
    '''waits for NIST to finish searching, by checking if the SRCREADY.txt file exists'''
    stp = 0
    while stp == 0:
        try:
            srcready = open(NIST_path.joinpath('SRCREADY.txt'),'r')
            srcready.close()
            sleep(2)  # waiting to make sure

            stp = 1
        except FileNotFoundError:
            sleep(10)
            continue
    return

def remove_HLMs():
    '''cleans all the *.HLM files nist search leaves behind'''
    for file in NIST_path.glob('*.HLM'):
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
    NIST_results = _read_SRCRESLT_type_file(file_path=NIST_path.joinpath('SRCRESLT.txt'))
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
    sh.copyfile(src=NIST_path.joinpath('SRCRESLT.txt'),dst=results_path)
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
