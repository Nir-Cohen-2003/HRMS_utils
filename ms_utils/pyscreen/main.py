import polars as pl
# import numpy as np
from pathlib import Path
from time import time
from typing import List, Dict,Optional
from ms_utils.interfaces.msdial import get_chromatogram, subtract_blank_frame
from ms_utils.formula_annotation.isotopic_pattern import fits_isotopic_pattern_batch
from ms_utils.pyscreen.pyscreen_config import blank_config, search_config, isotopic_pattern_config, suspect_list_config,pyscreen_config, adducts_neg, adducts_pos
from ms_utils.pyscreen.spectral_search import NIST_search_external , custom_search, get_NIST
from ms_utils.pyscreen.epa import get_EPA
VERBOSE = False
SHORT = False

def cross_with_EPA(chromatogram : pl.LazyFrame | pl.DataFrame, EPA: pl.LazyFrame | pl.DataFrame, config:search_config) -> pl.DataFrame:
    '''
    crosses with EPA based on exact mass and isotopic pattern, with tolerance defined in the global constants file. 
    then fitlers based on Height and haz level.
    every Peak ID can get more than one suspect.'''
    #TODO: make this work with lazy frames, so it can be used with streaming. it fails when there are not matches.
    chromatogram_lf = chromatogram.select(
        [
            'Peak ID',
            'Precursor_mz_MSDIAL',
            'Precursor_type_MSDIAL',
            'Height',
            'ms1_isotopes_m/z', 'ms1_isotopes_intensity'

        ]
    )#.lazy()
    EPA_lf=EPA.select(
        [
            'DTXSID',
            'MONOISOTOPIC_MASS',
            'Haz_level',
            'inchikey_EPA',
            'Formula_EPA'
        ]
    )#.lazy()

    match  config.polarity.lower():
        case "positive": 
            adducts = adducts_pos
        case "negative":
            adducts = adducts_neg
        case _:
            print("ERROR: Must choose operational mode, positive or negative")
            raise Exception("Mode Error: mode was not set, or is not positive/negative!")
        
    adducts_list : List[pl.LazyFrame] = []
    for adduct in adducts.keys():
        adduct_mass = adducts[adduct]
        suspects = chromatogram_lf.join_where(
            EPA_lf,
            pl.col('Precursor_mz_MSDIAL') - adduct_mass < pl.col('MONOISOTOPIC_MASS').mul(1+config.ms1_mass_tolerance),
            pl.col('Precursor_mz_MSDIAL') - adduct_mass >  pl.col('MONOISOTOPIC_MASS').mul(1-config.ms1_mass_tolerance),
            pl.col('Precursor_type_MSDIAL').str.contains(adduct,literal=True)
        )
        adducts_list.append(suspects)
    
    if VERBOSE:
        print('all adduct done')
    
    suspects = pl.concat(adducts_list,how='vertical')
    del adducts_list

    # suspects=suspects.collect(engine="streaming")
    return suspects

def annotate_isotopic_pattern(suspects:pl.DataFrame,config:isotopic_pattern_config) -> pl.DataFrame:
    suspects = suspects.with_columns(
        pl.struct(
            pl.col('ms1_isotopes_m/z'),
            pl.col('ms1_isotopes_intensity'),
            pl.col('Formula_EPA'),
            pl.col('Precursor_mz_MSDIAL')
        ).map_batches(
            function= lambda x: fits_isotopic_pattern_batch(
                x.struct.field('ms1_isotopes_m/z').to_numpy(),
                x.struct.field('ms1_isotopes_intensity').to_numpy(),
                x.struct.field('Formula_EPA'),
                x.struct.field('Precursor_mz_MSDIAL').to_numpy(),
                config
            ), is_elementwise=True, return_dtype=pl.Boolean
        ).alias('isotopic_pattern_match')
    )

    suspects = suspects.drop(
        ['ms1_isotopes_m/z', 'ms1_isotopes_intensity']
    )
    return suspects

def filter_suspects(suspects:pl.DataFrame,chromatogram:pl.DataFrame)->pl.DataFrame:
    Height_threshold = chromatogram.select(pl.col('Height')).min().item(0,0)  # minimal height, represents the limit of sensitivity
    suspects = suspects.filter(
        pl.col('Haz_level').eq(0)  |
        pl.col('Haz_level').eq(1) & pl.col('Height').gt(2.5*Height_threshold) |
        pl.col('Haz_level').eq(2) & pl.col('Height').gt(25*Height_threshold) |
        pl.col('Haz_level').eq(3) & pl.col('Height').gt(250*Height_threshold)
    )
    return suspects


def search_in_NIST(
    suspects: pl.DataFrame,
    NIST: pl.DataFrame,
    chromatogram:pl.DataFrame,
    MSDIAL_file_path : Path | str,
    config:search_config
    ) -> pl.DataFrame:
    
    suspect_to_NIST = chromatogram.join(
        suspects, on='Peak ID', how='semi'
    ).filter(
        pl.col('msms_m/z').is_not_null()
    ).select(
        [
            'Peak ID',
            'Precursor_mz_MSDIAL',
            'msms_intensity',
            'msms_m/z',
        ], 
    )
   
    #### TESTING ONLY #####
    if SHORT:
        suspect_to_NIST = suspect_to_NIST.head(100) 
    ######################

    if config.search_engine.lower() == 'nist': # so through the MS search software, which is slow but gives a graphical interface
        if isinstance(suspect_to_NIST,pl.LazyFrame):
            suspect_to_NIST = suspect_to_NIST.collect(engine="streaming")
        NIST_search_results = NIST_search_external(
            query_df=suspect_to_NIST,
            NIST=NIST,
            MSDIAL_file_path=MSDIAL_file_path,
            ignore_previous_results=True)   
    else:
        NIST_search_results = custom_search(query_df=suspect_to_NIST,NIST=NIST,config=config)

    NIST_search_results = NIST_search_results.join(NIST.drop('Synonyms_NIST',strict=False),on='NIST_ID')
    NIST_search_results = NIST_search_results.select(
        [
            'NIST_ID', 
            'Peak ID', 
            'DotProd', 
            'Name_NIST', 
            'DB_ID', 
            # 'DB_Name', 
            'PrecursorMZ',
            'Precursor_type_NIST', 
            'Instrument_type', 
            'Formula_NIST', 
            'Num_Peaks', 
            'inchikey_NIST',
            'collision_energy_NIST' 
            ]
    )
    return NIST_search_results


def cross_NIST_results_with_suspects(
    NIST_search_results : pl.DataFrame,
    suspects: pl.DataFrame,
    config:search_config
    ):
    suspects_lf = suspects.lazy()
    NIST_search_results_lf = NIST_search_results.lazy()

    # checks if the hit mathces the suspect.
    # pl.Config.set_streaming_chunk_size(size=100) # this lowers maximal memory usage, in principal. I'm really not sure it works.

    suspects_found_in_NIST = suspects_lf.join(
        NIST_search_results_lf,
        left_on=['Peak ID','Precursor_type_MSDIAL','inchikey_EPA'],
        right_on=['Peak ID',"Precursor_type_NIST",'inchikey_NIST'],
        how='inner',
        coalesce=False
    ).drop(
        ['Precursor_type_NIST','inchikey_NIST','Peak ID_right']
        ).rename(
        {
            'Precursor_type_MSDIAL': 'Precursor_type',
            'inchikey_EPA': 'inchikey'
         }
    ).collect()
    #TODO: think what do to when suspects returns with several identifications, some are wiht the same inchikey as suspected some are different.

    if VERBOSE:
        print('joined NIST search results with suspect list')
        print(suspects_found_in_NIST.shape)
        print(suspects_found_in_NIST.columns)
    suspects_found_in_NIST = suspects_found_in_NIST.filter( # filters based on quality of detection, lower threshold for lower Haz index (i.e. "scarier")
        pl.col('DotProd').ge(config.DotProd_threshold[0]) & pl.col('Haz_level').eq(0) |
        pl.col('DotProd').ge(config.DotProd_threshold[1]) & pl.col('Haz_level').eq(1) |
        pl.col('DotProd').ge(config.DotProd_threshold[2]) & pl.col('Haz_level').eq(2) |
        pl.col('DotProd').ge(config.DotProd_threshold[3]) & pl.col('Haz_level').eq(3)
        )
    if VERBOSE:
        print('filtered based on dotprod and Haz level')
        print(suspects_found_in_NIST.shape)
        print(suspects_found_in_NIST.columns)
    
    # suspects_found_in_NIST = suspects_found_in_NIST.lazy()
    suspects_found_in_NIST = suspects_found_in_NIST.sort('DotProd')
    if VERBOSE:
        print('sorted based on dotprod')
        print(suspects_found_in_NIST.shape)
        print(suspects_found_in_NIST.columns)
    # this removes the fit to several energies, since we already made sure nist found the same compound as we suspected it is
    suspects_found_in_NIST = suspects_found_in_NIST.group_by(['Peak ID','DTXSID'],maintain_order=True).last() #TODO:check if using unique is better
    # suspects_found_in_NIST = suspects_found_in_NIST.collect(engine="streaming")
    if VERBOSE:
        print('grouped')
 
    return suspects_found_in_NIST

# returns the suspects with haz level 1 or 2 which without good identification (750 DotProd),
# which are also not present in NIST.
# TODO: fix this to work with my search
def not_in_lib(
        NIST_search_results:pl.DataFrame, 
        suspects:pl.DataFrame,
        NIST:pl.DataFrame,
        EPA:pl.DataFrame,
        config:search_config) -> pl.DataFrame:
    # must be lazy, otherwise fails
    # NIST_search_results = NIST_search_results.lazy()
    # suspects = suspects.lazy()
    # all_compounds_in_NIST = NIST.select(['inchikey_NIST','PrecursorMZ']).lazy()
    all_compounds_in_NIST = NIST.select(['inchikey_NIST','PrecursorMZ']).lazy()
    results_with_good_id = NIST_search_results.filter(pl.col('DotProd').ge(750)).lazy()
    suspects_lf = suspects.lazy()
    EPA_names = EPA.select(['DTXSID','Name_EPA']).lazy()

    suspects_found = suspects_lf.join(
        results_with_good_id, 
        left_on='inchikey_EPA',
        right_on='inchikey_NIST'
        ).filter(
        pl.col('Precursor_mz_MSDIAL').ge(pl.col('PrecursorMZ').mul(1-config.ms1_mass_tolerance)),
        pl.col('Precursor_mz_MSDIAL').le(pl.col('PrecursorMZ').mul(1+config.ms1_mass_tolerance)),
        )
    
    suspects_not_discovered_in_sample=suspects_lf.join(suspects_found,on='Peak ID',how='anti')

    suspects_not_present_in_sample = suspects_not_discovered_in_sample.join( 
        # this finds the suspects that do exist in NIST, but where not found in search, and hence are not present in the given sample
        all_compounds_in_NIST, 
        left_on='inchikey_EPA',
        right_on='inchikey_NIST'
        ).filter(
        pl.col('Precursor_mz_MSDIAL').ge(pl.col('PrecursorMZ').mul(1-config.ms1_mass_tolerance)),
        pl.col('Precursor_mz_MSDIAL').le(pl.col('PrecursorMZ').mul(1+config.ms1_mass_tolerance)),
       )
    
    suspects_not_in_lib = suspects_not_discovered_in_sample.join(suspects_not_present_in_sample,on='Peak ID',how='anti')
    important_suspects_not_in_lib = suspects_not_in_lib.filter(pl.col('Haz_level').le(1))
    important_suspects_not_in_lib = important_suspects_not_in_lib.join(EPA_names,on='DTXSID',how='left')
    return important_suspects_not_in_lib.collect(engine="streaming")


def foramt_results(
        suspects_found_in_NIST:pl.DataFrame,
        NIST:pl.DataFrame,
        EPA:pl.DataFrame,
        chromatogram:pl.DataFrame
        ) -> pl.DataFrame:

    suspects_found_in_NIST = suspects_found_in_NIST.join(
        NIST.select(['NIST_ID','Synonyms_NIST']),on='NIST_ID',how='left')
    suspects_found_in_NIST = suspects_found_in_NIST.join(
        chromatogram.select(['Peak ID','RT (min)','energy_is_too_low', 'energy_is_too_high']),on='Peak ID',how='left')
    suspects_found_in_NIST = suspects_found_in_NIST.join(
        EPA.select(['DTXSID','Synonyms_EPA','CAS_EPA']),on='DTXSID',how='left')
    suspects_found_in_NIST = suspects_found_in_NIST.select(
    ['Peak ID',
    'RT (min)', 
    'Name_NIST', 
    'Haz_level', 
    'DotProd', 
    'Height', 
    'Precursor_mz_MSDIAL', 
    'Formula_NIST', 
    'NIST_ID',
    'collision_energy_NIST',
    'CAS_EPA', 
    'DTXSID', 
    'Synonyms_NIST',
    'Synonyms_EPA',
    'Precursor_type', 
    'MONOISOTOPIC_MASS', 
    'inchikey', 
    'energy_is_too_low', 'energy_is_too_high',
    'isotopic_pattern_match'
    ]
    )

    return suspects_found_in_NIST

def screen_per_file(
        MSDIAL_file_path: str | Path, 
        blank_chromatogram: pl.DataFrame,
        NIST:pl.DataFrame,
        EPA: pl.DataFrame,
        config:pyscreen_config
        ):
    '''
    Takes an MSDIAL export file, reads and cleans it.
    Then crosses it with EPA, and filters based on Height and Haz_level
    then sends all suspects (each peak only once, since this is a time intensive operation) to NIST.
    if previous results are present, uses them.
    reads the reuslts, filters based on molecular identifier and iddentification quality (again- based on haz_level)
    writes the results to excel.'''
    if isinstance(MSDIAL_file_path, str):
        MSDIAL_file_path = Path(MSDIAL_file_path)
    if not MSDIAL_file_path.exists():
        raise FileNotFoundError(f"MSDIAL file {MSDIAL_file_path} does not exist.")

    print(f'MSDIAL_file_path {MSDIAL_file_path}')

    chromatogram = get_chromatogram(MSDIAL_file_path)
    start_blank_subtraction = time()
    if blank_chromatogram is not None: #meaning there is a blank file
        chromatogram = subtract_blank_frame(sample_df=chromatogram,blank_df=blank_chromatogram,config=config.blank)
    end_blank_subtraction = time()
    if VERBOSE:
        print('blank subtraction done')
        print(chromatogram.shape)
        print(chromatogram.schema)
        print(f'time to subtract: {end_blank_subtraction-start_blank_subtraction}')
        chromatogram.write_excel(MSDIAL_file_path.parent.joinpath(MSDIAL_file_path.stem+"_chromatogram.xlsx"))
    if SHORT:
        chromatogram = chromatogram.head(1000)

    start_cross = time()
    suspects = cross_with_EPA(chromatogram=chromatogram,EPA=EPA,config=config.search)
    if suspects.height == 0:
        print(f'No suspects found in {str(MSDIAL_file_path)}, skipping the rest of the process')
        #now write the empty results file, and add the suffix "-no_suspects found"
        empty_results = pl.DataFrame(
            {
                'Peak ID': [],
                'RT (min)': [],
                'Name_NIST': [],
                'Haz_level': [],
                'DotProd': [],
                'Height': [],
                'Precursor_mz_MSDIAL': [],
                'Formula_NIST': [],
                'NIST_ID': [],
                'collision_energy_NIST': [],
                'CAS_EPA': [],
                'DTXSID': [],
                'Synonyms_NIST': [],
                'Synonyms_EPA': [],
                'Precursor_type': [],
                'MONOISOTOPIC_MASS': [],
                'inchikey': [],
                'energy_is_too_low':[],
                'energy_is_too_high':[],
                'isotopic_pattern_match':[]
            }
        )
        empty_results.write_excel(MSDIAL_file_path.parent.joinpath(MSDIAL_file_path.stem+"_no_suspects_found.xlsx"))
        print(f'Wrote empty results file for {str(MSDIAL_file_path)}')
        return
    suspects = annotate_isotopic_pattern(suspects=suspects,config=config.isotopic_pattern)
    #TODO: somewhere here we should send to sirius or msbuddy (or something that I'll write) to get possible formulae, and see if there ius one that matches the isotopic pattern, if the epa formula doesn't match.
    suspects = filter_suspects(suspects=suspects,chromatogram=chromatogram)
    end_cross = time()
    if VERBOSE:
        print('got suspects')
        print(suspects.shape)
        print(suspects.columns)
        print(f'time to cross: {end_cross-start_cross}')
        suspects.write_excel(MSDIAL_file_path.parent.joinpath(MSDIAL_file_path.stem+"_suspects.xlsx"))

    start_search = time()

    NIST_search_results = search_in_NIST(
        suspects=suspects, 
        NIST=NIST,
        chromatogram=chromatogram,
        MSDIAL_file_path=MSDIAL_file_path,
        config=config.search)
    end_search = time()
    if VERBOSE:
        print('searched against NIST')
        print(NIST_search_results.shape)
        print(NIST_search_results.columns)
        print(f'time to search NIST: {end_search-start_search}')
        NIST_search_results.write_excel(MSDIAL_file_path.parent.joinpath(MSDIAL_file_path.stem+"_NIST_search_results.xlsx"))

    start_result_crossing = time()
    suspects_found_in_NIST = cross_NIST_results_with_suspects(
        NIST_search_results=NIST_search_results,
        suspects=suspects,
        config=config.search)
    end_result_crossing = time()
    if VERBOSE:
        print('crossed NIST results with suspects')
        print(suspects_found_in_NIST.shape)
        print(suspects_found_in_NIST.columns)
        print(f'time to cross results: {end_result_crossing-start_result_crossing}')
        suspects_found_in_NIST.write_excel(MSDIAL_file_path.parent.joinpath(MSDIAL_file_path.stem+"_suspects_found_in_NIST.xlsx"))

    suspects_found_in_NIST = foramt_results(
        suspects_found_in_NIST=suspects_found_in_NIST,
        NIST=NIST,
        EPA=EPA,
        chromatogram=chromatogram)

    start_writing_results= time()
    suspects_found_in_NIST.sort(
            by=['Haz_level','DotProd'],descending=[False,True]
        ).write_excel(MSDIAL_file_path.parent.joinpath(MSDIAL_file_path.stem+"_results.xlsx"))
    del suspects_found_in_NIST
    end_writing_results = time()
    if VERBOSE:
        print(f'time to write results: {end_writing_results-start_writing_results}')
    
    start_not_in_lib = time()
    suspects_not_in_lib = not_in_lib(
        NIST_search_results=NIST_search_results,
        suspects=suspects,
        NIST=NIST,
        EPA=EPA,
        config=config.search)
    if VERBOSE:
        print('not in lib')
        print(suspects_not_in_lib.shape)
        print(suspects_not_in_lib.columns)
    end_not_in_lib = time()
    if VERBOSE:
        print(f'time to do not in lib: {end_not_in_lib-start_not_in_lib}')

    suspects_not_in_lib.write_excel(MSDIAL_file_path.parent.joinpath(MSDIAL_file_path.stem+"_not_in_lib.xlsx"))
    if VERBOSE:
        print('wrote not in lib')
    return

# Takes a list of files (as paths), a blank (as path) a the config for this run.
# Then splits the work using the "screen_per_file" function.
# currently impossible to make parrallel, since NIST cant work this way and its the most time consuming step by a mile.
# if not using NIST, it can be parallelized, but then its already fast enough, and polars already does it in parallel.
def main(
        sample_file_paths:  list[str] | list[Path] , 
        blank_file_path : str | Path | None, 
        config:pyscreen_config | dict
        ):
    
    if isinstance(config,dict):
        config = pyscreen_config(**config) ## converts it to the correct format and sets default values. also works when reading from yaml
    if not isinstance(config,pyscreen_config):
        raise Exception("config must be a pyscreen_config object or a dict that can be converted to one")
    if blank_file_path in sample_file_paths:
        sample_file_paths.remove(blank_file_path)
    # tells you what it got.
    print('sample files:')
    for path in sample_file_paths:
        print(str(path))
    print('blank file')
    print(str(blank_file_path))
    print('config:')
    print(config)
    

    NIST = get_NIST(config=config.search)
    EPA = get_EPA(config=config.suspect_list)

    if VERBOSE:
        print('fetched DBs')
    # some error checking
    

    if len(sample_file_paths) <1 :
        raise Exception("Error: must have at least one sample file")


    if blank_file_path is not None:
        blank_chromatogram = get_chromatogram(path=blank_file_path)
    else:
        blank_chromatogram = None

    for MSDIAL_file_path in sample_file_paths:
        try:
            screen_per_file(MSDIAL_file_path=MSDIAL_file_path,blank_chromatogram=blank_chromatogram,NIST=NIST,EPA=EPA,config=config)
        except Exception as e:
            print("error in file: "+ str(MSDIAL_file_path))
            print(e)
            raise e


if __name__ == "__main__":
    start = time()
    sample_dir = Path(r"")
    sample_file_paths = list(sample_dir.glob(pattern=r'*.txt',case_sensitive=True))
    sample_file_paths = [
        "/home/analytit_admin/Data/raw_data/wine/0717_kinetex_wine_50_4min_pos_IDA_A1.txt",
        "/home/analytit_admin/Data/raw_data/wine/0717_kinetex_wine_50_4min_pos_IDA_B1.txt",  
        "/home/analytit_admin/Data/raw_data/wine/0717_kinetex_wine_50_4min_pos_IDA_C1.txt",
        "/home/analytit_admin/Data/raw_data/wine/0717_kinetex_wine_50_4min_pos_IDA_D1.txt",
        "/home/analytit_admin/Data/raw_data/wine/0717_kinetex_wine_50_4min_pos_IDA_E1.txt",
        "/home/analytit_admin/Data/raw_data/wine/0717_kinetex_wine_50_4min_pos_IDA_F1.txt",
              
    ]

    # blank_file_path = Path(r"/home/analytit_admin/Data/iibr_data/250515_003.txt")
    blank_file_path = None
    config_path = Path("/home/analytit_admin/pyscreen_test_method.yaml")
    config = pyscreen_config.from_yaml(config_path)
    main(
        sample_file_paths=sample_file_paths,
        blank_file_path=blank_file_path,
        config=config)

    print("time: "+str(time()-start))