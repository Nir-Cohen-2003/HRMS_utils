from dataclasses import dataclass
import polars as pl
from pathlib import Path
from typing import Iterable, Tuple
from ms_utils.formats.epa_xlsx import read_xlsx_EPA_list_file

@dataclass
class suspect_list_config:
    exclusion_list:str=None



def exclude_boring_compounds(EPA:pl.DataFrame) -> pl.DataFrame:
    '''
    NON VALIDATED! DO NOT USE! this is Nir's quick-n-dirty exclusion list
    overwrites the Haz_level of the natural compounds to be 3, mathces based on inchikey and DTXSID
    '''
    print("NON VALIDATED! DO NOT USE! this is Nir's quick-n-dirty exclusion list")
    boring_compounds = pl.read_excel(source=Path(r"boring_compounds.xlsx"))
    boring_compounds = boring_compounds.select(['DTXSID','inchikey']).unique()
    boring_compounds = boring_compounds.rename({
        'inchikey' :'inchikey_EPA'
    },strict=False)
    EPA_boring = EPA.join(boring_compounds,on=['DTXSID','inchikey_EPA'],how='inner')
    EPA_boring = EPA_boring.with_columns(
        pl.lit(value=3,dtype=pl.Int64).alias('Haz_level')
    )
    EPA = EPA.join(boring_compounds,on=['DTXSID','inchikey_EPA'],how='anti')
    EPA = pl.concat([EPA,EPA_boring])
    return EPA

def get_EPA(config:suspect_list_config,epa_path:Path|str='EPA_with_Haz_level')-> pl.DataFrame:
    EPA_important_columns = [
    'DTXSID','Haz_level','MS_READY_SMILES',
    'PREFERRED NAME','CASRN','INCHIKEY',
    'IUPAC NAME','SMILES',
    'MOLECULAR FORMULA','MONOISOTOPIC MASS',
    'synonyms'
    ]
    #make sure the path is a Path object
    if isinstance(epa_path, str):
        epa_path = Path(epa_path)
    if not epa_path.exists():
        raise FileNotFoundError(f"EPA file {epa_path} does not exist. Please provide a valid path.")
    #add the parquet file extension if it is not already there
    if not epa_path.suffix == '.parquet':
        epa_path = epa_path.with_suffix('.parquet')
    EPA = pl.scan_parquet(epa_path).select(EPA_important_columns).rename(
            {'INCHIKEY':'inchikey_EPA',
             'CASRN':'CAS_EPA',
             'synonyms':'Synonyms_EPA',
             'PREFERRED NAME':'Name_EPA',
             'MOLECULAR FORMULA':'Formula_EPA',
             },
        strict=False
    ).collect()

    if config.exclusion_list is None:
        pass
    else:
        match config.exclusion_list.lower():
            case 'boring_compounds':
                EPA = exclude_boring_compounds(EPA)
            case None:
                pass
            case 'none':
                pass
            case _:
                print(config.exclusion_list)
                raise Exception("invalid exclusion list given")
    return EPA


# TODO: add the function for construction of the suspect list. might add a GUI for it later.
# takes an iterable of tuples (name, haz_level) and returns a polars DataFrame, which can be written to disk or used directly.
def construct_suspect_list(list_dir:str|Path,lists:Iterable[Tuple[str,int]]) -> pl.DataFrame:
    '''
    Constructs a suspect list from a directory containing Excel files with the given names and hazard levels.
    Args:
        list_dir (str|Path): The directory containing the Excel files.
        lists (Iterable[Tuple[str, int]]): An iterable of tuples, where each tuple contains a name (str) and a hazard level (int).
            The hazard level must be an integer between 0 and 5 (inclusive). lower haz level means more hazardous, yes this is confusing.
    Returns:
        pl.DataFrame: A DataFrame containing the suspect list, with columns for DTXSID, Haz_level, and other relevant information.
    Raises:
        FileNotFoundError: If the specified directory does not exist.
        NotADirectoryError: If the specified path is not a directory.
        ValueError: If the lists parameter is empty or contains invalid tuples.
        ValueError: If any tuple in the lists parameter does not contain a string and an integer, or if the haz_level is not between 0 and 5.
    Usage:
    >>> from ms_utils.pyscreen.epa import construct_suspect_list
    >>> suspect_list = construct_suspect_list('/path/to/lists', [('list1', 2), ('list2', 3)])
    '''
    #assert the path points to an exisiting directory
    list_dir = Path(list_dir)
    if not list_dir.exists():
        raise FileNotFoundError(f"Directory {list_dir} does not exist")
    if not list_dir.is_dir():
        raise NotADirectoryError(f"{list_dir} is not a directory")

    # make sure that the list are not empty, and that they are tuples of (name, haz_level)
    if not lists:
        raise ValueError("The lists parameter is empty. Please provide a non-empty iterable of tuples (name, haz_level).")
    if not all(isinstance(item, tuple) and len(item) == 2 for item in lists):
        raise ValueError("All items in the lists parameter must be tuples of (name, haz_level).")
    if not all(isinstance(name, str) and isinstance(haz_level, int) for name, haz_level in lists):
        raise ValueError("Each tuple in the lists parameter must contain a string (name) and an integer (haz_level).")
    if not all(0 <= haz_level <= 5 for _, haz_level in lists):
        raise ValueError("Haz_level must be an integer between 0 and 5 (inclusive).")
    #make sure that all banes exist in the directory
    list_paths = [list_dir / f"{name}.xlsx" for name, _ in lists]
    if not all(path.exists() for path in list_paths):
        missing_files = [path for path in list_paths if not path.exists()]
        raise FileNotFoundError(f"Files {missing_files} do not exist in the directory {list_dir}")

    # for each list, read the file and add the haz_level to the DataFrame
    suspect_list = []
    for name, haz_level in lists:
        file_path = list_dir / f"{name}.xlsx"
        df = read_xlsx_EPA_list_file(file_path)
        if df.is_empty():
            print(f"Warning: {file_path} is empty, skipping.")
            continue
        df = df.with_columns(pl.lit(haz_level).alias('Haz_level'))
        suspect_list.append(df)
    # concatenate all DataFrames into one
    suspect_list_df = pl.DataFrame(pl.concat(suspect_list, how='vertical'))
    # remove duplicates based on DTXSID, make sure the Haz_level is the minimum of the duplicates
    suspect_list_df = suspect_list_df.sort('Haz_level',descending=False).unique(subset='DTXSID', keep='first')
    return suspect_list_df


if __name__ == "__main__":
    
    # Example of constructing a suspect list
    list_of_lists = [
        ('PPDB Pesticide Properties DataBase ', 0),
        ('GHS Skin and Eye  II', 3),
        ('Agilent PCDL Veterinarian Drug library', 1)
    ]
    suspect_list = construct_suspect_list('/home/analytit_admin/Data/EPA/EPA_lists/', list_of_lists)
    print(suspect_list.head())
    suspect_list.write_parquet('/home/analytit_admin/Data/EPA/suspect_list.parquet')
    # Example usage
    config = suspect_list_config(exclusion_list='boring_compounds')
    EPA_df = get_EPA(config, epa_path='/home/analytit_admin/Data/EPA/suspect_list.parquet')
    print(EPA_df.head())