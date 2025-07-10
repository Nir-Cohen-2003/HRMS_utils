from dataclasses import dataclass
import polars as pl
from pathlib import Path
from typing import Iterable, Tuple
from ..formats.epa_xlsx import read_xlsx_EPA_list_file_full_format, read_file_idetifiers_only, read_xlsx_EPA_list_file_short_format
import requests
import zipfile

@dataclass
class suspect_list_config:
    epa_db_path:Path|str # this is the path to the EPA database, which is a parquet file.
    exclusion_list:str=None
    def __post_init__(self):
        '''
        Post initialization to ensure epa_db_path is a Path object.
        '''
        if isinstance(self.epa_db_path, str):
            self.epa_db_path = Path(self.epa_db_path)
            # if its not a parquet file, raise an error
        if not isinstance(self.epa_db_path, Path):
            raise TypeError("epa_db_path must be a Path object or a string")
        if not self.epa_db_path.suffix == '.parquet':
            raise ValueError("epa_db_path must point to a .parquet file")
        if not self.epa_db_path.exists():
            raise FileNotFoundError(f"EPA database file {self.epa_db_path} does not exist. Please provide a valid path.")
    def to_dict(self) -> dict:
        '''
        Converts the suspect_list_config to a dictionary.
        '''
        return {
            'epa_db_path':str(self.epa_db_path),
            'exclusion_list':self.exclusion_list
        }
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'suspect_list_config':
        epa_db_path = config_dict.get('epa_db_path')
        if epa_db_path is None:
            raise ValueError("epa_db_path is required and cannot be None.")
        kwargs = {}
        for field_ in cls.__dataclass_fields__:
            if field_ in config_dict and config_dict[field_] is not None:
                kwargs[field_] = config_dict[field_]
        kwargs['epa_db_path'] = epa_db_path
        return cls(**kwargs)



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

def get_EPA(config:suspect_list_config,)-> pl.DataFrame:
    EPA_important_columns = [
    'DTXSID','Haz_level',
    'PREFERRED_NAME','CASRN','INCHIKEY',
    'IUPAC_NAME','SMILES',
    'MOLECULAR_FORMULA','MONOISOTOPIC_MASS',
    'synonyms'
    ]

    
    try :
        EPA = pl.scan_parquet(config.epa_db_path).select(EPA_important_columns).rename(
            {'INCHIKEY':'inchikey_EPA',
             'CASRN':'CAS_EPA',
             'synonyms':'Synonyms_EPA',
             'PREFERRED_NAME':'Name_EPA',
             'MOLECULAR_FORMULA':'Formula_EPA',
             },
            strict=False
        ).collect()
    except pl.exceptions.ColumnNotFoundError as e:
        print(f"Error: Missing {e}. The file {config.epa_db_path} might not be in the expected format or missing some columns.")
        raise 

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


def construct_suspect_list_from_full_lists(
        list_dir:str|Path,
        lists:Iterable[Tuple[str,int]]) -> pl.DataFrame:
    '''
    Constructs a suspect list from a directory containing Excel files with the given names and hazard levels.
    the lists should be excel files with at least the sheet "Main Data" and "Synonym Identifier" in the format of the EPA lists, as opbtained usign the batch serach and export in the comptox dashboard. if you only have the short format lists, you can use the `construct_suspect_list_from_short_lists` function instead, togehther with the dsstox dump files, from which the synonyms can be extracted. an helper function to download the DSSTox dump file is also provided, as `download_DSS_Tox_dump_zip`.
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
    # we want to accept both with .xlsx and without .xlsx, so we will append .xlsx to the names only if they don't already have it
    list_paths = [(name if name.endswith('.xlsx') else f"{name}.xlsx", haz_level) for name, haz_level in lists]
    # now we append the directory to the names
    list_paths = [list_dir / name for name, _ in list_paths]
    if not all(path.exists() for path in list_paths):
        missing_files = [path for path in list_paths if not path.exists()]
        raise FileNotFoundError(f"Files {missing_files} do not exist in the directory {list_dir}")

    # for each list, read the file and add the haz_level to the DataFrame
    suspect_list = []
    for name, haz_level in lists:
        file_path = list_dir / f"{name}.xlsx"
        df = read_xlsx_EPA_list_file_full_format(file_path)
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

def construct_suspect_list_from_short_lists(
        list_dir:str|Path,
        lists:Iterable[Tuple[str,int]],
        epa_dump_dir:Path|str|None,
        ) -> pl.DataFrame:
    """ Constructs a suspect list from a directory containing short format Excel files with the given names and hazard levels."""
    list_dir = Path(list_dir)
    assert list_dir.exists(), f"Directory {list_dir} does not exist"
    assert list_dir.is_dir(), f"{list_dir} is not a directory"
    assert lists, "The lists parameter is empty. Please provide a non-empty iterable of tuples (name, haz_level)."
    assert all(isinstance(item, tuple) and len(item) == 2 for item in lists), "All items in the lists parameter must be tuples of (name, haz_level)."
    assert all(isinstance(name, str) and isinstance(haz_level, int) for name, haz_level in lists), "Each tuple in the lists parameter must contain a string (name) and an integer (haz_level)."
    assert all(0 <= haz_level <= 5 for _, haz_level in lists), "Haz_level must be an integer between 0 and 5 (inclusive)."

    # Accept both with .xlsx and without .xlsx
    list_paths = [(name if name.endswith('.xlsx') else f"{name}.xlsx", haz_level) for name, haz_level in lists]
    list_paths = [list_dir / name for name, _ in list_paths]
    assert all(path.exists() for path in list_paths), f"Files {[path for path in list_paths if not path.exists()]} do not exist in the directory {list_dir}"

    # epa_dump_dir is required
    assert epa_dump_dir is not None, "epa_dump_dir must be provided for short lists."
    epa_dump_dir = Path(epa_dump_dir)
    assert epa_dump_dir.exists(), f"epa_dump_dir {epa_dump_dir} does not exist"
    assert epa_dump_dir.is_dir(), f"{epa_dump_dir} is not a directory"

    # Read the DSSTox dump files, using read_file_idetifiers_only
    dsstox_files = list(epa_dump_dir.glob('DSSToxDump*.xlsx'))
    assert dsstox_files, f"No DSSTox dump files found in {epa_dump_dir}. Please provide the directory containing the DSSTox dump files."
    dsstox = pl.concat(
        [read_file_idetifiers_only(file_path) for file_path in dsstox_files],
        how='vertical'
    ).unique(subset='DTXSID')


    # read the list files and add the haz_level to the DataFrame. read using read_xlsx_EPA_list_file_short_format
    suspect_list = []
    for name, haz_level in lists:
        file_path = list_dir / f"{name}.xlsx"
        df = read_xlsx_EPA_list_file_short_format(file_path)
        if df.is_empty():
            print(f"Warning: {file_path} is empty, skipping.")
            continue
        # join with the dsstox to get the synonyms and inchikey
        df = df.join(dsstox, on='DTXSID', how='left')
        # add the haz_level column
        df = df.with_columns(pl.lit(haz_level).alias('Haz_level'))
        suspect_list.append(df)

    suspect_list_df = pl.concat(suspect_list, how='vertical')
    suspect_list_df = suspect_list_df.sort('Haz_level', descending=False).unique(subset='DTXSID', keep='first')
    return suspect_list_df
    

def download_DSSTox_dump(
        output_dir:Path|str,
        url:str='https://clowder.edap-cluster.com/files/6616d8d7e4b063812d70fc95') -> None:
    '''
    Downloads the DSSTox Dump file from the EPA website and saves it to the specified output path.
    Args:
        output_path (Path|str): The path where the downloaded file will be saved.
        url (str): The URL of the DSSTox Dump file. Default is the EPA's DSSTox Dump fule url, for february 2024.
    Returns:
        None
    '''
    output_dir = Path(output_dir)
    # make sure the output directory exists, by creating it if it doesn't
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # get the zip from the url
    response = requests.get(url)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to download the file from {url}. Status code: {response.status_code}")
    # save the zip file
    zip_file_path = output_dir.joinpath('DSSToxDump.zip')
    with open(zip_file_path, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded DSSTox Dump file to {zip_file_path}")
    # extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted DSSTox Dump file to {output_dir}")

if __name__ == "__main__":
    
    # Example of constructing a suspect list
    list_of_lists = [
        ('PPDB Pesticide Properties DataBase ', 0),
        ('FDA Center for Drug Evaluation & Research - Maximum (Recommended) Daily Dose ', 0),
        ('FDA Orange Book Approved Drug Products', 0),
        ("Agilent PCDL Veterinarian Drug library",0),
        ("Swiss Pesticides and Metabolites from Keifer et al 2019", 0),
        ("PA Office of Pesticide Programs Information Network (OPPIN) ", 0),
        ("EPA Pesticide Chemical Search Database " ,0), 
        
        #pfas are level 2
        ("PFAS structures in DSSTox 2022", 2),
        ("Chemical Contaminants - CCL 5 PFAS subset ", 2),
        ("PFAS structures in DSSTox", 2),
        ("PFASToxic Substances Control Act", 2),


        ('GHS Skin and Eye  II', 3),
        # ("Chemical List MZCLOUD0722-2025-07-03", 0),
    ]
    suspect_list = construct_suspect_list_from_full_lists('/home/analytit_admin/Data/EPA/EPA_lists_full_format/', list_of_lists)
    print(suspect_list.head())
    suspect_list.write_parquet('/home/analytit_admin/Data/EPA/suspect_list.parquet')
    # Example usage
    config = suspect_list_config(
        epa_db_path='/home/analytit_admin/Data/EPA/suspect_list.parquet'
    )
    EPA_df = get_EPA(config)
    print(EPA_df.head())


    # dsstox = pl.read_parquet('/home/analytit_admin/Data/EPA/DSSTox.parquet')
    # print(dsstox)
    # print(dsstox.schema)