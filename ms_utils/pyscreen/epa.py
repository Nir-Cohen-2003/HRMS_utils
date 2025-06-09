from dataclasses import dataclass
import polars as pl
from pathlib import Path

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

def get_EPA(config:suspect_list_config)-> pl.DataFrame:
    EPA_important_columns = [
    'DTXSID','Haz_level','MS_READY_SMILES',
    'PREFERRED NAME','CASRN','INCHIKEY',
    'IUPAC NAME','SMILES',
    'MOLECULAR FORMULA','MONOISOTOPIC MASS',
    'synonyms'
    ]
    EPA = pl.scan_parquet(r"EPA_with_Haz_level.parquet").select(EPA_important_columns).rename(
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