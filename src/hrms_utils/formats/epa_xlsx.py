import time
import polars as pl
from pathlib import Path
from ..formula_annotation.utils import formula_to_array

EPA_main_data_schema_initial = dict({
    'INPUT': pl.String,
    'FOUND_BY': pl.String,
    'DTXSID': pl.String,
    'PREFERRED_NAME': pl.String,
    'DTXCID': pl.String,
    'CASRN': pl.String,
    'INCHIKEY': pl.String,
    'IUPAC_NAME': pl.String,
    'SMILES': pl.String,
    'INCHI_STRING': pl.String,
    'MOLECULAR_FORMULA': pl.String,
    'AVERAGE_MASS': pl.Float64,
    'MONOISOTOPIC_MASS': pl.Float64,
    'QC_LEVEL': pl.Int64,
    'SAFETY_DATA': pl.String,
    'EXPOCAST': pl.String,
    'DATA_SOURCES': pl.Int64,
    'TOXVAL_DATA': pl.String,
    'NUMBER_OF_PUBMED_ARTICLES': pl.Int64,
    'PUBCHEM_DATA_SOURCES': pl.Int64,
    'CPDAT_COUNT': pl.Int64,
    'IRIS_LINK': pl.String,
    'PPRTV_LINK': pl.String,
    'WIKIPEDIA_ARTICLE': pl.String,
    'QC_NOTES': pl.String,
    'BIOCONCENTRATION_FACTOR_TEST_PRED': pl.Float64,
    'BOILING_POINT_DEGC_TEST_PRED': pl.Float64,
    '48HR_DAPHNIA_LC50_MOL/L_TEST_PRED': pl.Float64,
    'DENSITY_G/CM^3_TEST_PRED': pl.Float64,
    'DEVTOX_TEST_PRED': pl.Float64,
    '96HR_FATHEAD_MINNOW_MOL/L_TEST_PRED': pl.Float64,
    'FLASH_POINT_DEGC_TEST_PRED': pl.Float64,
    'MELTING_POINT_DEGC_TEST_PRED': pl.Float64,
    'AMES_MUTAGENICITY_TEST_PRED': pl.Float64,
    'ORAL_RAT_LD50_MOL/KG_TEST_PRED': pl.Float64,
    'SURFACE_TENSION_DYN/CM_TEST_PRED': pl.Float64,
    'THERMAL_CONDUCTIVITY_MW/(M*K)_TEST_PRED': pl.Float64,
    'TETRAHYMENA_PYRIFORMIS_IGC50_MOL/L_TEST_PRED': pl.Float64,
    'VISCOSITY_CP_CP_TEST_PRED': pl.Float64,
    'VAPOR_PRESSURE_MMHG_TEST_PRED': pl.Float64,
    'WATER_SOLUBILITY_MOL/L_TEST_PRED': pl.Float64,
    'ATMOSPHERIC_HYDROXYLATION_RATE_(AOH)_CM3/MOLECULE*SEC_OPERA_PRED': pl.Float64,
    'BIOCONCENTRATION_FACTOR_OPERA_PRED': pl.Float64,
    'BIODEGRADATION_HALF_LIFE_DAYS_DAYS_OPERA_PRED': pl.Float64,
    'BOILING_POINT_DEGC_OPERA_PRED': pl.Float64,
    'HENRYS_LAW_ATM-M3/MOLE_OPERA_PRED': pl.Float64,
    'OPERA_KM_DAYS_OPERA_PRED': pl.Float64,
    'OCTANOL_AIR_PARTITION_COEFF_LOGKOA_OPERA_PRED': pl.Float64,
    'SOIL_ADSORPTION_COEFFICIENT_KOC_L/KG_OPERA_PRED': pl.Float64,
    'OCTANOL_WATER_PARTITION_LOGP_OPERA_PRED': pl.Float64,
    'MELTING_POINT_DEGC_OPERA_PRED': pl.Float64,
    'OPERA_PKAA_OPERA_PRED': pl.Float64,
    'OPERA_PKAB_OPERA_PRED': pl.Float64,
    'VAPOR_PRESSURE_MMHG_OPERA_PRED': pl.Float64,
    'WATER_SOLUBILITY_MOL/L_OPERA_PRED': pl.Float64,
    'LOGD5.5': pl.Float64,
    'LOGD7.4': pl.Float64,
    'READY_BIO_DEG': pl.Int64,
    'EXPOCAST_MEDIAN_EXPOSURE_PREDICTION_MG/KG-BW/DAY': pl.Float64,
    'NHANES': pl.String,
    'TOXCAST_NUMBER_OF_ASSAYS/TOTAL': pl.String,
    'TOXCAST_PERCENT_ACTIVE': pl.Float64
})

def read_file_idetifiers_only(file_path: Path | str):
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    assert isinstance(file_path, (Path, str)), "file_path must be a Path or str"
    assert file_path.exists(), f"file_path does not exist: {file_path}"
    assert file_path.suffix.lower() == '.xlsx', "file_path must be an Excel file with .xlsx extension"

    df = pl.read_excel(file_path, schema_overrides={
        'AVERAGE_MASS': pl.Float64,
        'MONOISOTOPIC_MASS': pl.Float64})
    df = df.with_columns(
        pl.col('DTXSID').str.strip_prefix('DTXSID').cast(pl.Int64).alias('DTXSID'),
        pl.col('IDENTIFIER').str.split(by='|').alias('synonyms')
        ) 
    df = df.drop('IDENTIFIER')
    df = formula_to_array(df, input_col_name='MOLECULAR_FORMULA', output_col_name='MOLECULAR_FORMULA_array')
    return df

def read_xlsx_EPA_list_file_short_format(file_path: Path | str):
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    assert isinstance(file_path, (Path, str)), "file_path must be a Path or str"
    assert file_path.exists(), f"file_path does not exist: {file_path}"
    assert file_path.suffix.lower() == '.xlsx', "file_path must be an Excel file with .xlsx extension"
    data = pl.read_excel(file_path).with_columns(
        pl.col('DTXSID').str.strip_prefix('DTXSID').cast(pl.Int64).alias('DTXSID'))
    return data
    
def read_xlsx_EPA_list_file_full_format(file_path):
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    assert isinstance(file_path, (Path, str)), "file_path must be a Path or str"
    assert file_path.exists(), f"file_path does not exist: {file_path}"
    assert file_path.suffix.lower() == '.xlsx', "file_path must be an Excel file with .xlsx extension"
    try:
        main_df = pl.read_excel(file_path, sheet_name=['Main Data'], schema_overrides=EPA_main_data_schema_initial)
        main_df = main_df['Main Data']
    except (KeyError, ValueError):
        raise KeyError("The Excel file must contain a sheet named 'Main Data' with the expected schema. Are you sure this is a full format list (with Main Data and Synonym Identifier sheets), or a short format list (without synonyms, and only one sheet)?")
    try:
        synonym_df = pl.read_excel(file_path, sheet_name=['Synonym Identifier'], infer_schema_length=None)
        synonym_df = synonym_df['Synonym Identifier']
    except (KeyError, ValueError):
        raise KeyError("The Excel file must contain a sheet named 'Synonym Identifier'. are you sure this is a full format list (with Main Data and Synonym Identifier sheets), or a short format list (without synonyms, and only one sheet)?")
    main_df = Main_sheet_cleaner(main_df)
    synonym_df = Synonym_sheet_cleaner(synonym_df)
    combined_df = main_df.join(synonym_df, left_on='PREFERRED_NAME', right_on='SEARCHED_CHEMICAL', how='left')
    combined_df = combined_df.unique(subset='DTXSID')
    return combined_df

def Main_sheet_cleaner(main_df):
    main_df = main_df.with_columns( # this mostly just convert the columns where Y/N are to boolean values
        pl.col('SAFETY_DATA').str.contains('Y').alias('SAFETY_DATA'),
        pl.col('EXPOCAST').str.contains('Y').alias('EXPOCAST'),
        pl.col('TOXVAL_DATA').str.contains('Y').alias('TOXVAL_DATA'),
        pl.col('PPRTV_LINK').str.contains('Y').alias('PPRTV_LINK'),
        pl.col('WIKIPEDIA_ARTICLE').str.contains('Y').alias('WIKIPEDIA_ARTICLE'),
        pl.col('IRIS_LINK').str.contains('Y').alias('IRIS_LINK'),
        pl.col('NHANES').str.contains('Y').alias('NHANES'),
        pl.col('DTXSID').str.strip_prefix('DTXSID').cast(pl.Int64).alias('DTXSID'),
        #pl.col('DTXCID').str.strip_prefix('DTXCID').cast(pl.Int64).alias('DTXCID'),
        ) 
    
    main_df = formula_to_array(main_df, input_col_name='MOLECULAR_FORMULA', output_col_name='MOLECULAR_FORMULA_array')
    return main_df

def Synonym_sheet_cleaner(synonym_df):
    synonym_df = synonym_df.with_columns(pl.col('IDENTIFIER').str.split(by='|').alias('synonyms'))
    synonym_df = synonym_df.drop('IDENTIFIER')
    return synonym_df




if __name__ == '__main__':
    start = time.time()
    # file_path = Path(r'D:\Nir\Data_from_Nitzan\EPA_format\SWGDRUG Mass Spectral Library Chemical Collection .xlsx')
    # data = read_xlsx_EPA_list_file(file_path)
    # file_path = Path(r'D:\Nir\Data_from_Nitzan\EPA_BY_DATA\identifiers_only\DSSToxDump1.xlsx')
    # data = read_file_idetifiers_only(file_path)
    # print(data.schema)
    # print(data.shape)
    # print(data.head(10))#'MOLECULAR_FORMULA_list',
    # print('Time:', time.time()-start)

    path = "/home/analytit_admin/Data/EPA/EPA_lists_short_format/Chemical List MZCLOUD0722-2025-07-03.xlsx"
    data = read_xlsx_EPA_list_file_short_format(path)
    print(data)
    data = read_xlsx_EPA_list_file_full_format(path)
    print(data)