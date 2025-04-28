import polars as pl
import numpy as np
from msbuddy import Msbuddy, MsbuddyConfig
from msbuddy.base import MetaFeature, Spectrum
from typing import List, Dict, Any, Optional
import time
import os  # For checking file existence


def create_metafeature_from_row(row: Dict[str, Any], identifier_col: str, precursor_mz_col: str,
                                ms2_mz_col: str, ms2_int_col: str,
                                rt_col: Optional[str] = None, adduct_col: Optional[str] = None, charge_col: Optional[str] = None,
                                default_adduct: str = '[M+H]+', default_charge: int = 1) -> Optional[MetaFeature]:
    """
    Creates a msbuddy.base.MetaFeature object from a dictionary representing a DataFrame row.
    Returns MetaFeature on success, None on error or if MS2 data is invalid/empty.
    Handles exceptions internally.
    """
    feature_id = row.get(identifier_col, 'Unknown')  # Get ID for logging
    required_keys = [identifier_col, precursor_mz_col, ms2_mz_col, ms2_int_col]
    if not all(key in row for key in required_keys):
        missing = [key for key in required_keys if key not in row]
        print(f"Warning: Skipping feature (ID: {feature_id}) due to missing required keys: {missing}")
        return None

    try:
        mz_data = row[ms2_mz_col]
        int_data = row[ms2_int_col]
        if not isinstance(mz_data, (list, np.ndarray, pl.Series)):
            raise TypeError(f"MS2 m/z data is not list-like (got {type(mz_data)})")
        if not isinstance(int_data, (list, np.ndarray, pl.Series)):
            raise TypeError(f"MS2 intensity data is not list-like (got {type(int_data)})")

        ms2_mz = np.array(mz_data, dtype=np.float64)
        ms2_int = np.array(int_data, dtype=np.float64)

        if ms2_mz.ndim != 1 or ms2_int.ndim != 1 or len(ms2_mz) != len(ms2_int):
            raise ValueError(f"MS2 m/z and intensity must be 1D arrays of the same length (lengths: {len(ms2_mz)}, {len(ms2_int)})")

        # Ensure no NaN/inf values which might cause issues later
        if np.any(np.isnan(ms2_mz)) or np.any(np.isinf(ms2_mz)):
            raise ValueError("MS2 m/z array contains NaN or Inf values.")
        if np.any(np.isnan(ms2_int)) or np.any(np.isinf(ms2_int)):
            raise ValueError("MS2 intensity array contains NaN or Inf values.")
        # Ensure intensities are non-negative
        if np.any(ms2_int < 0):
            raise ValueError("MS2 intensity array contains negative values.")

        # Skip if spectrum is empty after validation
        if len(ms2_mz) == 0:
            return None

        ms2_spec = Spectrum(mz_array=ms2_mz, int_array=ms2_int)

        adduct = row.get(adduct_col, default_adduct) if adduct_col else default_adduct
        charge = row.get(charge_col, default_charge) if charge_col else default_charge
        rt = row.get(rt_col) if rt_col else None

        # Ensure required numeric types are correct before creating MetaFeature
        mz_val = float(row[precursor_mz_col])
        charge_val = int(charge)
        rt_val = float(rt) if rt is not None else None

        metafeature = MetaFeature(
            identifier=row[identifier_col],
            mz=mz_val,
            charge=charge_val,
            adduct=adduct,
            rt=rt_val,
            ms1=None,  # Assuming MS1 is not used here
            ms2=ms2_spec
        )
        return metafeature

    except (ValueError, TypeError) as e:
        print(f"Warning: Skipping feature (ID: {feature_id}) due to error during MetaFeature creation: {e}")
        return None
    except Exception as e:
        print(f"Warning: Skipping feature (ID: {feature_id}) due to unexpected error during MetaFeature creation: {e}")
        return None


def annotate_formulas_msbuddy(query_df: pl.DataFrame, config: MsbuddyConfig,
                              identifier_col: str = 'NIST_ID',
                              precursor_mz_col: str = 'PrecursorMZ',
                              ms2_mz_col: str = 'raw_spectrum_mz',
                              ms2_int_col: str = 'raw_spectrum_intensity',
                              rt_col: Optional[str] = None,
                              adduct_col: Optional[str] = None,
                              charge_col: Optional[str] = None) -> pl.DataFrame:
    """
    Annotates molecular formulas using Msbuddy.
    Creates MetaFeature objects sequentially.
    Msbuddy engine handles annotation parallelism based on its config.
    """
    required_input_cols = [identifier_col, precursor_mz_col, ms2_mz_col, ms2_int_col]
    cols_to_select = list(required_input_cols)
    # Add optional columns only if they exist in the DataFrame
    if rt_col and rt_col in query_df.columns: cols_to_select.append(rt_col)
    if adduct_col and adduct_col in query_df.columns: cols_to_select.append(adduct_col)
    if charge_col and charge_col in query_df.columns: cols_to_select.append(charge_col)

    missing_required = [col for col in required_input_cols if col not in query_df.columns]
    if missing_required:
        raise ValueError(f"Input DataFrame must contain columns: {missing_required}")

    print(f"Preparing {len(query_df)} features for Msbuddy sequentially...")
    start_prep = time.time()

    # Select only necessary columns before converting to dicts for efficiency
    df_subset = query_df.select(cols_to_select)
    row_dicts = df_subset.to_dicts()

    metafeature_list: List[MetaFeature] = []
    for row_dict in row_dicts:
        # Pass optional columns correctly, using None if not present in selected columns
        mf = create_metafeature_from_row(
            row=row_dict,
            identifier_col=identifier_col,
            precursor_mz_col=precursor_mz_col,
            ms2_mz_col=ms2_mz_col,
            ms2_int_col=ms2_int_col,
            rt_col=rt_col if rt_col in cols_to_select else None,
            adduct_col=adduct_col if adduct_col in cols_to_select else None,
            charge_col=charge_col if charge_col in cols_to_select else None
        )
        if mf is not None:
            metafeature_list.append(mf)

    print(f"MetaFeature preparation finished in {time.time() - start_prep:.2f} seconds.")
    print(f"Successfully created {len(metafeature_list)} valid MetaFeatures.")

    if not metafeature_list:
        print("No valid features with MS2 spectra to process.")
        return pl.DataFrame()  # Return empty DataFrame if no features are valid

    print(f"Initializing Msbuddy engine with config: parallel={config.parallel}, n_cpu={config.n_cpu}")
    msb_engine = Msbuddy(config)

    print(f"Adding {len(metafeature_list)} valid features to Msbuddy engine...")
    msb_engine.add_data(metafeature_list)

    print("Annotating formulas using Msbuddy engine...")
    start_annotate = time.time()
    msb_engine.annotate_formula()
    print(f"Msbuddy annotation finished in {time.time() - start_annotate:.2f} seconds.")

    print("Retrieving summary...")
    results_list = msb_engine.get_summary()

    if not results_list:
        print("No annotation results generated by Msbuddy.")
        return pl.DataFrame()

    print(f"Retrieved {len(results_list)} results from Msbuddy.")
    # Convert results to Polars DataFrame
    results_df = pl.from_dicts(results_list)

    # Rename identifier column back to original name for joining
    results_df = results_df.rename({"identifier": identifier_col})

    return results_df


if __name__ == "__main__":
    pl.enable_string_cache()
    pl.set_random_seed(42)
    start_time = time.time()

    msb_config = MsbuddyConfig(
        ms_instr="orbitrap",
        ppm=True,
        ms1_tol=5,
        ms2_tol=10,
        halogen=False,
        parallel=True,  # Msbuddy engine parallelism is kept
        n_cpu=16,        # Msbuddy engine parallelism is kept
        timeout_secs=300,
    )

    DATA_PATH = r"/home/analytit_admin/Data/NIST_hr_msms/NIST_hr_msms.parquet"
    ID_COL = 'NIST_ID'
    MZ_COL = 'PrecursorMZ'
    MS2_MZ_COL = 'raw_spectrum_mz'
    MS2_INT_COL = 'raw_spectrum_intensity'
    # Define optional columns (set to None if not used)
    RT_COL = None
    ADDUCT_COL = None
    CHARGE_COL = None

    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        exit()

    try:
        # Dynamically build the list of columns to load
        cols_to_load = [ID_COL, MZ_COL, MS2_MZ_COL, MS2_INT_COL]
        if RT_COL: 
            cols_to_load.append(RT_COL)
        if ADDUCT_COL: 
            cols_to_load.append(ADDUCT_COL)
        if CHARGE_COL: 
            cols_to_load.append(CHARGE_COL)

        NIST_full = pl.read_parquet(DATA_PATH, columns=list(set(cols_to_load))).filter(
            pl.col(MZ_COL).le(900),
        )  # Use set to avoid duplicates
        print(f"Loaded {len(NIST_full)} entries.")
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        exit()

    n_samples = 2500
    if len(NIST_full) < n_samples:
        query_df = NIST_full
    else:
        query_df = NIST_full.sample(n=n_samples, seed=42)
    print(f"Selected {len(query_df)} features for annotation.")

    print("Starting annotation process...")

    try:
        annotation_results = annotate_formulas_msbuddy(
            query_df=query_df,
            config=msb_config,
            identifier_col=ID_COL,
            precursor_mz_col=MZ_COL,
            ms2_mz_col=MS2_MZ_COL,
            ms2_int_col=MS2_INT_COL,
            rt_col=RT_COL,
            adduct_col=ADDUCT_COL,
            charge_col=CHARGE_COL
        )
    except ValueError as ve:  # Catch specific expected errors like missing columns
        print(f"Configuration Error: {ve}")
        annotation_results = pl.DataFrame()  # Ensure it's an empty DF on error
    except Exception as e:
        print(f"An unexpected error occurred during annotation function call: {e}")
        annotation_results = pl.DataFrame()  # Ensure it's an empty DF on error

    if not annotation_results.is_empty():
        print("\nAnnotation Results Summary:")
        print(annotation_results.head())  # Show head for brevity

        try:
            # Ensure the identifier column exists before joining
            if ID_COL not in annotation_results.columns:
                print(f"Error: Identifier column '{ID_COL}' not found in annotation results. Cannot join.")
            else:
                query_with_results = query_df.join(annotation_results, on=ID_COL, how="left")
                print(f"\nQuery DataFrame with Top Annotation (showing first {min(10, len(query_with_results))} rows):")
                # Dynamically build display columns based on what's available
                display_cols = [ID_COL, MZ_COL]
                if 'formula_rank_1' in query_with_results.columns: display_cols.append('formula_rank_1')
                if 'estimated_fdr' in query_with_results.columns: display_cols.append('estimated_fdr')
                if 'formula_rank_2' in query_with_results.columns: display_cols.append('formula_rank_2')
                if 'adduct' in query_with_results.columns: display_cols.append('adduct')  # Check if adduct is in results
                # Ensure all selected display columns actually exist
                display_cols = [col for col in display_cols if col in query_with_results.columns]
                print(query_with_results.select(display_cols).head(10))
        except Exception as e:
            print(f"Error joining or displaying results: {e}")
    else:
        print("\nNo annotation results were generated or an error occurred.")

    print(f"\nTotal script execution time: {time.time() - start_time:.2f} seconds.")
    pl.disable_string_cache()