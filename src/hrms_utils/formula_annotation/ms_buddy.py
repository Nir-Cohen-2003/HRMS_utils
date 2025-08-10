import polars as pl
import numpy as np
from msbuddy import Msbuddy, MsbuddyConfig
from msbuddy.base import MetaFeature, Spectrum
from typing import List, Dict, Any, Optional, Literal # Added Literal
import time
import os
from dataclasses import dataclass

@dataclass
class msbuddyInterfaceConfig:
    data_path: Optional[str]
    identifier_col: str = "NIST_ID"
    precursor_mz_col: str = "PrecursorMZ"
    ms2_mz_col: str = "raw_spectrum_mz"
    ms2_int_col: str = "raw_spectrum_intensity"
    rt_col: Optional[str] = None
    adduct_col: Optional[str] = None
    charge_col: Optional[str] = None
    ms1_isotope_mz_col: Optional[str] = None
    ms1_isotope_int_col: Optional[str] = None
    polarity_col: Optional[str] = None  # New: column name for polarity
    default_polarity: Literal["positive", "negative"] = "positive" # New: 'positive' or 'negative'

    def __post_init__(self):
        # Normalize and validate default_polarity
        if isinstance(self.default_polarity, str):
            self.default_polarity = self.default_polarity.lower() # type: ignore
        if self.default_polarity not in ["positive", "negative"]:
            raise ValueError("default_polarity must be 'positive' or 'negative'")

def create_metafeature_from_row(
    row: Dict[str, Any],
    interface_config: msbuddyInterfaceConfig
    # default_adduct: str = "[M+H]+", # Removed
    # default_charge: int = 1, # Removed
) -> Optional[MetaFeature]:
    """
    Creates a msbuddy.base.MetaFeature object from a dictionary representing a DataFrame row.
    Returns MetaFeature on success, None on error or if MS2 data is invalid/empty.
    Handles exceptions internally.
    """
    # pull all column names from interface_cfg
    id_col = interface_config.identifier_col
    pmz_col = interface_config.precursor_mz_col
    mz2_col = interface_config.ms2_mz_col
    int2_col = interface_config.ms2_int_col
    rt_col = interface_config.rt_col
    add_col = interface_config.adduct_col
    chg_col = interface_config.charge_col
    ms1_iso_mz_c = interface_config.ms1_isotope_mz_col
    ms1_iso_int_c = interface_config.ms1_isotope_int_col
    pol_col = interface_config.polarity_col # New

    feature_id = row.get(id_col, "Unknown")  # Get ID for logging
    required = [id_col, pmz_col, mz2_col, int2_col]
    if not all(k in row for k in required):
        print(f"Warning: skipping {feature_id}, missing {set(required)-row.keys()}")
        return None

    try:
        mz_data = row[mz2_col]
        int_data = row[int2_col]
        if not isinstance(mz_data, (list, np.ndarray, pl.Series)):
            raise TypeError(f"MS2 m/z data is not list-like (got {type(mz_data)})")
        if not isinstance(int_data, (list, np.ndarray, pl.Series)):
            raise TypeError(
                f"MS2 intensity data is not list-like (got {type(int_data)})"
            )

        ms2_mz = np.array(mz_data, dtype=np.float64)
        ms2_int = np.array(int_data, dtype=np.float64)

        if ms2_mz.ndim != 1 or ms2_int.ndim != 1 or len(ms2_mz) != len(ms2_int):
            raise ValueError(
                f"MS2 m/z and intensity must be 1D arrays of the same length (lengths: {len(ms2_mz)}, {len(ms2_int)})"
            )

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
            # print(f"Warning: skipping {feature_id} due to empty MS2 spectrum after validation.") # Optional: more verbose logging
            return None

        ms2_spec = Spectrum(mz_array=ms2_mz, int_array=ms2_int)

        # Determine is_positive_mode based on polarity_col or default_polarity
        is_positive_mode: bool
        if pol_col and pol_col in row and row[pol_col] is not None:
            polarity_val_str = str(row[pol_col]).lower()
            if polarity_val_str == "positive":
                is_positive_mode = True
            elif polarity_val_str == "negative":
                is_positive_mode = False
            else:
                print(f"Warning: Invalid polarity value '{row[pol_col]}' for feature {feature_id}. Using default polarity '{interface_config.default_polarity}'.")
                is_positive_mode = interface_config.default_polarity == "positive"
        else:
            is_positive_mode = interface_config.default_polarity == "positive"

        # Determine adduct string
        adduct_str: str
        if add_col and add_col in row and row[add_col] is not None:
            adduct_str = str(row[add_col])
        else:
            adduct_str = "[M+H]+" if is_positive_mode else "[M-H]-"
            # Optional: Log if default adduct is used
            # print(f"Feature {feature_id}: No adduct column '{add_col}' or value missing. Using default adduct '{adduct_str}' based on polarity: {'positive' if is_positive_mode else 'negative'}.")

        # Determine charge_val
        charge_val: int
        if chg_col and chg_col in row and row[chg_col] is not None:
            try:
                charge_val = int(row[chg_col])
                # Optional: Basic consistency check with is_positive_mode, can be noisy
                # if (charge_val > 0 and not is_positive_mode) or \
                #    (charge_val < 0 and is_positive_mode) and \
                #    charge_val != 0 : # Allow charge 0 for neutral M
                #     print(f"Warning: Charge {charge_val} from column '{chg_col}' might be inconsistent with determined polarity for feature {feature_id}.")
            except ValueError:
                print(f"Warning: Invalid charge value '{row[chg_col]}' for feature {feature_id}. Using polarity-derived charge.")
                charge_val = 1 if is_positive_mode else -1
        else: # No charge column, or value is None
            charge_val = 1 if is_positive_mode else -1
        
        # Ensure charge is non-zero if default adducts [M+H]+ or [M-H]- are used.
        if charge_val == 0:
            if adduct_str == "[M+H]+":
                # print(f"Warning: Charge was 0 for feature {feature_id} with adduct {adduct_str}. Setting charge to 1.")
                charge_val = 1
            elif adduct_str == "[M-H]-":
                # print(f"Warning: Charge was 0 for feature {feature_id} with adduct {adduct_str}. Setting charge to -1.")
                charge_val = -1
            # If charge is 0 and adduct is something else (e.g. "[M]"), msbuddy should handle it.

        rt = row.get(rt_col) if rt_col else None

        # Ensure required numeric types are correct before creating MetaFeature
        mz_val = float(row[pmz_col])
        rt_val = float(rt) if rt is not None else None

        # Handle MS1 isotopic pattern
        ms1_spectrum = None
        if ms1_iso_mz_c and ms1_iso_int_c and \
           ms1_iso_mz_c in row and ms1_iso_int_c in row:
            
            iso_mz_data = row[ms1_iso_mz_c]
            iso_int_data = row[ms1_iso_int_c]

            if isinstance(iso_mz_data, (list, np.ndarray, pl.Series)) and \
               isinstance(iso_int_data, (list, np.ndarray, pl.Series)):
                
                ms1_iso_mz_arr = np.array(iso_mz_data, dtype=np.float64)
                ms1_iso_int_arr = np.array(iso_int_data, dtype=np.float64)

                if ms1_iso_mz_arr.ndim == 1 and ms1_iso_int_arr.ndim == 1 and \
                   len(ms1_iso_mz_arr) == len(ms1_iso_int_arr) and len(ms1_iso_mz_arr) > 0:
                    
                    if not (np.any(np.isnan(ms1_iso_mz_arr)) or np.any(np.isinf(ms1_iso_mz_arr)) or \
                            np.any(np.isnan(ms1_iso_int_arr)) or np.any(np.isinf(ms1_iso_int_arr)) or \
                            np.any(ms1_iso_int_arr < 0)):
                        ms1_spectrum = Spectrum(mz_array=ms1_iso_mz_arr, int_array=ms1_iso_int_arr)
                    else:
                        print(f"Warning: Skipping MS1 isotopic pattern for {feature_id} due to invalid values (NaN, Inf, or negative intensity).")
                elif len(ms1_iso_mz_arr) > 0: # Only warn if there was data but it was malformed or empty after processing
                    print(f"Warning: Skipping MS1 isotopic pattern for {feature_id} due to mismatched, non-1D, or empty m/z and intensity arrays (lengths: {len(ms1_iso_mz_arr)}, {len(ms1_iso_int_arr)}).")
            elif iso_mz_data is not None or iso_int_data is not None: # Warn if columns exist but data is not list-like
                 print(f"Warning: Skipping MS1 isotopic pattern for {feature_id} due to non list-like data for m/z (type: {type(iso_mz_data)}) or intensity (type: {type(iso_int_data)}).")

        metafeature = MetaFeature(
            identifier=row[id_col],
            mz=mz_val,
            charge=charge_val, # Use the determined charge_val
            adduct=adduct_str, # Use the determined adduct_str
            rt=rt_val,
            ms1=ms1_spectrum, # Use the created Spectrum object for MS1 isotopic pattern
            ms2=ms2_spec,
        )
        return metafeature

    except Exception as e:
        print(f"Warning: skipping {feature_id} due to error: {e}")
        return None

def convert_df_to_metafeature_list(
    query_df: pl.DataFrame,
    interface_config: msbuddyInterfaceConfig
) -> List[MetaFeature]:
    cols = [
        interface_config.identifier_col,
        interface_config.precursor_mz_col,
        interface_config.ms2_mz_col,
        interface_config.ms2_int_col,
    ]
    # Add optional columns if they exist in the DataFrame and are configured
    for opt_col_attr in ['rt_col', 'adduct_col', 'charge_col', 
                         'ms1_isotope_mz_col', 'ms1_isotope_int_col']:
        col_name = getattr(interface_config, opt_col_attr)
        if col_name and col_name in query_df.columns:
            cols.append(col_name)
        elif col_name and col_name not in query_df.columns:
             # Only print warning if column is configured but missing in DataFrame
            print(f"Warning: Optional column '{col_name}' not found in DataFrame, it will not be used.")


    # Ensure no duplicate columns
    cols = sorted(list(set(cols)))
    
    # Check if all essential columns are present before attempting to select
    essential_cols_in_df = [
        interface_config.identifier_col,
        interface_config.precursor_mz_col,
        interface_config.ms2_mz_col,
        interface_config.ms2_int_col,
    ]
    missing_essential = [c for c in essential_cols_in_df if c not in query_df.columns]
    if missing_essential:
        raise ValueError(f"DataFrame missing essential columns for MetaFeature conversion: {missing_essential}")

    row_dicts = query_df.select(cols).to_dicts()
    mf_list: List[MetaFeature] = []
    for r in row_dicts:
        mf = create_metafeature_from_row(r, interface_config)
        if mf:
            mf_list.append(mf)
    return mf_list

def annotate_formulas_msbuddy(
    query_df: pl.DataFrame,
    interface_config: msbuddyInterfaceConfig,
    msbuddy_config: MsbuddyConfig,
) -> pl.DataFrame:
    """
    Annotates molecular formulas using Msbuddy.
    Creates MetaFeature objects sequentially.
    Msbuddy engine handles annotation parallelism based on its config.
    """
    # build and check required cols
    req = [
        interface_config.identifier_col,
        interface_config.precursor_mz_col,
        interface_config.ms2_mz_col,
        interface_config.ms2_int_col,
    ]
    missing = [c for c in req if c not in query_df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    print(f"Preparing {len(query_df)} features…")
    start = time.time()
    mf_list = convert_df_to_metafeature_list(query_df, interface_config)
    print(f"Prepared {len(mf_list)} valid features in {time.time()-start:.2f}s")
    if not mf_list:
        return pl.DataFrame()

    msb_engine = Msbuddy(msbuddy_config)
    msb_engine.add_data(mf_list)
    msb_engine.annotate_formula()

    print("Retrieving detailed annotation results including subformulas...")
    all_results_data = []
    if not msb_engine.data:
        print("No data found in Msbuddy engine after annotation.")
        return pl.DataFrame()
    start_post_process = time.time()
    for meta_feature in msb_engine.data:
        if not meta_feature.candidate_formula_list:
            print(f"Feature {meta_feature.identifier} has no candidate formulas.")
            continue

        for i, candidate in enumerate(meta_feature.candidate_formula_list):
            result_entry = {
                "identifier": meta_feature.identifier,
                "rank": i + 1,
                "neutral_formula": candidate.formula.__str__()
                if candidate.formula
                else None,
                "neutral_formula_array": candidate.formula.array
                if candidate.formula
                else None,
                "charged_formula": candidate.charged_formula.__str__()
                if candidate.charged_formula
                else None,
                "charged_formula_array": candidate.charged_formula.array
                if candidate.charged_formula
                else None,
                "estimated_fdr": candidate.estimated_fdr,
                "ms1_isotope_similarity": candidate.ms1_isotope_similarity,
                "estimated_prob": candidate.estimated_prob,
                "normed_estimated_prob": candidate.normed_estimated_prob,
                "subformula_mz": None,  # Initialize new keys
                "subformula_int": None,  # Initialize new keys
                "subformula_str": None,  # Initialize new keys
                "subformula_arr": None,  # Initialize new keys
            }

            # Extract subformula explanations if available
            subformula_mz_list = []
            subformula_int_list = []
            subformula_str_list = []
            subformula_arr_list = []

            # Check if raw explanation, raw spectrum, and its arrays exist
            if (
                candidate.ms2_raw_explanation
                and meta_feature.ms2_raw
                and meta_feature.ms2_raw.mz_array is not None
                and meta_feature.ms2_raw.int_array is not None
            ):
                try:
                    explained_indices = candidate.ms2_raw_explanation.idx_array
                    raw_mz_array = meta_feature.ms2_raw.mz_array
                    raw_int_array = (
                        meta_feature.ms2_raw.int_array
                    )  # Get intensity array
                    subformulas = candidate.ms2_raw_explanation.explanation_list

                    if (
                        explained_indices is not None
                        and raw_mz_array is not None
                        and raw_int_array is not None
                        and subformulas is not None
                    ):
                        # Ensure raw arrays have the same length for safety, although they should
                        if len(raw_mz_array) == len(raw_int_array):
                            for idx, sub_formula in zip(explained_indices, subformulas):
                                if sub_formula is not None:
                                    # Ensure index is within bounds of the RAW arrays
                                    if 0 <= idx < len(raw_mz_array):
                                        # Get fragment mz and intensity from the RAW spectrum using the index
                                        fragment_mz = raw_mz_array[idx]
                                        fragment_int = raw_int_array[idx]
                                        sub_formula_str = sub_formula.__str__()
                                        sub_formula_arr = (
                                            sub_formula.array
                                        )  # Get the array representation

                                        subformula_mz_list.append(fragment_mz)
                                        subformula_int_list.append(fragment_int)
                                        subformula_str_list.append(sub_formula_str)
                                        subformula_arr_list.append(
                                            sub_formula_arr.tolist()
                                            if sub_formula_arr is not None
                                            else None
                                        )  # Convert numpy array to list for Polars compatibility
                                    else:
                                        # This warning should now be less likely unless the index is invalid even for the raw spectrum
                                        print(
                                            f"Warning: Subformula explanation index {idx} out of bounds for RAW spectrum (len={len(raw_mz_array)}) for feature {meta_feature.identifier}, candidate rank {i + 1}."
                                        )
                        else:
                            print(
                                f"Warning: Raw m/z and intensity arrays have different lengths for feature {meta_feature.identifier}. Skipping subformula extraction."
                            )

                except AttributeError as ae:
                    # Catch cases where expected attributes might be missing
                    print(
                        f"Warning: Missing attribute while processing subformulas for feature {meta_feature.identifier}, candidate rank {i + 1}: {ae}"
                    )
                except Exception as e:
                    print(
                        f"Warning: Error processing subformulas for feature {meta_feature.identifier}, candidate rank {i + 1}: {e}"
                    )

            # Assign lists to result entry if they are not empty
            if subformula_mz_list:
                result_entry["subformula_mz"] = subformula_mz_list
                result_entry["subformula_int"] = subformula_int_list
                result_entry["subformula_str"] = subformula_str_list
                result_entry["subformula_arr"] = subformula_arr_list

            all_results_data.append(result_entry)

    if not all_results_data:
        print("No annotation results generated by Msbuddy.")
        return pl.DataFrame()
    print(
        f"Post-processing of annotation results finished in {time.time() - start_post_process:.2f} seconds."
    )
    print(f"Retrieved {len(all_results_data)} candidate formula results from Msbuddy.")
    start_convert = time.time()
    # Convert results to Polars DataFrame
    # Use infer_schema_length=None to handle potentially large/complex list structures
    results_df = pl.from_dicts(
        all_results_data, 
        schema={
            "identifier": pl.Int64,
            "rank": pl.Int32,
            "neutral_formula": pl.String,
            "neutral_formula_array": pl.List(pl.Int32),
            "charged_formula": pl.String,
            "charged_formula_array": pl.List(pl.Int32),
            "estimated_fdr": pl.Float64,
            "ms1_isotope_similarity": pl.Float64,
            "estimated_prob": pl.Float64,
            "normed_estimated_prob": pl.Float64,
            "subformula_mz": pl.List(pl.Float64),
            "subformula_int": pl.List(pl.Float64),
            "subformula_str": pl.List(pl.String),
            "subformula_arr": pl.List(pl.List(pl.Int32))
        }
    )
    print(
        f"Converted annotation results to Polars DataFrame with {len(results_df)} rows."
    )
    print(f"this took {time.time() - start_convert:.2f} seconds.")
    # Rename identifier column back to original name for joining
    results_df = results_df.rename({"identifier": interface_config.identifier_col})

    return results_df


if __name__ == "__main__":
    from ms_utils.interfaces.msdial import get_chromatogram
    pl.enable_string_cache()
    pl.set_random_seed(42)
    start_time = time.time()


    msbuddy_config = MsbuddyConfig(
        ms_instr="orbitrap", ppm=True, 
        ms1_tol=3, 
        ms2_tol=5,
        halogen=True, parallel=True, n_cpu=16,
        timeout_secs=300, rel_int_denoise_cutoff=0.001,
        top_n_per_50_da=100,
        isotope_bin_mztol=0.008
    )
    interface_config = msbuddyInterfaceConfig(
        data_path="placeholder",
        identifier_col="Peak ID",
        precursor_mz_col="Precursor_mz_MSDIAL",
        adduct_col="Precursor_type_MSDIAL", # Msbuddy will parse this if present
        charge_col="Charge", # Msbuddy will use this if present and valid
        ms2_mz_col="msms_m/z",
        ms2_int_col="msms_intensity",
        ms1_isotope_mz_col='ms1_isotopes_m/z',
        ms1_isotope_int_col='ms1_isotopes_intensity',
        polarity_col=None,  # Set to your polarity column name if you have one, e.g., "Polarity"
        default_polarity="negative"  # Or "negative" if that's your data's common mode
    )

    # Example: Ensure your DataFrame (query_df) might have 'Charge' and 'Polarity' columns
    # if you specify them in interface_cfg. Otherwise, the defaults will apply.
    # For the sample data loading:
    # df = get_chromatogram(...).filter(...)
    # If 'Charge' column is missing from df, charge will be 1 or -1 based on polarity.
    # If 'Precursor_type_MSDIAL' is missing, adduct will be [M+H]+ or [M-H]- based on polarity.

    # Example of how you might add these columns if they don't exist for testing:
    # if "Charge" not in query_df.columns and interface_cfg.charge_col == "Charge":
    #     print("Adding dummy 'Charge' column for testing based on default_polarity.")
    #     query_df = query_df.with_columns(
    #         pl.when(pl.lit(interface_cfg.default_polarity == "positive")).then(pl.lit(1)).otherwise(pl.lit(-1)).alias("Charge")
    #     )
    # if "Polarity" not in query_df.columns and interface_cfg.polarity_col == "Polarity":
    #     print("Adding dummy 'Polarity' column for testing.")
    #     query_df = query_df.with_columns(pl.lit(interface_cfg.default_polarity).alias("Polarity"))
    query_df= get_chromatogram(
    path = '/home/analytit_admin/dev/PFAS/data/raw_data/250514_015.txt'
    ).filter(
    pl.col('msms_m/z').is_not_null() &
    pl.col('msms_m/z').list.len() > 0,
    pl.col('RT (min)').gt(2)
    )
    try:
        results = annotate_formulas_msbuddy(query_df, interface_config, msbuddy_config).filter(
            pl.col('subformula_mz').list.len()> 0
        )
    except Exception as e:
        print(f"Error: {e}")
        results = pl.DataFrame()

    if not results.is_empty():
        print(results.select(
            pl.col(interface_config.identifier_col),
            pl.col("rank"),
            # pl.col("neutral_formula"),
            pl.col("charged_formula"),
            pl.col("estimated_prob"),
            pl.col("normed_estimated_prob"),
            pl.col("estimated_fdr"),
            pl.col("ms1_isotope_similarity"),
            # pl.col("subformula_mz"),
            # pl.col("subformula_int"),
            pl.col("subformula_str"),
            # pl.col("subformula_arr")
        ).head(), results.schema)

    print(f"Total time: {time.time()-start_time:.2f}s")
    pl.disable_string_cache()
