import polars as pl
import numpy as np
from msbuddy import Msbuddy, MsbuddyConfig
from msbuddy.base import MetaFeature, Spectrum
from typing import List, Dict, Any, Optional
import time
import os
from dataclasses import dataclass

@dataclass
class msbuddyInterfaceConfig:
    data_path: str
    identifier_col: str = "NIST_ID"
    precursor_mz_col: str = "PrecursorMZ"
    ms2_mz_col: str = "raw_spectrum_mz"
    ms2_int_col: str = "raw_spectrum_intensity"
    rt_col: Optional[str] = None
    adduct_col: Optional[str] = None
    charge_col: Optional[str] = None

def create_metafeature_from_row(
    row: Dict[str, Any],
    interface_config: msbuddyInterfaceConfig,
    default_adduct: str = "[M+H]+",
    default_charge: int = 1,
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
            return None

        ms2_spec = Spectrum(mz_array=ms2_mz, int_array=ms2_int)

        adduct = row.get(add_col, default_adduct) if add_col else default_adduct
        charge = row.get(chg_col, default_charge) if chg_col else default_charge
        rt = row.get(rt_col) if rt_col else None

        # Ensure required numeric types are correct before creating MetaFeature
        mz_val = float(row[pmz_col])
        charge_val = int(charge)
        rt_val = float(rt) if rt is not None else None

        metafeature = MetaFeature(
            identifier=row[id_col],
            mz=mz_val,
            charge=charge_val,
            adduct=adduct,
            rt=rt_val,
            ms1=None,  # Assuming MS1 is not used here
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
    for opt in (interface_config.rt_col, interface_config.adduct_col, interface_config.charge_col):
        if opt and opt in query_df.columns:
            cols.append(opt)

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
    pl.enable_string_cache()
    pl.set_random_seed(42)
    start_time = time.time()

    msb_cfg = MsbuddyConfig(
        ms_instr="orbitrap", ppm=True, ms1_tol=5, ms2_tol=10,
        halogen=True, parallel=True, n_cpu=16,
        timeout_secs=300, rel_int_denoise_cutoff=0.001,
        top_n_per_50_da=100,
    )

    # build one interface_cfg instead of 7 separate vars
    interface_cfg = msbuddyInterfaceConfig(
        data_path="/home/analytit_admin/Data/NIST_hr_msms/NIST_hr_msms.parquet",
        identifier_col="NIST_ID",
        precursor_mz_col="PrecursorMZ",
        ms2_mz_col="raw_spectrum_mz",
        ms2_int_col="raw_spectrum_intensity",
    )

    if not os.path.exists(interface_cfg.data_path):
        print(f"Error: Data not found at {interface_cfg.data_path}")
        exit()

    df = pl.read_parquet(interface_cfg.data_path, columns=[
        interface_cfg.identifier_col,
        interface_cfg.precursor_mz_col,
        interface_cfg.ms2_mz_col,
        interface_cfg.ms2_int_col,
    ]).filter(pl.col(interface_cfg.precursor_mz_col) <= 900)

    query_df = df.sample(n=10, seed=42) if len(df) >= 10 else df
    try:
        results = annotate_formulas_msbuddy(query_df, interface_cfg, msb_cfg)
    except Exception as e:
        print(f"Error: {e}")
        results = pl.DataFrame()

    if not results.is_empty():
        print(results.head(), results.schema)

    print(f"Total time: {time.time()-start_time:.2f}s")
    pl.disable_string_cache()
