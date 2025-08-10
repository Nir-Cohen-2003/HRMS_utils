import polars as pl
from pathlib import Path
import re
from typing import Iterable, List
from ..formula_annotation.utils import formula_to_array


def read_all_ms2_files(dir_path:Path|str)->pl.DataFrame:
    '''reads all mgf files and extracts MS2 sepctra, specifiaclly those that are not merged but resulted from a signle authentic scan, and thus have a single USI'''
    dir_path = Path(dir_path)
    dfs: List[pl.DataFrame] = []
    for file_path in dir_path.glob(pattern="*.mgf"):
        dfs.append(read_mgf_to_dataframe(file_path,includes_MSn=False))
    df = pl.concat(dfs)
    df = df.filter(
        pl.col("is_single_spectra"),
        pl.col("MSLEVEL").eq(2)
        )
    # df = df.with_columns(
    #     USI=pl.col("USI").list.get(0)
    # )
    # df = df.unique(["USI"])
    return df


def read_mgf_to_dataframe(
        mgf_path: str | Path,
        includes_MSn: bool = False
        ) -> pl.DataFrame:
    
    with open(mgf_path, 'r') as f:
        mgf_text = f.read()
        entries = re.findall(r'BEGIN IONS(.*?)END IONS', mgf_text, re.DOTALL)
    lf = pl.DataFrame({'entry': entries})
    del entries

    meta_keys = [
        "NAME", "DESCRIPTION", "EXACTMASS", "FORMULA", "INCHI", "INCHIAUX", "SMILES",
        "FEATURE_ID", "MSLEVEL", "RTINSECONDS", "ADDUCT", "PEPMASS", "CHARGE",
        "FEATURE_MS1_HEIGHT", "SPECTYPE", "COLLISION_ENERGY", "FRAGMENTATION_METHOD",
        "ISOLATION_WINDOW", "ACQUISITION", "INSTRUMENT_TYPE", "SOURCE_INSTRUMENT",
        "IMS_TYPE", "ION_SOURCE", "IONMODE", "PI", "DATACOLLECTOR", "DATASET_ID",
        "USI", "SCANS", "PRECURSOR_PURITY", "QUALITY_CHIMERIC",
        "QUALITY_EXPLAINED_INTENSITY", "QUALITY_EXPLAINED_SIGNALS", "Num peaks"
    ]
    msn_keys = [
        "MSn_collision_energies", "MSn_precursor_mzs", "MSn_fragmentation_methods", "MSn_isolation_windows"
    ]
    exprs = []
    for key in meta_keys:
        exprs.append(
            pl.col("entry").str.extract(rf"(?m)^{key}=(.+)$", 1).alias(key)
        )
    if includes_MSn:
        for key in msn_keys:
            exprs.append(
                pl.col("entry").str.extract(rf"(?m)^{key}=(.+)$", 1).alias(key)
            )

    
    exprs.append(
        pl.col("entry")
        .str.extract_all(r"(?m)^(\d+\.\d+)\s+(\d+\.\d+(?:[eE][+-]?\d+)?)$")
        .alias("mz_int_pairs")
    )
    lf = lf.with_columns(exprs).drop(["entry"]).lazy()
    #renaming and casting
    lf = lf.rename(
        {
            "INCHIAUX": "inchikey"
        }
    ).with_columns(
        pl.col("SPECTYPE").fill_null(value="SINGLE_BEST_SCAN"),
        pl.col("IONMODE").str.to_lowercase()
    ).cast(
        {
            "EXACTMASS": pl.Float64,
            "RTINSECONDS": pl.Float64,
            "PEPMASS": pl.Float64,
            "CHARGE": pl.Int64,
            "FEATURE_MS1_HEIGHT": pl.Float64,
            "MSLEVEL": pl.Int64,
            "ISOLATION_WINDOW": pl.Float64,
            "Num peaks": pl.Int64,
            "ION_SOURCE": pl.Enum(['ESI']),
            "SPECTYPE": pl.Enum(['SINGLE_SCAN',"SINGLE_BEST_SCAN","SAME_ENERGY","ALL_ENERGIES","ALL_MSN_TO_PSEUDO_MS2"]),
            "IONMODE":pl.Enum(["positive","negative"]),
            "PRECURSOR_PURITY": pl.Float64,
            "QUALITY_EXPLAINED_INTENSITY":pl.Float64,
            "QUALITY_EXPLAINED_SIGNALS":pl.Float64
        }
    )
    # handling of spectrum
    lf = lf.with_columns(
        pl.col("mz_int_pairs")
        .list.eval(pl.element().str.split(by=" ").list.get(0).cast(pl.Float64))
        .alias("spectrum_mz"),
        pl.col("mz_int_pairs")
        .list.eval(pl.element().str.split(by=" ").list.get(1).cast(pl.Float64))
        .alias("spectrum_intensity"),
    ).drop(
        ["mz_int_pairs"]
    )

    # handling of USI, which might be a list of USI or just one
    lf = lf.with_columns(
        pl.col("USI").str.strip_chars("[]").str.split(by=",").alias("USI"),
    ).with_columns(
        pl.col("USI").list.len().alias("num_merged_spectra"),
    ).with_columns(
        pl.col("num_merged_spectra").eq(1).alias("is_single_spectra")
    )

    # what it says. 
    lf = formula_to_array(lf, input_col_name='FORMULA', output_col_name='FORMULA_array')

    # COLLISION_ENERGY handling (MS2)
    lf = lf.with_columns(
        pl.col("COLLISION_ENERGY").str.strip_chars("[]").str.split(by=",").list.eval(
            pl.element().str.strip_chars(" ")
        ).cast(
            pl.List(pl.Float64)
        ).alias("collision_energy_list")
    ).drop(
        "COLLISION_ENERGY"
    ).with_columns(
        pl.when(pl.col("collision_energy_list").list.len() > 1)
        .then(pl.lit(True)).otherwise(pl.lit(False)).alias("multiple_collision_energies"),
        pl.col("collision_energy_list").list.mean().alias("collision_energy_mean")
    )
 
    # --- MSn fields ---
    if includes_MSn:
        lf = lf.with_columns(
            pl.when(pl.col("MSn_precursor_mzs").is_not_null())
            .then(
                pl.col("MSn_precursor_mzs")
                .str.strip_chars("[]")
                .str.split(",")
                .list.eval(pl.element().str.strip_chars(" ").cast(pl.Float64))
            )
            .alias("MSn_precursor_mzs"),
        
            # fragmentation methods
            pl.when(pl.col("MSn_fragmentation_methods").is_not_null())
            .then(
                pl.col("MSn_fragmentation_methods")
                .str.strip_chars("[]")
                .str.split(",")
                .list.eval(pl.element().str.strip_chars(" "))
            )
            .alias("MSn_fragmentation_methods"),
            # isolation windows
            pl.when(pl.col("MSn_isolation_windows").is_not_null())
            .then(
                pl.col("MSn_isolation_windows")
                .str.strip_chars("[]")
                .str.split(",")
                .list.eval(pl.element().str.strip_chars(" ").cast(pl.Float64))
            )
            .alias("MSn_isolation_windows"),

            # collision energies- can be nested, since each fragmentation step can be done with several energies at once, or one
            # examples:
            # [[30.0, 40.0, 20.0], [30.0, 40.0, 40.0], [30.0, 40.0, 60.0]]
            # [30.0, 40.0, 20.0]
            # [[30.0, 40.0, 20.0], 30.0, 40.0, 60.0]
            # TODO: fix this, it splits everything into a list of single valued lists
            # note- I'm not sure my analysis of this structure is correct, i need to understand it better. hold of for now
            pl.when(pl.col("MSn_collision_energies").is_not_null())
            .then(
                pl.col("MSn_collision_energies")
                .str.strip_prefix('[').str.strip_suffix(']')
                .str.replace_all("],", "]|")
                .str.split("|")
                .list.eval(# each element might be a list of itself, so we need to strip and split again
                    pl.element().str.strip_prefix('[').str.strip_suffix(']')
                    .str.split(",")
                    .list.eval(
                        pl.element().str.strip_chars(" ").cast(pl.Float64,strict=False)  # allow for empty strings, which will be cast to None
                    )   
                )
            ).alias("MSn_collision_energies")
        )

    return lf.collect()

# Example usage:
if __name__ == "__main__":
    from time import perf_counter
    start_time = perf_counter()
    mgf_file = Path("/home/analytit_admin/Data/MSnLib/20241003_enammol_pos_msn.mgf")
    df = read_mgf_to_dataframe(mgf_path=mgf_file)
    # df = read_all_ms2_files("/home/analytit_admin/Data/MSnLib/")
    end_time = perf_counter()
    print(df)
    print(df.schema)
    # what is the MSLEVEL values, thier distributoion etc
    print(df["MSLEVEL"].value_counts(sort=True))
    print(df["SPECTYPE"].value_counts(sort=True))
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Number of entries: {df.height}")
    print(f'time per entry: {(end_time - start_time) / df.height:.8f} seconds')
