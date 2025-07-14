import polars as pl
from pathlib import Path
import re
from typing import Iterable
from ..formula_annotation.utils import formula_to_array

def read_mgf_to_dataframe(
        mgf_path: str | Path,
        includes_MSn: bool = False
        ) -> pl.DataFrame:
    
    with open(mgf_path, 'r') as f:
        mgf_text = f.read()
        entries = re.findall(r'BEGIN IONS(.*?)END IONS', mgf_text, re.DOTALL)
    df = pl.DataFrame({'entry': entries})
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
    df = df.with_columns(exprs).drop(["entry"])

    df = df.with_columns(
        pl.col("mz_int_pairs")
        .list.eval(pl.element().str.split(by=" ").list.get(0).cast(pl.Float64))
        .alias("spectrum_mz"),
        pl.col("mz_int_pairs")
        .list.eval(pl.element().str.split(by=" ").list.get(1).cast(pl.Float64))
        .alias("spectrum_intensity"),
    ).drop(
        ["mz_int_pairs"]
    )

    df = df.rename(
        {
            "INCHIAUX": "inchikey"
        }
    ).cast(
        {
            "EXACTMASS": pl.Float64,
            "RTINSECONDS": pl.Float64,
            "PEPMASS": pl.Float64,
            "CHARGE": pl.Int64,
            "FEATURE_MS1_HEIGHT": pl.Float64,
            "MSLEVEL": pl.Int64,
            "ISOLATION_WINDOW": pl.Float64,
            "Num peaks": pl.Int64
        }
    )

    df = formula_to_array(df, input_col_name='FORMULA', output_col_name='FORMULA_array')

    # COLLISION_ENERGY handling (MS2)
    df = df.with_columns(
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
        df = df.with_columns(
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
            pl.when(pl.col("MSn_collision_energies").is_not_null())
            .then(
                pl.col("MSn_collision_energies")
                .str.strip_prefix('[').str.strip_suffix(']')
                .str.split(",")
                .list.eval(# each element might be a list of itself, so we need to strip and split again
                    pl.element().str.strip_prefix('[').str.strip_suffix(']')
                    .str.split(",")
                    .list.eval(
                        pl.element().str.strip_chars(" ").cast(pl.Float64,strict=False)  # allow for empty strings, which will be cast to None
                    )   
                )
            ).alias("MSn_collision_energies")
        )

    return df

# Example usage:
if __name__ == "__main__":
    from time import perf_counter
    start_time = perf_counter()
    mgf_file = Path("/home/analytit_admin/Downloads/20241003_enamdisc_neg_msn.mgf")
    df = read_mgf_to_dataframe(mgf_file, includes_MSn=True)
    end_time = perf_counter()
    print(df)
    # what is the MSLEVEL values, thier distributoion etc
    print(df["MSLEVEL"].value_counts(sort=True))
    #print the msn stuff using select, in one line, filtering nulls
    print(df.select(
        pl.col("MSn_collision_energies"),
        pl.col("MSn_precursor_mzs"),
        pl.col("MSn_fragmentation_methods"),
        pl.col("MSn_isolation_windows")
    ).filter(
        pl.col("MSn_collision_energies").is_not_null() |
        pl.col("MSn_precursor_mzs").is_not_null() |
        pl.col("MSn_fragmentation_methods").is_not_null() |
        pl.col("MSn_isolation_windows").is_not_null()  
    )
    )
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Number of entries: {df.height}")
    print(f'time per entry: {(end_time - start_time) / df.height:.8f} seconds')
