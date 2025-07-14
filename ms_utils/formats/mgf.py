import polars as pl
from pathlib import Path
import re
from ..formula_annotation.utils import formula_to_array

def split_mgf_entries(mgf_text: str) -> list[str]:
    # Split on BEGIN IONS ... END IONS blocks
    entries = re.findall(r'BEGIN IONS(.*?)END IONS', mgf_text, re.DOTALL)
    return [entry.strip() for entry in entries if entry.strip()]

def mgf_entries_to_dataframe(entries: list[str]) -> pl.DataFrame:
    # Put entries in a DataFrame
    df = pl.DataFrame({'entry': entries})

    # Extract all key=value metadata using regex
    meta_keys = [
        "NAME", "DESCRIPTION", "EXACTMASS", "FORMULA", "INCHI", "INCHIAUX", "SMILES",
        "FEATURE_ID", "MSLEVEL", "RTINSECONDS", "ADDUCT", "PEPMASS", "CHARGE",
        "FEATURE_MS1_HEIGHT", "SPECTYPE", "COLLISION_ENERGY", "FRAGMENTATION_METHOD",
        "ISOLATION_WINDOW", "ACQUISITION", "INSTRUMENT_TYPE", "SOURCE_INSTRUMENT",
        "IMS_TYPE", "ION_SOURCE", "IONMODE", "PI", "DATACOLLECTOR", "DATASET_ID",
        "USI", "SCANS", "PRECURSOR_PURITY", "QUALITY_CHIMERIC",
        "QUALITY_EXPLAINED_INTENSITY", "QUALITY_EXPLAINED_SIGNALS", "Num peaks"
    ]
    exprs = []
    for key in meta_keys:
        exprs.append(
            pl.col("entry").str.extract(rf"(?m)^{key}=(.+)$", 1).alias(key)
        )

    # Extract all lines that look like "mz intensity" (two floats separated by whitespace)
    exprs.append(
        pl.col("entry")
        .str.extract_all(r"(?m)^(\d+\.\d+)\s+(\d+\.\d+(?:[eE][+-]?\d+)?)$")
        .alias("mz_int_pairs")
    )
    # Apply the expressions to the DataFrame, all at once
    df = df.with_columns(exprs)
    # Drop the original entry column, as it's no longer needed and can be large
    df = df.drop(["entry"])

    df = df.with_columns(
    # Split mz_int_pairs into two lists: mz and intensity
        pl.col("mz_int_pairs")
        .list.eval(pl.element().str.split(by=" ").list.get(0).cast(pl.Float64))
        .alias("spectrum_mz"),
        pl.col("mz_int_pairs")
        .list.eval(pl.element().str.split(by=" ").list.get(1).cast(pl.Float64))
        .alias("spectrum_intensity"),
    ).drop(
        ["mz_int_pairs"]
    )

    df=df.rename(
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

    # convert the formula to an array of elements
    df = formula_to_array(df, input_col_name='FORMULA', output_col_name='FORMULA_array')

    # the collision energy might be a list with multiple values, which would be in [] brackets, or a single float value.
    # make a column to indicate if it is a list or not named multiple_collision_energies and then create a list of collision energies, which might be a single value or multiple values
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

    return df

def read_mgf_to_dataframe(mgf_path: str | Path) -> pl.DataFrame:
    with open(mgf_path, 'r') as f:
        mgf_text = f.read()
    entries = split_mgf_entries(mgf_text)
    return mgf_entries_to_dataframe(entries)

# Example usage:
if __name__ == "__main__":
    from time import perf_counter
    start_time = perf_counter()
    mgf_file = Path("/home/analytit_admin/Downloads/20241003_enamdisc_neg_ms2.mgf")
    df = read_mgf_to_dataframe(mgf_file)
    end_time = perf_counter()
    print(df.head())
    # print only the casted columns
    print(df.select([
        pl.col("EXACTMASS"),
        pl.col("RTINSECONDS"),
        pl.col("PEPMASS"),
        pl.col("CHARGE"),
        pl.col("FEATURE_MS1_HEIGHT"),
        pl.col("MSLEVEL"),
        pl.col("ISOLATION_WINDOW"),
        pl.col("Num peaks"),
        pl.col("collision_energy_mean"),
        pl.col("multiple_collision_energies")
    ]))
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Number of entries: {df.height}")
    print(f'time per entry: {(end_time - start_time) / df.height:.8f} seconds')
