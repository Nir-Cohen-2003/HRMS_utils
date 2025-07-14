import polars as pl
from pathlib import Path
import re

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
    # name changes for clarity
        pl.col("INCHIAUX").alias("inchikey"),
    # conversion fo formula to array
    )

    # Optionally, drop the raw entry and mz_int_pairs columns
    df = df.drop(["mz_int_pairs"])



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
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Number of entries: {df.height}")
    print(f'time per entry: {(end_time - start_time) / df.height:.8f} seconds')
