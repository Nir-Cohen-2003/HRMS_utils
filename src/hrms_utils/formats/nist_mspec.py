import re
from pathlib import Path
import polars as pl

def parse_mspec(path: Path | str) -> pl.LazyFrame:
    """Thin parser for MSPEC/MSP files. Extracts raw fields from text."""
    path = Path(path)
    with open(path, "r") as file:
        file_contents = file.read()
    
    entries = _split_entries(file_contents)
    data = pl.DataFrame(entries, schema={"raw": pl.String}).lazy()
    
    mz_intensity_pattern = r"(\d+\.\d+)\s(\d+(\.\d+)?)"
    
    return (
        data.with_columns(
            pl.col("raw").str.extract(pattern=r"(?i)Name: (.+)", group_index=1).alias("name"),
            pl.col("raw").str.extract(pattern=r"(?i)NIST#: (\d+)", group_index=1).alias("nist_id"),
            pl.col("raw").str.extract(pattern=r"(?i)DB#: (\d+)", group_index=1).alias("db_id"),
            pl.col("raw").str.extract(pattern=r"(?i)Instrument_?type: (.+)", group_index=1).alias("instrument_type"),
            pl.col("raw").str.extract(pattern=r"(?i)Instrument: (.+)", group_index=1).alias("instrument"),
            pl.col("raw").str.extract(pattern=r"(?i)(?:Spectrum_type|MSLEVEL): (?:MS)?(\d+)", group_index=1).cast(pl.Int64, strict=False).alias("mslevel"),
            pl.col("raw").str.extract(pattern=r"(?i)Collision_gas: (.+)", group_index=1).alias("collision_gas"),
            pl.col("raw").str.extract(pattern=r"(?i)Collision_?energy: (.+)", group_index=1).alias("collision_energy_raw"),
            pl.col("raw").str.extract(pattern=r"(?i)Ionization: (.+)", group_index=1).alias("ionization"),
            pl.col("raw").str.extract(pattern=r"(?i)Ion_?mode: (p|n)", group_index=1).alias("ion_mode"),
            pl.col("raw").str.extract(pattern=r"(?i)Precursor_?type: (.+)", group_index=1).alias("precursor_type"),
            pl.col("raw").str.extract(pattern=r"(?i)PrecursorMZ: (\d+\.?\d*)", group_index=1).alias("precursor_mz"),
            pl.col("raw").str.extract(pattern=r"(?i)MW: (\d+)", group_index=1).alias("mw"),
            pl.col("raw").str.extract(pattern=r"(?i)Formula: (.+)", group_index=1).alias("molecular_formula"),
            pl.col("raw").str.extract(pattern=r"(?i)Num Peaks: (\d+)", group_index=1).alias("num_peaks"),
            pl.col("raw").str.extract(pattern=r"(?i)\nCAS#: ([0-9,-]+)", group_index=1).alias("cas"),
            pl.col("raw").str.extract(pattern=r"(?i)\nInChIKey: (.+)", group_index=1).alias("inchikey"),
            pl.col("raw").str.extract(pattern=r"(?i)\nExactMass: (\d+\.\d+)", group_index=1).alias("exact_mass"),
            pl.col("raw").str.extract(pattern=r"(?i)SMILES: (.+)").alias("smiles"),
            pl.col("raw").str.extract(pattern=r"(?i)InChI: (.+)").alias("inchi"),
            pl.col("raw").str.extract_all(pattern=mz_intensity_pattern).alias("mz_intensity"),
        )
        .drop("raw")
        .with_columns(
            pl.col("nist_id").str.to_integer(),
            pl.col("db_id").str.to_integer(),
            pl.col("mw").str.to_integer(),
            pl.col("ion_mode").str.to_uppercase(),
            pl.col("num_peaks").str.to_integer(),
            pl.col("precursor_mz").cast(pl.Float64),
            pl.col("exact_mass").cast(pl.Float64, strict=False),
            pl.col("mz_intensity").list.eval(pl.element().str.split(by=" ").list.get(index=0).cast(pl.Float64)).alias("raw_spectrum_mz"),
            pl.col("mz_intensity").list.eval(pl.element().str.split(by=" ").list.get(index=1).cast(pl.Float64)).alias("raw_spectrum_intensity"),
        )
        .drop("mz_intensity")
    )

def _split_entries(file_contents: str) -> list[str]:
    entries = re.split(r"\n\s*\n", file_contents)
    if entries and entries[-1] == "":
        entries.pop()
    return entries

# Backward compatibility
from .spectral_library import process_single_file as read_MSPEC_file
from .spectral_library import process_spectral_library as create_nist_dataframe
