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
            pl.col("raw").str.extract(pattern=r"(?mi)^\s*COMPOUND_?NAME:\s*(.+)", group_index=1).alias("_compound_name"),
            pl.col("raw").str.extract(pattern=r"(?mi)^\s*Name:\s*(.+)", group_index=1).alias("_name"),
            pl.col("raw").str.extract(pattern=r"(?mi)^\s*TITLE:\s*(.+)", group_index=1).alias("_title"),
            pl.col("raw").str.extract(pattern=r"(?i)NIST#: (\d+)", group_index=1).alias("nist_id"),
            pl.col("raw").str.extract(pattern=r"(?i)DB#: (\d+)", group_index=1).alias("db_id"),
            pl.col("raw").str.extract(pattern=r"(?i)Instrument_?type: (.+)", group_index=1).alias("instrument_type"),
            pl.col("raw").str.extract(pattern=r"(?i)Instrument: (.+)", group_index=1).alias("instrument"),
            pl.col("raw").str.extract(pattern=r"(?i)(?:Spectrum_type|MSLEVEL): (?:MS)?(\d+)", group_index=1).cast(pl.Int64, strict=False).alias("mslevel"),
            pl.col("raw").str.extract(pattern=r"(?i)Collision_gas: (.+)", group_index=1).alias("collision_gas"),
            # Any non-MSn key containing "energy" or "energies" is treated as the
            # collision-energy field. MSn_* keys are parsed separately when
            # includes_MSn is enabled. Polars (Rust regex) does not support
            # look-around, so the ``MSn_`` exclusion is expressed as a
            # hand-rolled state machine in the second alternative: keys that
            # start with ``M``, ``MS``, ``MSn`` then ``_``/``-``/``\n`` are
            # rejected.
            pl.col("raw").str.extract(
                pattern=r"(?mi)^\s*(?:energ(?:y|ies)[^:\n]*|(?:[^M\n]|M[^S\n]|MS[^n\n]|MSn[^_\-\n])[^:\n]*energ(?:y|ies))[^:\n]*:\s*(.+)$",
                group_index=1,
            ).alias("collision_energy_raw"),
            pl.col("raw").str.extract(pattern=r"(?mi)^\s*COLLISION_ENERGY:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)", group_index=1).alias("collision_energy"),
            pl.col("raw").str.extract(pattern=r"(?mi)^\s*COLLISION_ENERGY_1:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)", group_index=1).alias("collision_energy_1"),
            pl.col("raw").str.extract(pattern=r"(?mi)^\s*COLLISION_ENERGY_2:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)", group_index=1).alias("collision_energy_2"),
            pl.col("raw").str.extract(pattern=r"(?mi)^\s*COLLISION_ENERGY_3:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)", group_index=1).alias("collision_energy_3"),
            pl.col("raw").str.extract(pattern=r"(?mi)^\s*NORMALIZED_COLLISION_ENERGY:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)", group_index=1).alias("normalized_collision_energy"),
            pl.col("raw").str.extract(pattern=r"(?mi)^\s*NORMALIZED_COLLISION_ENERGY_1:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)", group_index=1).alias("normalized_collision_energy_1"),
            pl.col("raw").str.extract(pattern=r"(?mi)^\s*NORMALIZED_COLLISION_ENERGY_2:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)", group_index=1).alias("normalized_collision_energy_2"),
            pl.col("raw").str.extract(pattern=r"(?mi)^\s*NORMALIZED_COLLISION_ENERGY_3:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?|nan)", group_index=1).alias("normalized_collision_energy_3"),
            pl.col("raw").str.extract(pattern=r"(?i)Ionization: (.+)", group_index=1).alias("ionization"),
            pl.col("raw").str.extract(pattern=r"(?i)Ion_?mode: (p|n)", group_index=1).alias("_ion_mode_short"),
            pl.col("raw").str.extract(pattern=r"(?i)Ion_?mode: (.+)", group_index=1).alias("ion_mode_full"),
            pl.col("raw").str.extract(pattern=r"(?i)Precursor_?type: (.+)", group_index=1).alias("precursor_type"),
            pl.col("raw").str.extract(pattern=r"(?i)(?:PrecursorMZ|Selected Ion m/z): (\d+\.?\d*)", group_index=1).alias("_precursor_mz_raw"),
            pl.col("raw").str.extract(pattern=r"(?i)PEPMASS: (\d+\.?\d*)", group_index=1).alias("_pepmass"),
            pl.col("raw").str.extract(pattern=r"(?i)MW: (\d+)", group_index=1).alias("mw"),
            pl.col("raw").str.extract(pattern=r"(?i)Formula: (.+)", group_index=1).alias("molecular_formula"),
            pl.col("raw").str.extract(pattern=r"(?i)Num Peaks: (\d+)", group_index=1).alias("num_peaks"),
            pl.col("raw").str.extract(pattern=r"(?i)\nCAS#: ([0-9,-]+)", group_index=1).alias("cas"),
            pl.col("raw").str.extract(pattern=r"(?i)\nInChIKey: (.+)", group_index=1).alias("inchikey"),
            pl.col("raw").str.extract(pattern=r"(?i)\nExactMass: (\d+\.\d+)", group_index=1).alias("exact_mass"),
            pl.col("raw").str.extract(pattern=r"(?i)SMILES: (.+)").alias("smiles"),
            pl.col("raw").str.extract(pattern=r"(?i)InChI: (.+)").alias("inchi"),
            pl.col("raw").str.extract(pattern=r"(?i)CHARGE: (.+)", group_index=1).alias("charge"),
            pl.col("raw").str.extract(pattern=r"(?i)ADDUCT: (.+)", group_index=1).alias("adduct"),
            pl.col("raw").str.extract(pattern=r"(?i)RTINSECONDS: (\d+\.?\d*)", group_index=1).alias("rt_seconds"),
            pl.col("raw").str.extract(pattern=r"(?i)CCS: (\d+\.?\d*)", group_index=1).alias("ccs"),
            pl.col("raw").str.extract(pattern=r"(?i)PRECURSOR_?INTENSITY: (\d+\.?\d*)", group_index=1).alias("precursor_intensity"),
            pl.col("raw").str.extract(pattern=r"(?i)PRECURSOR_?IM: (\d+\.?\d*)", group_index=1).alias("precursor_im"),
            pl.col("raw").str.extract(pattern=r"(?i)SAMPLE_?INLET: (.+)", group_index=1).alias("sample_inlet"),
            pl.col("raw").str.extract(pattern=r"(?i)COLUMN_?TYPE: (.+)", group_index=1).alias("column_type"),
            pl.col("raw").str.extract(pattern=r"(?i)SPECTRAL_?ENTROPY: (\d+\.?\d*)", group_index=1).alias("spectral_entropy"),
            pl.col("raw").str.extract(pattern=r"(?i)ENAMINE_?CATALOG_?ID: (.+)", group_index=1).alias("enamine_catalog_id"),
            pl.col("raw").str.extract(pattern=r"(?i)PUBCHEM_?CID: (\d+)", group_index=1).alias("pubchem_cid"),
            pl.col("raw").str.extract(pattern=r"(?i)IUPAC_?NAME: (.+)", group_index=1).alias("iupac_name"),
            pl.col("raw").str.extract(pattern=r"(?i)NUM_?EXPLAINED_?PEAKS: (\d+)", group_index=1).alias("num_explained_peaks"),
            pl.col("raw").str.extract(pattern=r"(?i)EXPLAINED_?INTENSITY: (\d+\.?\d*)", group_index=1).alias("explained_intensity_raw"),
            pl.col("raw").str.extract(pattern=r"(?i)PEAKS_?PPM: (-?\d+\.?\d*)", group_index=1).alias("peaks_ppm"),
            pl.col("raw").str.extract(pattern=r"(?i)PEAKS_?ABS_?PPM: (\d+\.?\d*)", group_index=1).alias("peaks_abs_ppm"),
            pl.col("raw").str.extract(pattern=r"(?i)PRECURSOR_?PPM: (-?\d+\.?\d*)", group_index=1).alias("precursor_ppm"),
            pl.col("raw").str.extract(pattern=r"(?i)ADDUCT_?FORMULA: (.+)", group_index=1).alias("adduct_formula"),
            pl.col("raw").str.extract_all(pattern=mz_intensity_pattern).alias("mz_intensity"),
        )
        .with_columns(
            pl.col("nist_id").str.to_integer(),
            pl.col("db_id").str.to_integer(),
            pl.col("mw").str.to_integer(),
            pl.col("_ion_mode_short").str.to_uppercase(),
            pl.when(pl.col("_ion_mode_short").is_not_null())
            .then(pl.col("_ion_mode_short"))
            .when(pl.col("ion_mode_full").str.contains(r"(?i)positive"))
            .then(pl.lit("P"))
            .when(pl.col("ion_mode_full").str.contains(r"(?i)negative"))
            .then(pl.lit("N"))
            .when(pl.col("raw").str.contains(r"(?mi)^\s*Positive\b"))
            .then(pl.lit("P"))
            .when(pl.col("raw").str.contains(r"(?mi)^\s*Negative\b"))
            .then(pl.lit("N"))
            .otherwise(None)
            .alias("ion_mode"),
            pl.when(pl.col("ionization").is_not_null())
            .then(pl.col("ionization"))
            .when(pl.col("ion_mode_full").str.extract(r"(?i)(ESI|APCI|EI|CI|FAB|MALDI|DESI)").is_not_null())
            .then(pl.col("ion_mode_full").str.extract(r"(?i)(ESI|APCI|EI|CI|FAB|MALDI|DESI)"))
            .when(pl.col("raw").str.contains(r"(?mi)^\s*Electrospr[a-z]* ionization\b"))
            .then(pl.lit("ESI"))
            .otherwise(None)
            .alias("ionization"),
            pl.col("num_peaks").str.to_integer(),
            pl.coalesce([pl.col("precursor_type"), pl.col("adduct")]).alias("precursor_type"),
            pl.coalesce([pl.col("_precursor_mz_raw"), pl.col("_pepmass")]).cast(pl.Float64).alias("precursor_mz"),
            pl.coalesce([pl.col("_compound_name"), pl.col("_name"), pl.col("_title")]).alias("name"),
            pl.col("exact_mass").cast(pl.Float64, strict=False),
            pl.col("rt_seconds").cast(pl.Float64, strict=False),
            pl.col("ccs").cast(pl.Float64, strict=False),
            pl.col("precursor_intensity").cast(pl.Float64, strict=False),
            pl.col("collision_energy").cast(pl.Float64, strict=False),
            pl.col("collision_energy_1").cast(pl.Float64, strict=False),
            pl.col("collision_energy_2").cast(pl.Float64, strict=False),
            pl.col("collision_energy_3").cast(pl.Float64, strict=False),
            pl.col("normalized_collision_energy").cast(pl.Float64, strict=False),
            pl.col("normalized_collision_energy_1").cast(pl.Float64, strict=False),
            pl.col("normalized_collision_energy_2").cast(pl.Float64, strict=False),
            pl.col("normalized_collision_energy_3").cast(pl.Float64, strict=False),
            pl.col("precursor_im").cast(pl.Float64, strict=False),
            pl.col("spectral_entropy").cast(pl.Float64, strict=False),
            pl.col("pubchem_cid").str.to_integer(),
            pl.col("num_explained_peaks").str.to_integer(),
            pl.col("explained_intensity_raw").cast(pl.Float64, strict=False),
            pl.col("peaks_ppm").cast(pl.Float64, strict=False),
            pl.col("peaks_abs_ppm").cast(pl.Float64, strict=False),
            pl.col("precursor_ppm").cast(pl.Float64, strict=False),
            pl.col("mz_intensity").list.eval(pl.element().str.replace_all(r"\s+", " ").str.split(by=" ").list.get(index=0).cast(pl.Float64)).alias("raw_spectrum_mz"),
            pl.col("mz_intensity").list.eval(pl.element().str.replace_all(r"\s+", " ").str.split(by=" ").list.get(index=1).cast(pl.Float64)).alias("raw_spectrum_intensity"),
        )
        .drop("raw", "mz_intensity", "_pepmass", "ion_mode_full", "_ion_mode_short", "_precursor_mz_raw", "_compound_name", "_name", "_title")
    )

def _split_entries(file_contents: str) -> list[str]:
    entries = re.split(r"\n\s*\n", file_contents)
    if entries and entries[-1] == "":
        entries.pop()
    return entries

# Backward compatibility
from .spectral_library import process_single_file as read_MSPEC_file
from .spectral_library import process_spectral_library as create_nist_dataframe
