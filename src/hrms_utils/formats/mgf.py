import polars as pl
from pathlib import Path
import re

def read_all_ms2_files(dir_path: Path | str) -> pl.DataFrame:
    """Reads all MGF files in a directory and extracts MS2 spectra."""
    dir_path = Path(dir_path)
    dfs = []
    for file_path in dir_path.glob("*.mgf"):
        dfs.append(parse_mgf(file_path, includes_MSn=False).collect())
    
    df = pl.concat(dfs)
    return df.filter(
        pl.col("is_single_spectra"),
        pl.col("mslevel").eq(2)
    )

def parse_mgf(mgf_path: str | Path, includes_MSn: bool = False) -> pl.LazyFrame:
    """Thin parser for MGF files. Returns a LazyFrame with unified column names."""
    mgf_path = Path(mgf_path)
    with open(mgf_path, 'r') as f:
        mgf_text = f.read()
        entries = re.findall(r'BEGIN IONS(.*?)END IONS', mgf_text, re.DOTALL)
    
    lf = pl.DataFrame({'entry': entries}).lazy()
    
    meta_keys = {
        "NAME": "name",
        "COMPOUND_NAME": "name",
        "EXACTMASS": "exact_mass",
        "EXACT_MASS": "exact_mass",
        "FORMULA": "molecular_formula",
        "MOLECULAR_FORMULA": "molecular_formula",
        "INCHI": "inchi",
        "INCHIAUX": "inchikey",
        "INCHIKEY": "inchikey",
        "SMILES": "smiles",
        "MSLEVEL": "mslevel",
        "MS_LEVEL": "mslevel",
        "ADDUCT": "precursor_type",
        "PRECURSOR_TYPE": "precursor_type",
        "PEPMASS": "precursor_mz",
        "PRECURSOR_MZ": "precursor_mz",
        "IONMODE": "ion_mode",
        "ION_MODE": "ion_mode",
        "INSTRUMENT_TYPE": "instrument_type",
        "SOURCE_INSTRUMENT": "instrument",
        "ION_SOURCE": "ionization",
        "RTINSECONDS": "rt_seconds",
        "RT_SECONDS": "rt_seconds",
        "CHARGE": "charge",
        "FEATURE_ID": "feature_id",
        "DESCRIPTION": "description",
        "FEATURE_MS1_HEIGHT": "feature_ms1_height",
        "SPECTYPE": "spectype",
        "SPEC_TYPE": "spectype",
        "FRAGMENTATION_METHOD": "fragmentation_method",
        "ISOLATION_WINDOW": "isolation_window",
        "ACQUISITION": "acquisition",
        "IMS_TYPE": "ims_type",
        "PI": "pi",
        "DATACOLLECTOR": "datacollector",
        "DATASET_ID": "dataset_id",
        "USI": "usi",
        "SCANS": "scans",
        "PRECURSOR_PURITY": "precursor_purity",
        "QUALITY_EXPLAINED_INTENSITY": "quality_explained_intensity",
        "QUALITY_EXPLAINED_SIGNALS": "quality_explained_signals",
        "Num peaks": "num_peaks",
        "NUM_PEAKS": "num_peaks"
    }
    
    # We first extract all raw values from meta_keys.
    # Note: multiple keys can map to the same unified_name.
    # We use (?mi) for case-insensitive and multi-line matching.
    raw_exprs = []
    for key, unified_name in meta_keys.items():
        raw_exprs.append(
            pl.col("entry").str.extract(rf"(?mi)^{key}=(.+)$", 1).alias(f"raw_{key}")
        )

    # Any non-MSn key containing "energy" is treated as the collision-energy
    # field. MSn_* keys are parsed separately when includes_MSn is enabled.
    raw_exprs.append(
        pl.col("entry").str.extract(
            r"(?mi)^(?:energy[^=\n]*|(?:[^M\n]|M[^S\n]|MS[^n\n]|MSn[^_\-\n])[^=\n]*energy)[^=\n]*=(.+)$",
            1,
        ).alias("collision_energy_raw")
    )

    lf = lf.with_columns(raw_exprs)

    # Check for mismatches when multiple keys map to the same unified name
    unified_to_keys = {}
    for key, unified_name in meta_keys.items():
        unified_to_keys.setdefault(unified_name, []).append(key)
    
    for unified_name, keys in unified_to_keys.items():
        if len(keys) > 1:
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    k1, k2 = keys[i], keys[j]
                    # Check if both raw columns exist in the LazyFrame
                    if f"raw_{k1}" not in lf.collect_schema().names() or f"raw_{k2}" not in lf.collect_schema().names():
                        continue
                    
                    # Check for mismatch between raw_k1 and raw_k2 (ignoring nulls)
                    mismatch_mask = (
                        pl.col(f"raw_{k1}").is_not_null() & 
                        pl.col(f"raw_{k2}").is_not_null() & 
                        (pl.col(f"raw_{k1}") != pl.col(f"raw_{k2}"))
                    )
                    
                    mismatch_count = lf.select(mismatch_mask.sum()).collect().item()
                    if mismatch_count > 0:
                        example = lf.filter(mismatch_mask).select(
                            f"raw_{k1}", f"raw_{k2}"
                        ).limit(1).collect()
                        val1 = example.item(0, 0)
                        val2 = example.item(0, 1)
                        raise ValueError(
                            f"Mismatch in file {mgf_path}: keys '{k1}' and '{k2}' both map to '{unified_name}' "
                            f"but have different values in {mismatch_count} entries. "
                            f"Example mismatch: '{k1}'='{val1}', '{k2}'='{val2}'"
                        )

    # Now create the unified columns using the first non-null available key
    exprs = []
    for unified_name, keys in unified_to_keys.items():
        # Only use keys that actually produced a raw column
        valid_keys = [k for k in keys if f"raw_{k}" in lf.collect_schema().names()]
        if not valid_keys:
            continue
        
        if len(valid_keys) == 1:
            exprs.append(pl.col(f"raw_{valid_keys[0]}").alias(unified_name))
        else:
            # coalescing across multiple possible raw keys
            exprs.append(
                pl.coalesce([pl.col(f"raw_{k}") for k in valid_keys]).alias(unified_name)
            )
    
    # Clean up raw columns
    available_raw_cols = [f"raw_{k}" for k in meta_keys.keys() if f"raw_{k}" in lf.collect_schema().names()]
    lf = lf.with_columns(exprs).drop(available_raw_cols)
    
    msn_keys = [
        "MSn_collision_energies", "MSn_precursor_mzs", "MSn_fragmentation_methods", "MSn_isolation_windows"
    ]
    exprs2 = []
    if includes_MSn:
        for key in msn_keys:
            exprs2.append(
                pl.col("entry").str.extract(rf"(?m)^{key}=(.+)$", 1).alias(key.lower())
            )
            
    exprs2.append(
        pl.col("entry")
        .str.extract_all(r"(?m)^(\d+\.\d+)\s+(\d+\.\d+(?:[eE][+-]?\d+)?)$")
        .alias("mz_int_pairs")
    )
    
    lf = lf.with_columns(exprs2).drop("entry")
    
    # Renaming and casting
    lf = lf.with_columns(
        pl.col("spectype").fill_null(value="SINGLE_BEST_SCAN"),
        pl.col("ion_mode").str.to_lowercase().map_elements(
            lambda x: "P" if x == "positive" else "N" if x == "negative" else x,
            return_dtype=pl.String
        ),
        pl.col("mslevel").str.extract(r"(?i)MS?(\d+)", 1).fill_null(pl.col("mslevel"))
    ).cast({
        "exact_mass": pl.Float64,
        "rt_seconds": pl.Float64,
        "precursor_mz": pl.Float64,
        "charge": pl.Int64,
        "feature_ms1_height": pl.Float64,
        "mslevel": pl.Int64,
        "isolation_window": pl.Float64,
        "num_peaks": pl.Int64,
        "precursor_purity": pl.Float64,
        "quality_explained_intensity": pl.Float64,
        "quality_explained_signals": pl.Float64
    })
    
    # Handling of spectrum
    lf = lf.with_columns(
        pl.col("mz_int_pairs")
        .list.eval(pl.element().str.replace_all(r"\s+", " ").str.split(by=" ").list.get(0).cast(pl.Float64))
        .alias("raw_spectrum_mz"),
        pl.col("mz_int_pairs")
        .list.eval(pl.element().str.replace_all(r"\s+", " ").str.split(by=" ").list.get(1).cast(pl.Float64))
        .alias("raw_spectrum_intensity"),
    ).drop("mz_int_pairs")
    
    # USI and merged spectra flags
    lf = lf.with_columns(
        pl.col("usi").str.strip_chars("[]").str.split(by=",").alias("usi"),
    ).with_columns(
        pl.col("usi").list.len().alias("num_merged_spectra"),
    ).with_columns(
        pl.col("num_merged_spectra").eq(1).alias("is_single_spectra")
    )
    
    # COLLISION_ENERGY handling (MS2) - keep as list if already formatted
    lf = lf.with_columns(
        pl.col("collision_energy_raw").str.strip_chars("[]").str.split(by=",").list.eval(
            pl.element().str.strip_chars(" ")
        ).cast(pl.List(pl.Float64)).alias("collision_energy_list")
    ).with_columns(
        pl.when(pl.col("collision_energy_list").list.len() > 1)
        .then(pl.lit(True)).otherwise(pl.lit(False)).alias("multiple_collision_energies"),
        pl.col("collision_energy_list").list.mean().alias("collision_energy_mean")
    )
    
    if includes_MSn:
        # We'll just do basic list parsing for MSn fields
        for key in ["msn_precursor_mzs", "msn_isolation_windows"]:
            lf = lf.with_columns(
                pl.when(pl.col(key).is_not_null())
                .then(pl.col(key).str.strip_chars("[]").str.split(",").list.eval(pl.element().str.strip_chars(" ").cast(pl.Float64)))
                .otherwise(None).alias(key)
            )
        lf = lf.with_columns(
            pl.when(pl.col("msn_fragmentation_methods").is_not_null())
            .then(pl.col("msn_fragmentation_methods").str.strip_chars("[]").str.split(",").list.eval(pl.element().str.strip_chars(" ")))
            .otherwise(None).alias("msn_fragmentation_methods")
        )
        # For energies, we use the complex logic from original mgf.py
        lf = lf.with_columns(
            pl.when(pl.col("msn_collision_energies").is_not_null())
            .then(
                pl.col("msn_collision_energies")
                .str.strip_prefix('[').str.strip_suffix(']')
                .str.replace_all("],", "]|")
                .str.split("|")
                .list.eval(
                    pl.element().str.strip_prefix('[').str.strip_suffix(']')
                    .str.split(",")
                    .list.eval(pl.element().str.strip_chars(" ").cast(pl.Float64, strict=False))
                )
            ).alias("msn_collision_energies")
        )
        
    return lf

# Backward compatibility
read_mgf_to_dataframe = parse_mgf
