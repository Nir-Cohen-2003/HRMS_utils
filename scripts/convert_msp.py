"""
Convert MSP/MSPEC files to a single Parquet file.

Usage:
    python convert_msp.py <path>

Where <path> can be:
- A single .msp/.mspec file
- A directory containing .msp/.mspec files

The script will read all matching files, concatenate them into a single Polars DataFrame,
and write the result as a Parquet file adjacent to the input with the same base name.
"""

import sys
from pathlib import Path
from typing import cast
import polars as pl
from hrms_utils.formats.nist_mspec import read_MSPEC_file
from time import perf_counter
import argparse
from hrms_utils.formula_annotation.utils import format_formula_string_to_array, num_elements, formula_to_array

def collect_mspec_files(path: Path) -> list[Path]:
    """
    Collect all MSPEC/MSP files from the given path.
    
    Args:
        path: Either a file or directory path
        
    Returns:
        List of Path objects for all matching files
    """
    valid_suffixes = {'.msp', '.mspec', '.MSP', '.MSPEC'}
    
    if path.is_file():
        assert path.suffix in valid_suffixes, f"File {path} does not have a valid MSPEC/MSP suffix: {path.suffix}"
        return [path]
    
    if path.is_dir():
        files = [f for f in path.iterdir() if f.is_file() and f.suffix in valid_suffixes]
        assert len(files) > 0, f"No MSPEC/MSP files found in directory: {path}"
        return sorted(files)
    
    raise ValueError(f"Path does not exist or is not a file/directory: {path}")


def main():
    parser = argparse.ArgumentParser(description="Convert MSP/MSPEC files to Parquet, optionally enrich with PubChem data")
    parser.add_argument("input_path", type=Path, help="File or directory with .msp/.mspec files")
    parser.add_argument("--pubchem", "-p", type=Path, default=None, help="Parquet (or directory partition) with PubChem data (schema includes InChIKey, SMILES, InChI, Formula, monoisotopic_mass, exact_mass)")
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    assert input_path.exists(), f"Input path does not exist: {input_path}"
    pubchem_path = args.pubchem.resolve() if args.pubchem is not None else None
    if pubchem_path is not None:
        assert pubchem_path.exists(), f"Pubchem path does not exist: {pubchem_path}"

    # Collect all matching files
    mspec_files = collect_mspec_files(input_path)
    print(f"Found {len(mspec_files)} MSPEC/MSP file(s) to process")
    
    # Read lazy frames for streaming
    start = perf_counter()
    lazyframes = []
    for file_path in mspec_files:
        print(f"Reading {file_path.name}...")
        df = read_MSPEC_file(
            file_path,
            raw_fragment_tolerance_ppm=10.0,
            normalized_fragment_tolerance_ppm=5.0,
            molecular_ion_tolerance_ppm=5.0,
            lazy=True
        )
        lazyframes.append(df)

    print(f"Concatenating {len(lazyframes)} lazyframes")
    combined_lf = cast(pl.LazyFrame, pl.concat(lazyframes, how="vertical")) # still LazyFrame
    combined_df = combined_lf.filter(
        pl.col("clean_precursor"),
        pl.col("explained_intensity") > 0.95,
        pl.col("is_ESI"),
        pl.col("is_orbitrap")
    ).collect(engine='streaming')  # Collect after filtering in streaming mode
    print(f"After filtering, we are left with {combined_df.height} spectra and {combined_df.unique(subset='base_inchikey').height} unique 2d structures (by base_inchikey)")
    combined_lf = combined_df.lazy()  # Convert back to LazyFrame for further processing

    # Add a stable row identifier to the MSP LazyFrame so we can reference specific spectra later.
    # Why: we will use this index to reduce PubChem rows deterministically and to report/trace matches.
    combined_lf = combined_lf.with_row_index("msp_index")

    # Create a minimal MSP lazyframe with only the fields required for joins and comparisons.
    # This reduces memory/IO when we join with the large PubChem table and makes intent explicit.
    msp_join_lf = combined_lf.select(
        [
            "msp_index",
            "base_inchikey",
            "exact_mass",
            "molecular_formula_array"
        ]
    )

    # Optionally join pubchem data using streaming before collection
    if pubchem_path is not None:
        print(f"Loading pubchem data from {pubchem_path} (streaming)")
        # Read only the fields we need for enrichment
        pubchem_lf = pl.scan_parquet(
            source=str(pubchem_path),
            low_memory=True
        ).select(
            [
                pl.col("CID"),
                pl.col("InChIKey"),
                pl.col("SMILES"),
                pl.col("InChI"),
                pl.col("Formula"),
                # pl.col("exact_mass"),
                pl.col("monoisotopic_mass")
            ]
        ).rename({
            "CID": "cid",
            "InChIKey": "inchikey",
            "SMILES": "smiles",
            "InChI": "inchi",
            "Formula": "formula",
            # "exact_mass": "exact_mass"
        }).with_columns(
            # create base_inchikey for joining to MSP; keep lower-case for consistency with MSP
            pl.col("inchikey").str.split(by="-").list.get(0).alias("base_inchikey")
        )

        # Convert Formula -> formula array using the repository util (works on LazyFrame)
        pubchem_lf = formula_to_array(pubchem_lf, "formula", "pubchem_formula_array")

        # Join PubChem with the MSP minimal LF before deduplication.
        # Why: we only want to keep PubChem rows that actually correspond to at least one MSP entry,
        # and where mass/formula agree — this allows safe deduplication of PubChem records.
        pubchem_with_msp_lf = pubchem_lf.join(
            msp_join_lf,
            on="base_inchikey",
            how="inner",  # only keep pubchem rows for base_inchikeys present in MSP
            suffix="_msp"
        ).with_columns(
            # boolean checks same as later, but applied to joined pubchem -> msp pairing
            masses_match=pl.col("exact_mass").is_not_null() & (
                pl.col("exact_mass").is_close(
                    pl.col("monoisotopic_mass"),
                    rel_tol=1e-6,
                    abs_tol=1e-3
                )
            ),
            formula_array_match_with_pubchem=pl.col("pubchem_formula_array").is_not_null() & (
                pl.col("molecular_formula_array") == pl.col("pubchem_formula_array")
            ),
        ).filter(
            # Keep only PubChem rows that match both mass and formula for at least one MSP
            pl.col("masses_match") & pl.col("formula_array_match_with_pubchem")
        )

        # Reduce PubChem rows. We want one pubchem row per base_inchikey for joining back.
        # Use the MSP index as tie-breaker when multiple pubchem rows match the same base_inchikey;
        # this makes deterministic reduction based on MSP ordering.
        pubchem_reduced_lf = (
            pubchem_with_msp_lf
            .sort("msp_index")  # ensure deterministic choice of which pubchem row to keep
            .unique(subset="base_inchikey", keep="first")
            .select(
                [
                    "cid",
                    "base_inchikey",
                    "smiles",
                    "inchi",
                    "inchikey",
                    "formula",
                    "pubchem_formula_array",
                    "monoisotopic_mass"
                ]
            )
        )

        # Left-join the reduced PubChem onto the full MSP lazyframe.
        # Why: keep all MSP rows (left join) even when there is no PubChem match.
        combined_lf = combined_lf.join(
            pubchem_reduced_lf,
            on="base_inchikey",
            how="left",
            suffix="_pubchem"
        )

        # Mark whether a PubChem row was found for this MSP entry.
        # This makes it explicit that "mismatch" -> either a disagreement with PubChem
        # (when pubchem match exists) or absence of any PubChem match (when false).
        combined_lf = combined_lf.with_columns(
            pubchem_match_found=pl.col("cid").is_not_null()
        )

        # Create aliases and compute boolean mismatch columns using Polars expressions (keeps processing streaming)
        # We compute masses/formula matches only when a pubchem match is present; otherwise they are False.
        combined_lf = combined_lf.with_columns(
            masses_match=pl.when(pl.col("pubchem_match_found")).then(
                pl.col("exact_mass").is_not_null() & (
                    pl.col("exact_mass").is_close(
                        pl.col("monoisotopic_mass"),
                        rel_tol=1e-6,
                        abs_tol=1e-3
                    )
                )
            ).otherwise(False),
            formula_array_match_with_pubchem=pl.when(pl.col("pubchem_match_found")).then(
                pl.col("pubchem_formula_array").is_not_null() & (
                    pl.col("molecular_formula_array") == pl.col("pubchem_formula_array")
                )
            ).otherwise(False),
        )

    # Now collect everything in streaming mode and continue with filtering/measures
    print(f"Collecting concatenated results (streaming)")
    combined_df = combined_lf.collect(engine='streaming')
    end = perf_counter()
    print(f"Combined dataframe has {combined_df.height} spectra, and {combined_df.unique(subset='base_inchikey').height} unique 2d structures (by base_inchikey)")
    print(f"Completed reading and concatenation in {end - start:.2f} seconds")

    # If pubchem was provided, compare masses and formulas; report mismatches
    if pubchem_path is not None:
        # Category 1: Match found, but mismatch in mass OR formula
        mismatch_filter = pl.col("pubchem_match_found") & (
            ~pl.col("masses_match") | ~pl.col("formula_array_match_with_pubchem")
        )
        mismatched_df = combined_df.filter(mismatch_filter)
        n_mismatched_spectra = mismatched_df.height
        n_mismatched_compounds = mismatched_df.unique(subset="base_inchikey").height
        
        print(f"\n--- Category 1: Spectra with PubChem match but Mass/Formula mismatch ---")
        print(f"Spectra: {n_mismatched_spectra}")
        print(f"Compounds: {n_mismatched_compounds}")
        if n_mismatched_spectra > 0:
            print("Examples (first 5):")
            print(mismatched_df.select(
                ["name", "base_inchikey", "exact_mass", "monoisotopic_mass", "molecular_formula", "pubchem_formula_array"]
            ).head(5))

        # Category 2: No match in PubChem, but has InChI OR SMILES (from MSP)
        # Note: At this stage, 'inchi' and 'smiles' are from MSP only (before fallback)
        has_struct_no_match_filter = ~pl.col("pubchem_match_found") & (
            (pl.col("inchi").is_not_null() & (pl.col("inchi") != "")) | 
            (pl.col("smiles").is_not_null() & (pl.col("smiles") != ""))
        )
        has_struct_no_match_df = combined_df.filter(has_struct_no_match_filter)
        n_struct_no_match_spectra = has_struct_no_match_df.height
        n_struct_no_match_compounds = has_struct_no_match_df.unique(subset="base_inchikey").height

        print(f"\n--- Category 2: Spectra with InChI/SMILES but NO PubChem match ---")
        print(f"Spectra: {n_struct_no_match_spectra}")
        print(f"Compounds: {n_struct_no_match_compounds}")
        if n_struct_no_match_spectra > 0:
            print("Examples (first 5):")
            print(has_struct_no_match_df.select(
                ["name", "base_inchikey", "inchi", "smiles"]
            ).head(5))

        # Category 3: No match in PubChem, AND no InChI AND no SMILES
        no_struct_no_match_filter = ~pl.col("pubchem_match_found") & (
            (pl.col("inchi").is_null() | (pl.col("inchi") == "")) & 
            (pl.col("smiles").is_null() | (pl.col("smiles") == ""))
        )
        no_struct_no_match_df = combined_df.filter(no_struct_no_match_filter)
        n_no_struct_no_match_spectra = no_struct_no_match_df.height
        n_no_struct_no_match_compounds = no_struct_no_match_df.unique(subset="base_inchikey").height

        print(f"\n--- Category 3: Spectra with NO InChI/SMILES and NO PubChem match ---")
        print(f"Spectra: {n_no_struct_no_match_spectra}")
        print(f"Compounds: {n_no_struct_no_match_compounds}")
        if n_no_struct_no_match_spectra > 0:
            print("Examples (first 5):")
            print(no_struct_no_match_df.select(
                ["name", "base_inchikey"]
            ).head(5))
        print("\n")

    # If we have pubchem and pubchem columns exist, prefer MSP values but fill in missing smiles/inchi from PubChem.
    # Why: Keep original curated identifiers (MSP) when present; otherwise fall back to authoritative PubChem annotation.
    if pubchem_path is not None:
        # Determine actual PubChem field names after join (may be suffixed if collision)
        pubchem_smiles_candidates = [c for c in ("smiles_pubchem", "smiles") if c in combined_df.columns]
        pubchem_inchi_candidates = [c for c in ("inchi_pubchem", "inchi") if c in combined_df.columns]

        pubchem_smiles_col = pubchem_smiles_candidates[0] if pubchem_smiles_candidates else None
        pubchem_inchi_col = pubchem_inchi_candidates[0] if pubchem_inchi_candidates else None

        # Fill missing 'smiles' / 'inchi' from PubChem when available (fallback per-field).
        # Why: we prefer the MSP-provided identifiers; if either is missing, use PubChem's value to salvage downstream annotation.
        fill_inchi = (pl.col(pubchem_inchi_col) if pubchem_inchi_col is not None else pl.lit(None))
        fill_smiles = (pl.col(pubchem_smiles_col) if pubchem_smiles_col is not None else pl.lit(None))

        combined_df = combined_df.with_columns(
            pl.when(pl.col("smiles").is_null() | (pl.col("smiles") == "")).then(fill_smiles).otherwise(pl.col("smiles")).alias("smiles"),
            pl.when(pl.col("inchi").is_null() | (pl.col("inchi") == "")).then(fill_inchi).otherwise(pl.col("inchi")).alias("inchi"),
        )

    # Determine output path: same location as input, with .parquet extension
    if input_path.is_file():
        output_path = input_path.with_suffix('.parquet')
    else:
        # For directories, use the directory name as the base file name
        output_path = input_path / f"{input_path.name}.parquet"

    # If any rows are still missing both smiles and inchi after fallback, report and write them separately.
    # We consider the compound-level reduction by base_inchikey for the compound counts.
    if pubchem_path is not None:
        no_identifiers = combined_df.filter(
            (pl.col("smiles").is_null() | (pl.col("smiles") == "")) &
            (pl.col("inchi").is_null() | (pl.col("inchi") == ""))
        )
        n_spectra_no_ident = no_identifiers.height
        n_compounds_no_ident = no_identifiers.unique(subset="base_inchikey").height

        print(f"Spectra lacking both InChI and SMILES after PubChem fallback: {n_spectra_no_ident}")
        print(f"Unique compounds (by base_inchikey) lacking both after PubChem fallback: {n_compounds_no_ident}")

        if n_spectra_no_ident > 0:
            no_pubchem_out = output_path.with_name(output_path.stem + "_no_pubchem_match.parquet")
            print(f"Writing {n_spectra_no_ident} spectra lacking both InChI/SMILES to {no_pubchem_out}")
            no_identifiers.write_parquet(no_pubchem_out)

    print(f"Writing to {output_path}...")
    combined_df.write_parquet(output_path)
    print(f"Successfully wrote {combined_df.height} spectra to {output_path}")


if __name__ == "__main__":
    main()