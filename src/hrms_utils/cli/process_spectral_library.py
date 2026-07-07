"""
Process Spectral Library files (MSP/MSPEC/MGF) to a single Parquet file.

Usage:
    build-spectral-library <path>

Where <path> can be:
- A single library file (.msp/.mspec/.mgf)
- A directory containing library files
"""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

import polars as pl

from hrms_utils.formats.spectral_library import deduplicate_spectra, process_spectral_library


def collect_library_files(path: Path) -> list[Path]:
    """Collect all supported library files from the given path."""
    valid_suffixes = {".msp", ".mspec", ".mgf", ".MSP", ".MSPEC", ".MGF"}

    if path.is_file():
        assert path.suffix in valid_suffixes, (
            f"File {path} does not have a valid library suffix: {path.suffix}"
        )
        return [path]

    if path.is_dir():
        files = [
            f for f in path.iterdir() if f.is_file() and f.suffix in valid_suffixes
        ]
        assert len(files) > 0, f"No library files found in directory: {path}"
        return sorted(files)

    raise ValueError(f"Path does not exist or is not a file/directory: {path}")


def configure_file_logging(log_path: Path) -> logging.Logger:
    """Configure real-time file logging so crashes can be pinpointed by timestamp."""
    logging.basicConfig(
        filename=str(log_path),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger("hrms_utils.formats.spectral_library")


def main():
    parser = argparse.ArgumentParser(
        description="Process Spectral Library files to Parquet with optional deduplication and enrichment"
    )
    parser.add_argument(
        "input_path", type=Path, help="File or directory with library files"
    )
    parser.add_argument(
        "--pubchem",
        "-p",
        type=Path,
        default=None,
        help="Parquet with PubChem data for enrichment",
    )
    parser.add_argument(
        "--raw-fragment-tolerance-ppm",
        "-r",
        type=float,
        default=10.0,
        help="Raw fragment tolerance in ppm (default: 10.0)",
    )
    parser.add_argument(
        "--normalized-fragment-tolerance-ppm",
        "-n",
        type=float,
        default=5.0,
        help="Normalized fragment tolerance in ppm (default: 5.0)",
    )
    parser.add_argument(
        "--molecular-ion-tolerance-ppm",
        "-m",
        type=float,
        default=5.0,
        help="Molecular ion tolerance in ppm (default: 5.0)",
    )
    parser.add_argument(
        "--ms2-tolerance-ppm",
        "-t",
        type=float,
        default=5.0,
        help="MS2 tolerance in ppm for deduplication (default: 5.0)",
    )
    parser.add_argument(
        "--dedup-threshold",
        "-d",
        type=float,
        default=0.99,
        help="Explained intensity threshold for deduplication (default: 0.99)",
    )
    parser.add_argument(
        "--min-explained-intensity",
        "-e",
        type=float,
        default=0.95,
        help="Minimum explained intensity to keep a spectrum (default: 0.95)",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        default=False,
        help="Skip pairwise spectrum deduplication",
    )
    parser.add_argument(
        "--prepared-library",
        action="store_true",
        default=False,
        help=(
            "Treat input_path as a single already-annotated .parquet library "
            "(e.g. from a prior run with --no-deduplicate) and run ONLY "
            "deduplication. Output is written to <input_stem>_deduplicated.parquet. "
            "In this mode --ms2-tolerance-ppm controls the fragment tolerance "
            "for deduplication (not --normalized-fragment-tolerance-ppm)."
        ),
    )
    parser.add_argument(
        "--no-clean-identifiers",
        action="store_true",
        default=False,
        help="Skip molecular identifier cleaning/standardization; use identifiers as-is from the input file",
    )
    parser.add_argument(
        "--pubchem-enrich-all",
        action="store_true",
        default=False,
        help="Use PubChem to enrich all rows, not only rows missing both SMILES and InChI",
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=Path,
        default=None,
        help="Path for the execution log file (default: next to the output Parquet)",
    )
    parser.add_argument(
        "--include-msn",
        action="store_true",
        default=False,
        help="Process and include MSn (multi-level MS) spectra from MGF files",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=500_000,
        help="Number of rows to process per batch (default: 500000)",
    )

    args = parser.parse_args()

    input_path = args.input_path.resolve()
    assert input_path.exists(), f"Input path does not exist: {input_path}"
    if args.prepared_library:
        # --- Dedup-only mode: input must be a single annotated Parquet. ---
        assert args.no_deduplicate is False, (
            "--prepared-library runs deduplication; it is incompatible with --no-deduplicate."
        )
        assert input_path.is_file(), (
            f"--prepared-library requires a single .parquet file, got directory: {input_path}"
        )
        assert input_path.suffix.lower() == ".parquet", (
            f"--prepared-library requires a .parquet file, got suffix '{input_path.suffix}' "
            f"on {input_path}"
        )

        output_path = input_path.with_name(f"{input_path.stem}_deduplicated.parquet")
        log_path = (
            args.log_file.resolve()
            if args.log_file is not None
            else output_path.with_suffix(".log")
        )
        logger = configure_file_logging(log_path)

        print(f"Deduplicating prepared library: {input_path}")
        start = perf_counter()

        result_lf = deduplicate_spectra(
            input_path=input_path,
            output_path=output_path,
            fragment_tolerance_ppm=args.ms2_tolerance_ppm,
            molecular_ion_tolerance_ppm=args.molecular_ion_tolerance_ppm,
            threshold=args.dedup_threshold,
            logger=logger,
        )

        end = perf_counter()
        n_spectra = result_lf.select(pl.len()).collect().item()
        n_molecules = result_lf.select(pl.col("base_inchikey").n_unique()).collect().item()
        print(f"Processed {n_spectra} spectra in {end - start:.2f} seconds")
        print(f"Unique compounds: {n_molecules}")
        print(f"Success! Output written to {output_path}")
        logger.info(f"Output written to {output_path}")
        print(f"Log written to {log_path}")
        return

    pubchem_path = args.pubchem.resolve() if args.pubchem is not None else None

    # Collect all matching files
    library_files = collect_library_files(input_path)
    print(f"Found {len(library_files)} file(s) to process")

    start = perf_counter()

    # Determine output path
    if input_path.is_file():
        output_path = input_path.with_suffix(".parquet")
        inchikey_changes_path = input_path.with_suffix(".inchikey_changes.csv")
    else:
        output_path = input_path / f"{input_path.name}.parquet"
        inchikey_changes_path = input_path / f"{input_path.name}.inchikey_changes.csv"

    # Configure real-time file logging so crashes can be pinpointed by timestamp.
    log_path = (
        args.log_file.resolve()
        if args.log_file is not None
        else output_path.with_suffix(".log")
    )
    logger = configure_file_logging(log_path)

    # Use the unified API
    result_lf = process_spectral_library(
        files=library_files,
        raw_fragment_tolerance_ppm=args.raw_fragment_tolerance_ppm,
        normalized_fragment_tolerance_ppm=args.normalized_fragment_tolerance_ppm,
        molecular_ion_tolerance_ppm=args.molecular_ion_tolerance_ppm,
        includes_MSn=args.include_msn,
        pubchem_path=pubchem_path,
        pubchem_fill_missing_only=not args.pubchem_enrich_all,
        min_explained_intensity=args.min_explained_intensity,
        dedup_threshold=args.dedup_threshold,
        deduplicate=not args.no_deduplicate,
        clean_identifiers=not args.no_clean_identifiers,
        inchikey_changes_path=inchikey_changes_path,
        output_path=output_path,
        logger=logger,
        batch_size=args.batch_size,
    )

    end = perf_counter()
    n_spectra = result_lf.select(pl.len()).collect().item()
    n_molecules = result_lf.select(pl.col("base_inchikey").n_unique()).collect().item()
    print(f"Processed {n_spectra} spectra in {end - start:.2f} seconds")
    print(f"Unique compounds: {n_molecules}")

    print(f"Success! Output written to {output_path}")
    logger.info(f"Output written to {output_path}")
    print(f"Log written to {log_path}")
