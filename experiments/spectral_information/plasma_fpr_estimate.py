"""
Plasma FPR (False Positive Rate) Estimate Script

This script takes mzML files (or MSDIAL chromatograms), runs MSDIAL processing if needed,
matches chromatograms against a spectral library using precursor mass tolerance and 
spectral similarity, and outputs the best matches for FPR estimation.

Usage with mzML files (MSDIAL will run automatically):
    pixi run -e experiments python experiments/spectral_information/plasma_fpr_estimate.py \
        --input-mzmls /path/to/mzmls/ \
        --library /path/to/library.parquet \
        --output ./results \
        --match-threshold 0.75 \
        --precursor-tolerance-ppm 5.0

Usage with pre-processed MSDIAL chromatograms:
    pixi run -e experiments python experiments/spectral_information/plasma_fpr_estimate.py \
        --chromatograms-dir /path/to/chromatograms/ \
        --library /path/to/library.parquet \
        --output ./results \
        --match-threshold 0.75 \
        --precursor-tolerance-ppm 5.0
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import polars as pl

from hrms_utils.formats.msdial import (
    annotate_chromatogram_with_formulas,
    get_chromatogram,
    run_msdial_lcmsdda,
    MSDialRunnerConfig,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Match chromatograms against spectral library for FPR estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input group - mutually exclusive input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-mzmls",
        type=str,
        help="Directory containing mzML files (MSDIAL will run automatically). Expects positive/ and negative/ subdirectories.",
    )
    input_group.add_argument(
        "--chromatograms-dir",
        type=str,
        help="Directory containing pre-processed MSDIAL chromatogram .txt or .mdpeak files",
    )

    parser.add_argument(
        "--library",
        type=str,
        required=True,
        help="Path to spectral library parquet file",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results",
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.75,
        help="Minimum spectral similarity score for a match",
    )

    parser.add_argument(
        "--precursor-tolerance-ppm",
        type=float,
        default=5.0,
        help="Precursor mass tolerance in ppm for matching",
    )

    parser.add_argument(
        "--ms2-tolerance-ppm",
        type=float,
        default=10.0,
        help="MS/MS fragment mass tolerance in ppm for similarity calculation",
    )

    parser.add_argument(
        "--precursor-mass-accuracy-ppm",
        type=float,
        default=3.0,
        help="Precursor mass accuracy in ppm for formula annotation",
    )

    parser.add_argument(
        "--fragment-mass-accuracy-ppm",
        type=float,
        default=5.0,
        help="Fragment mass accuracy in ppm for formula annotation",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="plasma_fpr_matches",
        help="Prefix for output files",
    )

    # MSDIAL parameters
    parser.add_argument(
        "--msdial-path",
        type=str,
        default=None,
        help="Path to MS-DIAL executable (auto-detected if not specified)",
    )

    parser.add_argument(
        "--msdial-threads",
        type=int,
        default=20,
        help="Number of threads for MS-DIAL processing",
    )

    parser.add_argument(
        "--msdial-min-peak-height",
        type=int,
        default=100000,
        help="Minimum peak height for MS-DIAL feature detection",
    )

    return parser.parse_args()


def run_msdial_on_mzmls(
    input_dir: Path,
    output_dir: Path,
    msdial_path: Optional[Path] = None,
    threads: int = 20,
    minimum_peak_height: int = 100000,
) -> Path:
    """
    Run MSDIAL on mzML files in the input directory.

    Expects input_dir to have positive/ and negative/ subdirectories with mzML files.

    Args:
        input_dir: Directory containing mzML files
        output_dir: Directory for MSDIAL output
        msdial_path: Path to MS-DIAL executable (auto-detected if None)
        threads: Number of threads for MS-DIAL processing
        minimum_peak_height: Minimum peak height for feature detection

    Returns:
        Path to the MSDIAL output directory
    """
    print(f"Running MS-DIAL on mzML files...")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create MSDIAL config
    config = MSDialRunnerConfig(
        msdial_path=msdial_path,
        threads=threads,
        minimum_peak_height=minimum_peak_height,
    )

    # Process positive mode if available
    positive_dir = input_dir / "positive"
    if positive_dir.exists() and any(positive_dir.glob("*.mzML")):
        print(f"\n  Processing positive mode mzMLs...")
        positive_output = output_dir / "positive"
        try:
            run_msdial_lcmsdda(
                input_dir=positive_dir,
                output_dir=positive_output,
                polarity="positive",
                config=config,
            )
            print(f"  Positive mode complete. Output: {positive_output}")
        except Exception as e:
            print(f"  Warning: Positive mode processing failed: {e}")
    else:
        print(f"  No positive mode mzML files found in {positive_dir}")

    # Process negative mode if available
    negative_dir = input_dir / "negative"
    if negative_dir.exists() and any(negative_dir.glob("*.mzML")):
        print(f"\n  Processing negative mode mzMLs...")
        negative_output = output_dir / "negative"
        try:
            run_msdial_lcmsdda(
                input_dir=negative_dir,
                output_dir=negative_output,
                polarity="negative",
                config=config,
            )
            print(f"  Negative mode complete. Output: {negative_output}")
        except Exception as e:
            print(f"  Warning: Negative mode processing failed: {e}")
    else:
        print(f"  No negative mode mzML files found in {negative_dir}")

    return output_dir


def find_chromatogram_files(directory: Path) -> "List[tuple[Path, str]]":
    """
    Find all chromatogram files (.txt or .mdpeak) in the given directory.
    
    Expects directory structure with 'positive' and 'negative' subdirectories.

    Args:
        directory: Path to the directory to search

    Returns:
        List of tuples (file_path, ion_mode) where ion_mode is 'positive' or 'negative'
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_files: List[tuple[Path, str]] = []
    
    # Look for positive and negative subdirectories
    for ion_mode in ["positive", "negative"]:
        mode_dir = directory / ion_mode
        if mode_dir.exists() and mode_dir.is_dir():
            txt_files = list(mode_dir.glob("*.txt"))
            mdpeak_files = list(mode_dir.glob("*.mdpeak"))
            for f in txt_files + mdpeak_files:
                all_files.append((f, ion_mode))
    
    # If no subdirs found, check the root directory (for backwards compatibility)
    if not all_files:
        txt_files = list(directory.glob("*.txt"))
        mdpeak_files = list(directory.glob("*.mdpeak"))
        for f in txt_files + mdpeak_files:
            # Try to infer from filename or default to positive
            fname_lower = f.name.lower()
            if "negative" in fname_lower or "_neg" in fname_lower:
                all_files.append((f, "negative"))
            else:
                all_files.append((f, "positive"))

    # Sort for deterministic ordering
    all_files.sort(key=lambda x: (x[1], x[0].name))

    return all_files


def load_and_process_chromatogram(
    chromatogram_path: Path,
    ion_mode: str,
    precursor_mass_accuracy_ppm: float,
    fragment_mass_accuracy_ppm: float,
) -> pl.DataFrame:
    """
    Load and process a single chromatogram file.

    Args:
        chromatogram_path: Path to the MSDIAL chromatogram file
        ion_mode: Ionization mode ('positive' or 'negative')
        precursor_mass_accuracy_ppm: Tolerance for precursor formula annotation
        fragment_mass_accuracy_ppm: Tolerance for fragment formula annotation

    Returns:
        Processed DataFrame with formula annotations and information scores
    """
    print(f"Loading chromatogram: {chromatogram_path.name} ({ion_mode} mode)")
    chromatogram = get_chromatogram(chromatogram_path)
    print(f"  Loaded {len(chromatogram)} peaks")

    # Filter to only peaks with MS/MS data
    chromatogram_with_msms = chromatogram.filter(pl.col("msms_m/z").is_not_null())
    print(f"  {len(chromatogram_with_msms)} peaks with MS/MS data")

    if chromatogram_with_msms.is_empty():
        print(f"  Warning: No MS/MS data found in {chromatogram_path.name}")
        return chromatogram_with_msms

    # Add source file name and ion_mode first so we have unique identifiers
    chromatogram_with_msms = chromatogram_with_msms.with_columns(
        pl.lit(chromatogram_path.name).alias("source_file"),
        pl.lit(ion_mode).alias("ion_mode"),
    )

    # Annotate with formulas
    print(f"  Annotating with formulas...")
    annotated = annotate_chromatogram_with_formulas(
        chromatogram_with_msms,
        precursor_mass_accuracy_ppm=precursor_mass_accuracy_ppm,
        fragment_mass_accuracy_ppm=fragment_mass_accuracy_ppm,
    )
    print(f"  Annotated {len(annotated)} formula candidates")

    # For peaks with multiple formulas, select the one with highest explained_intensity
    # Group by both source_file and Peak ID to ensure uniqueness across files
    print(f"  Selecting best formula per peak (by explained_intensity)...")
    annotated_best = (
        annotated.sort(
            ["source_file", "Peak ID", "explained_intensity"],
            descending=[False, False, True]
        )
        .group_by(["source_file", "Peak ID"])
        .first()
    )
    print(f"  Selected {len(annotated_best)} best formulas")

    # Calculate spectral information score for the chromatogram
    print(f"  Calculating spectral information scores...")
    annotated_with_info = annotated_best.with_columns(
        pl.struct(
            precursor_formula=pl.col("precursor_formula"),
            fragment_formulas=pl.col("cleaned_spectrum_formulas"),
        )
        .spectral_info.spectral_info_score(
            distance_metric="l2",
            ignore_hydrogens=True,
        )
        .alias("spectral_information_score")
    )

    return annotated_with_info


def load_spectral_library(library_path: Path) -> pl.LazyFrame:
    """
    Load the spectral library as a LazyFrame.

    Args:
        library_path: Path to the spectral library parquet file

    Returns:
        LazyFrame of the processed library
    """
    print(f"Loading spectral library: {library_path}")
    library = pl.scan_parquet(library_path)

    # Filter for spectra with formulas and good quality
    library_clean = library.filter(
        pl.col("precursor_formula_array").is_not_null(),
        pl.col("cleaned_fragment_formulas").is_not_null(),
    )

    # Add nominal mass for binning
    library_clean = library_clean.with_columns(
        pl.col("precursor_mz").round(0).cast(pl.Int64).alias("nominal_mass")
    )

    print(f"  Library loaded as LazyFrame with formula annotations")

    return library_clean


def match_chromatogram_to_library(
    chromatogram: pl.DataFrame,
    library_lf: pl.LazyFrame,
    precursor_tolerance_ppm: float,
    ms2_tolerance_ppm: float,
    match_threshold: float,
) -> pl.DataFrame:
    """
    Match chromatogram peaks against the spectral library.

    Args:
        chromatogram: Annotated chromatogram DataFrame with information scores
        library_lf: Processed library LazyFrame
        precursor_tolerance_ppm: Precursor mass tolerance in ppm
        ms2_tolerance_ppm: MS/MS fragment tolerance in ppm
        match_threshold: Minimum similarity score threshold

    Returns:
        DataFrame with all matches above threshold
    """
    print(f"Matching precursors (tolerance: {precursor_tolerance_ppm} ppm)...")

    # Convert chromatogram to lazy and add nominal mass
    chrom_lf = chromatogram.lazy().with_columns(
        pl.col("Precursor_mz_MSDIAL").round(0).cast(pl.Int64).alias("nominal_mass")
    )

    # Join on nominal mass first, then filter by precise ppm tolerance - all in lazy mode
    matches = (
        chrom_lf
        .join(
            library_lf,
            on="nominal_mass",
            how="inner",
        )
        .filter(
            # Apply precise precursor mass tolerance
            (
                pl.col("Precursor_mz_MSDIAL") / pl.col("precursor_mz") - 1.0
            ).abs() <= (precursor_tolerance_ppm * 1e-6)
        )
        .collect(engine="streaming")
    )

    print(f"  Found {len(matches)} precursor matches")

    if matches.is_empty():
        return matches

    # Compute spectral similarity
    print(f"Computing spectral similarity...")
    similarity_results = matches.with_columns(
        pl.struct(
            mz1=pl.col("cleaned_msms_mz"),
            intensities1=pl.col("cleaned_msms_intensity"),
            precursor_mz1=pl.col("Precursor_mz_MSDIAL"),
            mz2=pl.col("cleaned_normalized_mz"),
            intensities2=pl.col("cleaned_normalized_intensity"),
            precursor_mz2=pl.col("precursor_mz"),
        )
        .spectral_similarity.dotprod_similarity(
            ms2_tolerance_in_ppm=ms2_tolerance_ppm,
            clean_spectra_first=True,
            noise_threshold=0.001,
            ignore_precursor=True,
        )
        .alias("similarity_score")
    )

    # Filter by threshold
    high_confidence = similarity_results.filter(
        pl.col("similarity_score").is_not_null(),
        pl.col("similarity_score") >= match_threshold,
    )

    print(f"  {len(high_confidence)} matches above threshold {match_threshold}")

    return high_confidence


def select_best_match_per_inchikey(matches: pl.DataFrame) -> pl.DataFrame:
    """
    For each chromatogram peak and library base_inchikey, select the best match.

    When multiple spectra from the same base_inchikey match a peak,
    keep only the one with the highest similarity score.

    Args:
        matches: DataFrame with all matches

    Returns:
        DataFrame with best match per base_inchikey per peak
    """
    print("Selecting best match per InChIKey...")

    best_matches = (
        matches.sort(
            ["Peak ID", "source_file", "base_inchikey", "similarity_score"],
            descending=[False, False, False, True],
        )
        .group_by(["Peak ID", "source_file", "base_inchikey"])
        .first()
    )

    print(f"  Selected {len(best_matches)} unique matches")

    return best_matches


def prepare_output(matches: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare the output DataFrame with required columns.

    Args:
        matches: DataFrame with matched results

    Returns:
        Formatted output DataFrame
    """
    output = matches.select(
        [
            "source_file",
            "Peak ID",
            "RT (min)",
            "Precursor_mz_MSDIAL",
            "Height",
            "name",
            "inchikey",
            "base_inchikey",
            "smiles",
            "collision_energy_NCE",
            "collision_energy_ev",
            "collision_energy_mean",
            "similarity_score",
            "spectral_information_score",
            "precursor_formula_str",
            "precursor_errors_ppm",
            "explained_intensity",
        ]
    )

    return output


def process_single_chromatogram(
    chromatogram_path: Path,
    ion_mode: str,
    library_lf: pl.LazyFrame,
    args: argparse.Namespace,
) -> pl.DataFrame:
    """
    Process a single chromatogram file.

    Args:
        chromatogram_path: Path to the chromatogram file
        ion_mode: Ionization mode ('positive' or 'negative')
        library_lf: Processed library LazyFrame
        args: Command line arguments

    Returns:
        DataFrame with matches for this chromatogram
    """
    # Load, annotate chromatogram, calculate information scores
    chromatogram = load_and_process_chromatogram(
        chromatogram_path,
        ion_mode,
        precursor_mass_accuracy_ppm=args.precursor_mass_accuracy_ppm,
        fragment_mass_accuracy_ppm=args.fragment_mass_accuracy_ppm,
    )

    if chromatogram.is_empty():
        print(f"  Skipping {chromatogram_path.name} - no valid data")
        return pl.DataFrame()

    # Match to library
    matches = match_chromatogram_to_library(
        chromatogram,
        library_lf,
        precursor_tolerance_ppm=args.precursor_tolerance_ppm,
        ms2_tolerance_ppm=args.ms2_tolerance_ppm,
        match_threshold=args.match_threshold,
    )

    if matches.is_empty():
        print(f"  No matches found for {chromatogram_path.name}")
        return pl.DataFrame()

    # Select best match per InChIKey
    matches = select_best_match_per_inchikey(matches)

    # Prepare output
    output = prepare_output(matches)

    return output


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    library_path = Path(args.library)
    if not library_path.exists():
        print(f"Error: Library file not found: {library_path}", file=sys.stderr)
        sys.exit(1)

    # Determine input source
    if args.input_mzmls:
        # Process mzML files through MSDIAL first
        input_dir = Path(args.input_mzmls)
        if not input_dir.exists():
            print(f"Error: mzML directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)

        # Create MSDIAL output directory
        msdial_output_dir = output_dir / "msdial_output"
        
        # Run MSDIAL
        msdial_path = Path(args.msdial_path) if args.msdial_path else None
        chromatograms_dir = run_msdial_on_mzmls(
            input_dir=input_dir,
            output_dir=msdial_output_dir,
            msdial_path=msdial_path,
            threads=args.msdial_threads,
            minimum_peak_height=args.msdial_min_peak_height,
        )
        
        print("\n" + "=" * 80)
        print(f"MS-DIAL processing complete. Results in: {chromatograms_dir}")
        print("=" * 80)
    else:
        # Use pre-processed chromatograms
        chromatograms_dir = Path(args.chromatograms_dir)
        if not chromatograms_dir.exists():
            print(f"Error: Chromatograms directory not found: {chromatograms_dir}", file=sys.stderr)
            sys.exit(1)

    # Find chromatogram files
    chromatogram_files = find_chromatogram_files(chromatograms_dir)
    if not chromatogram_files:
        print(f"Error: No .txt or .mdpeak files found in {chromatograms_dir}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Plasma FPR Estimate - Spectral Library Search")
    print("=" * 80)
    print(f"Chromatograms directory: {chromatograms_dir}")
    print(f"Found {len(chromatogram_files)} chromatogram files")
    print(f"Match threshold: {args.match_threshold}")
    print(f"Precursor tolerance: {args.precursor_tolerance_ppm} ppm")
    print(f"MS/MS tolerance: {args.ms2_tolerance_ppm} ppm")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load library (once for all chromatograms)
    library_lf = load_spectral_library(library_path)

    # Process each chromatogram
    all_results: List[pl.DataFrame] = []

    for i, (chromatogram_path, ion_mode) in enumerate(chromatogram_files, 1):
        print(f"\n[{i}/{len(chromatogram_files)}] Processing: {chromatogram_path.name}")
        print("-" * 80)

        result = process_single_chromatogram(
            chromatogram_path, 
            ion_mode, 
            library_lf, 
            args
        )

        if not result.is_empty():
            all_results.append(result)

    if not all_results:
        print("\nNo matches found for any chromatogram. Exiting.")
        sys.exit(0)

    # Combine all results
    print("\n" + "=" * 80)
    print("Combining results...")
    combined_results = pl.concat(all_results, how="diagonal_relaxed")
    print(f"Total matches: {len(combined_results)}")

    # Write outputs
    parquet_path = output_dir / f"{args.output_prefix}.parquet"
    excel_path = output_dir / f"{args.output_prefix}.xlsx"

    print(f"\nWriting Parquet: {parquet_path}")
    combined_results.write_parquet(parquet_path)

    print(f"Writing Excel: {excel_path}")
    # For Excel, we need to ensure all columns are serializable
    excel_df = combined_results.with_columns(
        pl.col("collision_energy_NCE").cast(pl.Float64),
        pl.col("collision_energy_ev").cast(pl.Float64),
        pl.col("collision_energy_mean").cast(pl.Float64),
        pl.col("similarity_score").cast(pl.Float64),
        pl.col("spectral_information_score").cast(pl.Float64),
        pl.col("explained_intensity").cast(pl.Float64),
    )
    excel_df.write_excel(excel_path)

    print("\n" + "=" * 80)
    print("Done!")
    print(f"Results saved to:")
    print(f"  - Parquet: {parquet_path}")
    print(f"  - Excel: {excel_path}")
    print("=" * 80)

    # Print summary statistics
    print("\nSummary:")
    print(f"  Total input files: {len(chromatogram_files)}")
    print(f"  Total matches: {len(combined_results)}")
    print(f"  Unique peaks matched: {combined_results['Peak ID'].n_unique()}")
    print(f"  Unique compounds (base_inchikey): {combined_results['base_inchikey'].n_unique()}")

    similarity_stats = combined_results.select(
        pl.col("similarity_score").mean().alias("mean_similarity"),
        pl.col("similarity_score").min().alias("min_similarity"),
        pl.col("similarity_score").max().alias("max_similarity"),
        pl.col("spectral_information_score").mean().alias("mean_info_score"),
    )

    print(f"  Mean similarity score: {similarity_stats['mean_similarity'][0]:.3f}")
    print(f"  Similarity range: {similarity_stats['min_similarity'][0]:.3f} - {similarity_stats['max_similarity'][0]:.3f}")
    print(f"  Mean information score: {similarity_stats['mean_info_score'][0]:.3f}")


if __name__ == "__main__":
    main()
