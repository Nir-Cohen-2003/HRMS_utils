"""
Plasma FPR (False Positive Rate) Estimate Script

This script takes mzML files (or MSDIAL chromatograms), runs MSDIAL processing if needed,
matches chromatograms against a spectral library using precursor mass tolerance and
spectral similarity, and outputs the best matches for FPR estimation.

After matching, it performs statistical analysis of informativity vs false positive rate
by aligning features across samples (MS1 mass and RT within tolerance), selecting the best
formula per aligned group based on average explained intensity, and comparing pairs.

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

Alignment parameters (for statistical analysis):
    --ms1-tolerance-ppm 3.0 \
    --rt-tolerance-min 0.1 \
    --fpr-thresholds 0.75 0.85 0.95 (similarity thresholds; lower = more likely FP)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import polars as pl

from hrms_utils.formats.msdial import (
    MSDialRunnerConfig,
    annotate_chromatogram_with_formulas,
    get_chromatogram,
    run_msdial_lcmsdda,
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
        help="Directory containing pre-processed MSDIAL chromatogram - .mdpeak files",
    )

    parser.add_argument(
        "--library",
        type=str,
        required=True,
        help="Path to spectral library parquet file",
    )

    parser.add_argument(
        "--epa-list-dir",
        type=str,
        default="experiments/spectral_information/epa_list",
        help="Directory containing EPA list CSV files for library filtering",
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

    parser.add_argument(
        "--ms1-tolerance-ppm",
        type=float,
        default=3.0,
        help="MS1 mass tolerance in ppm for aligning features across samples",
    )

    parser.add_argument(
        "--rt-tolerance-min",
        type=float,
        default=0.1,
        help="RT tolerance in minutes for aligning features across samples",
    )

    parser.add_argument(
        "--fpr-thresholds",
        type=float,
        nargs="+",
        default=[0.75, 0.85, 0.95],
        help="Similarity thresholds for false positive classification",
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of chromatogram annotation and library matching (ignore cached parquet files)",
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
    Find all chromatogram files (.mdpeak) in the given directory.

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
            mdpeak_files = list(mode_dir.glob("*.mdpeak"))
            for f in mdpeak_files:
                all_files.append((f, ion_mode))

    # If no subdirs found, check the root directory (for backwards compatibility)
    if not all_files:
        mdpeak_files = list(directory.glob("*.mdpeak"))
        for f in mdpeak_files:
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
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load and process a single chromatogram file.

    Args:
        chromatogram_path: Path to the MSDIAL chromatogram file
        ion_mode: Ionization mode ('positive' or 'negative')
        precursor_mass_accuracy_ppm: Tolerance for precursor formula annotation
        fragment_mass_accuracy_ppm: Tolerance for fragment formula annotation

    Returns:
        Tuple of:
        - Processed DataFrame with best formula per peak and information scores
        - DataFrame with all formula candidates (for alignment analysis)
    """
    print(f"Loading chromatogram: {chromatogram_path.name} ({ion_mode} mode)")
    chromatogram = get_chromatogram(chromatogram_path)
    print(f"  Loaded {len(chromatogram)} peaks")

    # Filter to only peaks with MS/MS data
    chromatogram_with_msms = chromatogram.filter(pl.col("msms_m/z").is_not_null())
    print(f"  {len(chromatogram_with_msms)} peaks with MS/MS data")

    if chromatogram_with_msms.is_empty():
        print(f"  Warning: No MS/MS data found in {chromatogram_path.name}")
        return chromatogram_with_msms, chromatogram_with_msms

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

    # Store all formula candidates before selecting best per peak
    annotated_all = annotated.clone()

    # For peaks with multiple formulas, select the one with highest explained_intensity
    # Group by both source_file and Peak ID to ensure uniqueness across files
    print(f"  Selecting best formula per peak (by explained_intensity)...")
    annotated_best = (
        annotated.sort(
            ["source_file", "Peak ID", "explained_intensity"],
            descending=[False, False, True],
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

    return annotated_with_info, annotated_all


def load_epa_inchikeys(epa_list_dir: Path) -> pl.DataFrame:
    """
    Load EPA list CSV files and extract unique base InChIKeys.

    Args:
        epa_list_dir: Directory containing EPA list CSV files

    Returns:
        DataFrame with unique base_inchikey values
    """
    print(f"Loading EPA lists from: {epa_list_dir}")

    if not epa_list_dir.exists():
        raise FileNotFoundError(f"EPA list directory not found: {epa_list_dir}")

    csv_files = list(epa_list_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {epa_list_dir}")

    print(f"  Found {len(csv_files)} CSV files")

    all_inchikeys = []
    for csv_file in csv_files:
        try:
            df = pl.read_csv(csv_file)
            if "INCHIKEY" in df.columns:
                # Extract base_inchikey (first part before hyphen)
                base_keys = (
                    df.select(pl.col("INCHIKEY"))
                    .with_columns(
                        pl.col("INCHIKEY")
                        .str.split("-")
                        .list.get(0)
                        .alias("base_inchikey")
                    )
                    .filter(pl.col("base_inchikey").is_not_null())
                )
                all_inchikeys.append(base_keys)
        except Exception as e:
            print(f"  Warning: Could not read {csv_file.name}: {e}")

    if not all_inchikeys:
        raise ValueError("No valid InChIKeys found in EPA lists")

    # Concatenate all and get unique base_inchikeys
    combined = pl.union(all_inchikeys)
    unique_keys = combined.select("base_inchikey").unique()

    print(f"  Loaded {len(unique_keys)} unique base InChIKeys from EPA lists")
    return unique_keys


def load_spectral_library(
    library_path: Path, epa_list_dir: Optional[Path] = None
) -> pl.LazyFrame:
    """
    Load the spectral library as a LazyFrame.

    Args:
        library_path: Path to the spectral library parquet file
        epa_list_dir: Optional directory with EPA lists to filter library

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

    # Filter by EPA lists if provided
    if epa_list_dir is not None:
        epa_keys = load_epa_inchikeys(epa_list_dir)
        library_clean = library_clean.join(
            epa_keys.lazy(),
            left_on="base_inchikey",
            right_on="base_inchikey",
            how="inner",
        )
        print(f"  Filtered library to EPA-listed compounds")

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
        chrom_lf.join(
            library_lf,
            on="nominal_mass",
            how="inner",
        )
        .filter(
            # Apply precise precursor mass tolerance
            (pl.col("Precursor_mz_MSDIAL") / pl.col("precursor_mz") - 1.0).abs()
            <= (precursor_tolerance_ppm * 1e-6)
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
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Process a single chromatogram file.

    Args:
        chromatogram_path: Path to the chromatogram file
        ion_mode: Ionization mode ('positive' or 'negative')
        library_lf: Processed library LazyFrame
        args: Command line arguments

    Returns:
        Tuple of:
        - DataFrame with matches for existing output
        - DataFrame with all formula candidates for alignment analysis
    """
    # Load, annotate chromatogram, calculate information scores
    chromatogram, chromatogram_all_candidates = load_and_process_chromatogram(
        chromatogram_path,
        ion_mode,
        precursor_mass_accuracy_ppm=args.precursor_mass_accuracy_ppm,
        fragment_mass_accuracy_ppm=args.fragment_mass_accuracy_ppm,
    )

    if chromatogram.is_empty():
        print(f"  Skipping {chromatogram_path.name} - no valid data")
        return pl.DataFrame(), pl.DataFrame()

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
        return pl.DataFrame(), chromatogram_all_candidates

    # Select best match per InChIKey
    matches = select_best_match_per_inchikey(matches)

    # Prepare output
    output = prepare_output(matches)

    return output, chromatogram_all_candidates


def align_features_across_samples_v2(
    candidates_lf: pl.LazyFrame,
    ms1_tolerance_ppm: float,
    rt_tolerance_min: float,
) -> pl.LazyFrame:
    """
    Align features across samples based on MS1 mass and RT (not formula).

    Creates aligned pairs of peaks from different samples that match in mass and RT.
    Uses lazy evaluation with streaming to handle large datasets.

    Args:
        candidates_lf: LazyFrame with all formula candidates per peak
        ms1_tolerance_ppm: MS1 mass tolerance in ppm
        rt_tolerance_min: RT tolerance in minutes

    Returns:
        LazyFrame with aligned pairs (each row is a pair of peaks with their formula candidates)
    """
    print("Aligning features across samples (MS1 + RT only)...")
    print("  Deduplicating to unique peaks first...")

    # First deduplicate to unique peaks (source_file + Peak ID) to avoid N^2 explosion
    # Each peak may have multiple formula candidates, but we only need to align the peaks
    unique_peaks = candidates_lf.select([
        "source_file",
        "Peak ID",
        "RT (min)",
        "Precursor_mz_MSDIAL",
        "precursor_formula_str",
        "explained_intensity",
        "cleaned_spectrum_formulas",
    ]).unique(subset=["source_file", "Peak ID"], keep="first")

    candidates_with_mass = unique_peaks.with_columns(
        pl.col("Precursor_mz_MSDIAL").round(0).cast(pl.Int64).alias("nominal_mass")
    )

    aligned_pairs = candidates_with_mass.join(
        candidates_with_mass,
        on="nominal_mass",
        how="inner",
        suffix="_right",
    ).filter(
        pl.col("source_file") != pl.col("source_file_right"),
        (
            (pl.col("Precursor_mz_MSDIAL") / pl.col("Precursor_mz_MSDIAL_right")) - 1.0
        ).abs()
        <= ms1_tolerance_ppm * 1e-6,
        (pl.col("RT (min)") - pl.col("RT (min)_right")).abs() <= rt_tolerance_min,
    )

    return aligned_pairs


def select_best_formula_per_feature(
    aligned_pairs_lf: pl.LazyFrame,
    candidates_lf: pl.LazyFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    For each aligned feature (connected component of aligned peaks), select the best formula
    based on SUM of explained intensity across all spectra in the feature.

    Args:
        aligned_pairs_lf: LazyFrame with aligned pairs (each row has both peaks in a pair)
        candidates_lf: LazyFrame with ALL formula candidates (to get all formulas for aligned peaks)

    Returns:
        Tuple of:
        - DataFrame with feature assignments for each peak
        - DataFrame with best formula per feature
    """
    print("Selecting best formula per aligned feature...")

    # Collect aligned pairs to build connected components
    aligned_pairs = aligned_pairs_lf.select([
        "source_file", "Peak ID",
        "source_file_right", "Peak ID_right"
    ]).collect(engine="streaming")

    if aligned_pairs.is_empty():
        print("  No aligned pairs found")
        return pl.DataFrame(), pl.DataFrame()

    print(f"  Building connected components from {len(aligned_pairs)} aligned pairs...")

    # Build graph and find connected components (features)
    from collections import defaultdict

    # Create mapping from peak to index
    left_peaks = aligned_pairs.select(["source_file", "Peak ID"]).to_numpy()
    right_peaks = aligned_pairs.select(["source_file_right", "Peak ID_right"]).to_numpy()

    all_peaks = set()
    peak_to_idx = {}
    idx = 0

    for sf, pid in left_peaks:
        peak_key = (sf, pid)
        if peak_key not in peak_to_idx:
            peak_to_idx[peak_key] = idx
            all_peaks.add(peak_key)
            idx += 1

    for sf, pid in right_peaks:
        peak_key = (sf, pid)
        if peak_key not in peak_to_idx:
            peak_to_idx[peak_key] = idx
            all_peaks.add(peak_key)
            idx += 1

    # Build adjacency list and find connected components using Union-Find
    n = len(peak_to_idx)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union all connected peaks
    for sf, pid, sf_r, pid_r in aligned_pairs.select([
        "source_file", "Peak ID", "source_file_right", "Peak ID_right"
    ]).to_numpy():
        union(peak_to_idx[(sf, pid)], peak_to_idx[(sf_r, pid_r)])

    # Assign feature IDs
    feature_id = {}
    next_feature_id = 0
    for peak_key in all_peaks:
        root = find(peak_to_idx[peak_key])
        if root not in feature_id:
            feature_id[root] = next_feature_id
            next_feature_id += 1
        feature_id[peak_key] = feature_id[root]

    print(f"  Found {next_feature_id} unique features")

    # Create feature assignment DataFrame
    feature_assignments = pl.DataFrame({
        "source_file": [pk[0] for pk in all_peaks],
        "Peak ID": [pk[1] for pk in all_peaks],
        "feature_id": [feature_id[pk] for pk in all_peaks],
    })

    # Join with all candidates to get formulas for each feature
    candidates_df = candidates_lf.select([
        "source_file", "Peak ID",
        "precursor_formula_str", "explained_intensity", "cleaned_spectrum_formulas"
    ]).collect(engine="streaming")

    feature_formulas = feature_assignments.join(
        candidates_df,
        on=["source_file", "Peak ID"],
        how="inner"
    )

    # Sum explained intensity per feature per formula
    formula_sums = feature_formulas.group_by(["feature_id", "precursor_formula_str"]).agg(
        pl.col("explained_intensity").sum().alias("total_explained"),
        pl.col("cleaned_spectrum_formulas").first().alias("fragment_formulas"),
    )

    # Select best formula per feature (max total explained intensity)
    best_formula_per_feature = formula_sums.sort(
        ["feature_id", "total_explained"],
        descending=[False, True]
    ).group_by("feature_id").first()

    print(f"  Selected best formula for {len(best_formula_per_feature)} features")

    return feature_assignments, best_formula_per_feature


def compute_fp_counts_per_threshold(
    feature_assignments: pl.DataFrame,
    best_formula_per_feature: pl.DataFrame,
    candidates_lf: pl.LazyFrame,
    library_lf: pl.LazyFrame,
    args: argparse.Namespace,
    fpr_thresholds: list[float],
) -> pl.DataFrame:
    """
    Compute spectral information scores and false positive counts per threshold.

    For each spectrum in features with the best formula:
    1. Compute spectral_information_score
    2. Match to library
    3. For each threshold, count matches above threshold (after selecting best per InChIKey)

    Args:
        feature_assignments: DataFrame mapping peaks to features
        best_formula_per_feature: DataFrame with best formula per feature
        candidates_lf: LazyFrame with all formula candidates
        library_lf: Spectral library LazyFrame
        args: Command line arguments
        fpr_thresholds: List of similarity thresholds for FP classification

    Returns:
        DataFrame with spectral_information_score and fp_count_{threshold} per spectrum
    """
    print("Computing spectral information scores and FP counts per threshold...")

    # Get spectra with best formulas - need array-type formula columns for spectral_info_score
    # Also need MS2 spectrum columns for library matching
    candidates_df = candidates_lf.select([
        "source_file", "Peak ID", "Precursor_mz_MSDIAL",
        "precursor_formula_str", "precursor_formula", "explained_intensity",
        "cleaned_spectrum_formulas", "cleaned_msms_mz", "cleaned_msms_intensity"
    ]).collect(engine="streaming")

    # Join feature assignments with best formula info
    spectra_with_formulas = feature_assignments.join(
        best_formula_per_feature.select(["feature_id", "precursor_formula_str", "fragment_formulas"]),
        on="feature_id",
        how="inner"
    )

    # Get actual MS2 data for these spectra (need precursor mz for matching)
    spectra_data = spectra_with_formulas.join(
        candidates_df,
        on=["source_file", "Peak ID", "precursor_formula_str"],
        how="inner"
    )

    print(f"  Processing {len(spectra_data)} spectra with best formulas...")

    # Compute spectral information scores using array-type formulas
    spectra_with_info = spectra_data.with_columns(
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

    # Prepare for library matching
    spectra_lf = spectra_with_info.lazy().with_columns(
        pl.col("Precursor_mz_MSDIAL").round(0).cast(pl.Int64).alias("nominal_mass")
    )

    # Match to library
    print(f"  Matching spectra to library...")
    matches = spectra_lf.join(library_lf, on="nominal_mass", how="inner").filter(
        (pl.col("Precursor_mz_MSDIAL") / pl.col("precursor_mz") - 1.0).abs()
        <= (args.precursor_tolerance_ppm * 1e-6)
    )

    # Compute similarity scores
    matches = matches.with_columns(
        pl.struct(
            mz1=pl.col("cleaned_msms_mz"),
            intensities1=pl.col("cleaned_msms_intensity"),
            precursor_mz1=pl.col("Precursor_mz_MSDIAL"),
            mz2=pl.col("cleaned_normalized_mz"),
            intensities2=pl.col("cleaned_normalized_intensity"),
            precursor_mz2=pl.col("precursor_mz"),
        )
        .spectral_similarity.dotprod_similarity(
            ms2_tolerance_in_ppm=args.ms2_tolerance_ppm,
            clean_spectra_first=True,
            noise_threshold=0.001,
            ignore_precursor=True,
        )
        .alias("similarity_score")
    )

    matches_df = matches.collect(engine="streaming")
    print(f"  Found {len(matches_df)} total library matches")

    if matches_df.is_empty():
        # Return empty DataFrame with correct columns
        result = spectra_with_info.select([
            "source_file", "Peak ID", "feature_id", "precursor_formula_str",
            "spectral_information_score"
        ])
        for thresh in fpr_thresholds:
            thresh_str = str(thresh).replace(".", "_")
            result = result.with_columns(pl.lit(0).alias(f"fp_count_{thresh_str}"))
        return result

    # For each threshold, count FP matches per spectrum
    # First, select best match per InChIKey per spectrum, then count remaining
    result_spectra = spectra_with_info.select([
        "source_file", "Peak ID", "feature_id", "precursor_formula_str",
        "spectral_information_score"
    ])

    for thresh in fpr_thresholds:
        thresh_str = str(thresh).replace(".", "_")

        # Filter matches above threshold
        matches_above = matches_df.filter(pl.col("similarity_score") >= thresh)

        if matches_above.is_empty():
            result_spectra = result_spectra.with_columns(
                pl.lit(0).alias(f"fp_count_{thresh_str}")
            )
            continue

        # Select best match per InChIKey per spectrum
        best_per_inchikey = (
            matches_above.sort(
                ["source_file", "Peak ID", "base_inchikey", "similarity_score"],
                descending=[False, False, False, True]
            )
            .group_by(["source_file", "Peak ID", "base_inchikey"])
            .first()
        )

        # Count matches per spectrum (these are the FP counts)
        fp_counts = best_per_inchikey.group_by(["source_file", "Peak ID"]).agg(
            pl.len().alias(f"fp_count_{thresh_str}")
        )

        # Join back to result
        result_spectra = result_spectra.join(
            fp_counts,
            on=["source_file", "Peak ID"],
            how="left"
        ).with_columns(
            pl.col(f"fp_count_{thresh_str}").fill_null(0)
        )

    print(f"  Computed FP counts for {len(result_spectra)} spectra")
    return result_spectra


def compute_pair_comparisons_v2(
    spectra_df: pl.DataFrame,
    feature_assignments: pl.DataFrame,
    fpr_thresholds: list[float],
) -> pl.DataFrame:
    """
    Compare informativity vs false positive rate across all pairs within each feature.

    For each pair of spectra within the same feature:
    - Skip if information scores are equal
    - Determine which has higher informativity
    - Compare FP counts (lower = better)
    - Categorize as tie, informative_wins, or less_informative_wins

    Args:
        spectra_df: DataFrame with spectra info scores and FP counts per threshold
        feature_assignments: DataFrame mapping peaks to features
        fpr_thresholds: List of similarity thresholds for FP classification

    Returns:
        DataFrame with pairwise comparisons
    """
    print("Computing pair comparisons...")

    # Join spectra with feature assignments
    spectra_with_features = spectra_df.join(
        feature_assignments.select(["source_file", "Peak ID", "feature_id"]),
        on=["source_file", "Peak ID"],
        how="inner"
    )

    # Build all pairs within each feature using self-join
    pairs = spectra_with_features.join(
        spectra_with_features,
        on="feature_id",
        suffix="_right"
    ).filter(
        # Avoid self-pairs and ensure consistent ordering
        (pl.col("source_file") < pl.col("source_file_right"))
        | (
            (pl.col("source_file") == pl.col("source_file_right"))
            & (pl.col("Peak ID") < pl.col("Peak ID_right"))
        )
    )

    print(f"  Generated {len(pairs)} pairs across all features")

    # Filter out pairs with equal information scores
    pairs = pairs.filter(pl.col("spectral_information_score") != pl.col("spectral_information_score_right"))
    print(f"  After filtering ties: {len(pairs)} pairs")

    # For each threshold, determine outcome
    for thresh in fpr_thresholds:
        thresh_str = str(thresh).replace(".", "_")
        fp_col = f"fp_count_{thresh_str}"
        fp_col_right = f"{fp_col}_right"

        pairs = pairs.with_columns(
            pl.when(pl.col(fp_col) == pl.col(fp_col_right))
            .then(pl.lit("tie"))
            .when(
                (pl.col("spectral_information_score") > pl.col("spectral_information_score_right"))
                & (pl.col(fp_col) < pl.col(fp_col_right))
            )
            .then(pl.lit("informative_wins"))
            .when(
                (pl.col("spectral_information_score") < pl.col("spectral_information_score_right"))
                & (pl.col(fp_col) > pl.col(fp_col_right))
            )
            .then(pl.lit("informative_wins"))
            .otherwise(pl.lit("less_informative_wins"))
            .alias(f"outcome_{thresh_str}")
        )

    return pairs


def compute_vs_most_informative_v2(
    spectra_df: pl.DataFrame,
    feature_assignments: pl.DataFrame,
    fpr_thresholds: list[float],
) -> pl.DataFrame:
    """
    Compare each spectrum against the most informative spectrum in its feature.

    For each feature, finds the spectrum with highest information score,
    then compares it to all other spectra in that feature.

    Args:
        spectra_df: DataFrame with spectra info scores and FP counts per threshold
        feature_assignments: DataFrame mapping peaks to features
        fpr_thresholds: List of similarity thresholds for FP classification

    Returns:
        DataFrame with comparisons to most informative spectrum
    """
    print("Computing comparisons vs most informative spectrum...")

    # Join spectra with feature assignments
    spectra_with_features = spectra_df.join(
        feature_assignments.select(["source_file", "Peak ID", "feature_id"]),
        on=["source_file", "Peak ID"],
        how="inner"
    )

    # Find most informative spectrum per feature (pick first if tie)
    # Need to include all columns for the comparison
    most_informative = (
        spectra_with_features.sort(
            ["feature_id", "spectral_information_score"],
            descending=[False, True]
        )
        .group_by("feature_id")
        .first()
    )

    # Join most informative with all other spectra in the same feature
    # Rename columns from right side to avoid naming conflicts
    spectra_renamed = spectra_with_features.rename({
        col: f"{col}_other" for col in spectra_with_features.columns if col != "feature_id"
    })
    
    comparisons = most_informative.join(
        spectra_renamed,
        on="feature_id",
        how="inner"
    ).filter(
        # Exclude self-comparisons
        (pl.col("source_file") != pl.col("source_file_other"))
        | (pl.col("Peak ID") != pl.col("Peak ID_other"))
    )

    print(f"  Generated {len(comparisons)} comparisons vs most informative")

    # For each threshold, determine outcome
    for thresh in fpr_thresholds:
        thresh_str = str(thresh).replace(".", "_")
        fp_col = f"fp_count_{thresh_str}"
        fp_col_other = f"{fp_col}_other"

        comparisons = comparisons.with_columns(
            pl.when(pl.col(fp_col) == pl.col(fp_col_other))
            .then(pl.lit("tie"))
            .when(pl.col(fp_col) < pl.col(fp_col_other))
            .then(pl.lit("most_informative_wins"))
            .otherwise(pl.lit("other_wins"))
            .alias(f"vs_best_outcome_{thresh_str}")
        )

    return comparisons


def write_statistical_analysis_v2(
    pairs_df: pl.DataFrame,
    vs_best_df: pl.DataFrame,
    output_dir: Path,
    output_prefix: str,
    fpr_thresholds: list[float],
    ms1_tolerance_ppm: float,
    rt_tolerance_min: float,
    n_features: int,
    n_spectra: int,
) -> None:
    """
    Write statistical analysis results to a text file.

    Args:
        pairs_df: DataFrame with pairwise comparisons
        vs_best_df: DataFrame with comparisons vs most informative
        output_dir: Output directory
        output_prefix: Prefix for output files
        fpr_thresholds: List of similarity thresholds used
        ms1_tolerance_ppm: MS1 tolerance used for alignment
        rt_tolerance_min: RT tolerance used for alignment
        n_features: Number of aligned features
        n_spectra: Number of spectra analyzed
    """
    total_pairs = len(pairs_df)

    lines = [
        "Informativity vs False Positive Rate Analysis",
        "=" * 50,
        "",
        "Alignment parameters:",
        f"  MS1 tolerance: {ms1_tolerance_ppm} ppm",
        f"  RT tolerance: {rt_tolerance_min} min",
        f"  Formula selection: best per feature by SUM of explained intensity",
        "",
        f"False positive classification:",
        f"  FP count = number of library matches with similarity >= threshold",
        f"  (after selecting best match per InChIKey)",
        "",
        f"Features with aligned spectra: {n_features}",
        f"Total spectra analyzed: {n_spectra}",
        f"Total pairs compared: {total_pairs}",
        "",
        "Pairwise Comparisons (all pairs within features):",
        "-" * 50,
        "",
    ]

    for thresh in fpr_thresholds:
        thresh_str = str(thresh).replace(".", "_")
        ties = (pairs_df[f"outcome_{thresh_str}"] == "tie").sum()
        inf_wins = (pairs_df[f"outcome_{thresh_str}"] == "informative_wins").sum()
        less_wins = (pairs_df[f"outcome_{thresh_str}"] == "less_informative_wins").sum()

        tie_pct = ties / total_pairs * 100 if total_pairs > 0 else 0
        inf_pct = inf_wins / total_pairs * 100 if total_pairs > 0 else 0
        less_pct = less_wins / total_pairs * 100 if total_pairs > 0 else 0

        lines.extend(
            [
                f"Threshold (similarity >= {thresh}):",
                f"  Ties (equal FP count): {ties} ({tie_pct:.1f}%)",
                f"  More informative wins (fewer FPs): {inf_wins} ({inf_pct:.1f}%)",
                f"  Less informative wins: {less_wins} ({less_pct:.1f}%)",
                "",
            ]
        )

    # Add comparisons vs most informative
    if not vs_best_df.is_empty():
        total_vs_best = len(vs_best_df)
        lines.extend(
            [
                "",
                "Comparisons vs Most Informative Spectrum per Feature:",
                "-" * 50,
                f"Total comparisons: {total_vs_best}",
                "",
            ]
        )

        for thresh in fpr_thresholds:
            thresh_str = str(thresh).replace(".", "_")
            ties = (vs_best_df[f"vs_best_outcome_{thresh_str}"] == "tie").sum()
            best_wins = (vs_best_df[f"vs_best_outcome_{thresh_str}"] == "most_informative_wins").sum()
            other_wins = (vs_best_df[f"vs_best_outcome_{thresh_str}"] == "other_wins").sum()

            tie_pct = ties / total_vs_best * 100 if total_vs_best > 0 else 0
            best_pct = best_wins / total_vs_best * 100 if total_vs_best > 0 else 0
            other_pct = other_wins / total_vs_best * 100 if total_vs_best > 0 else 0

            lines.extend(
                [
                    f"Threshold (similarity >= {thresh}):",
                    f"  Ties (equal FP count): {ties} ({tie_pct:.1f}%)",
                    f"  Most informative wins (fewer FPs): {best_wins} ({best_pct:.1f}%)",
                    f"  Other spectrum wins: {other_wins} ({other_pct:.1f}%)",
                    "",
                ]
            )

    output_path = output_dir / f"{output_prefix}_informativity_fpr_analysis.txt"
    output_path.write_text("\n".join(lines))
    print(f"  Written to: {output_path}")


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
            print(
                f"Error: Chromatograms directory not found: {chromatograms_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Find chromatogram files
    chromatogram_files = find_chromatogram_files(chromatograms_dir)
    if not chromatogram_files:
        print(
            f"Error: No .mdpeak files found in {chromatograms_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load library (once for all chromatograms)
    epa_list_dir = Path(args.epa_list_dir) if args.epa_list_dir else None
    library_lf = load_spectral_library(library_path, epa_list_dir)

    print("\n" + "=" * 80)
    print("Plasma FPR Estimate - Spectral Library Search")
    print("=" * 80)
    print(f"Chromatograms directory: {chromatograms_dir}")
    print(f"Found {len(chromatogram_files)} chromatogram files")
    print(f"Library filtering: {'EPA list' if epa_list_dir is not None else 'None (full library)'}")
    print(f"Match threshold: {args.match_threshold}")
    print(f"Precursor tolerance: {args.precursor_tolerance_ppm} ppm")
    print(f"MS/MS tolerance: {args.ms2_tolerance_ppm} ppm")
    print(f"Output directory: {output_dir}")
    output_prefix_with_filter = f"{args.output_prefix}_epa_filtered" if epa_list_dir is not None else args.output_prefix
    print(f"Output files: {output_prefix_with_filter}.parquet/.xlsx")
    print("=" * 80)

    # Setup cache directory for processed chromatogram data
    cache_dir = output_dir / "chromatogram_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine cache suffix based on EPA filtering
    cache_suffix = "_epa_filtered" if epa_list_dir is not None else "_unfiltered"

    if args.recompute:
        print("  --recompute flag set: ignoring cached files and recomputing all")

    # Process each chromatogram
    all_results: List[pl.DataFrame] = []
    all_candidates: List[pl.DataFrame] = []

    for i, (chromatogram_path, ion_mode) in enumerate(chromatogram_files, 1):
        print(f"\n[{i}/{len(chromatogram_files)}] Processing: {chromatogram_path.name}")
        print("-" * 80)

        # Check for cached parquet files
        base_name = chromatogram_path.stem
        cached_candidates_path = cache_dir / f"{base_name}_candidates{cache_suffix}.parquet"
        cached_results_path = cache_dir / f"{base_name}_results{cache_suffix}.parquet"

        if (
            not args.recompute
            and cached_candidates_path.exists()
            and cached_results_path.exists()
        ):
            print(f"  Loading cached data...")
            try:
                candidates = pl.read_parquet(cached_candidates_path)
                result = pl.read_parquet(cached_results_path)

                if len(result) == 0:
                    print(
                        f"  Loaded {len(candidates)} candidates, 0 matches from cache (no matches found in previous run)"
                    )
                else:
                    print(
                        f"  Loaded {len(candidates)} candidates, {len(result)} matches from cache"
                    )

                if not result.is_empty():
                    all_results.append(result)
                if not candidates.is_empty():
                    all_candidates.append(candidates)
                continue
            except Exception as e:
                print(f"  Warning: Failed to load cache, recomputing: {e}")

        # Process the chromatogram
        result, candidates = process_single_chromatogram(
            chromatogram_path, ion_mode, library_lf, args
        )

        # Cache the results (always write both files, even if empty)
        if not candidates.is_empty():
            print(f"  Caching {len(candidates)} candidates...")
            candidates.write_parquet(cached_candidates_path)
            all_candidates.append(candidates)

            # Always write results file, even if empty (signals that processing is complete)
            print(f"  Caching {len(result)} matches...")
            result.write_parquet(cached_results_path)

            if not result.is_empty():
                all_results.append(result)

    if not all_results:
        print("\nNo matches found for any chromatogram. Exiting.")
        sys.exit(0)

    # Combine all results
    print("\n" + "=" * 80)
    print("Combining results...")
    combined_results = pl.union(all_results, how="diagonal_relaxed")
    print(f"Total matches: {len(combined_results)}")

    # Write outputs
    # Include EPA filtering indicator in output filenames
    parquet_path = output_dir / f"{output_prefix_with_filter}.parquet"
    excel_path = output_dir / f"{output_prefix_with_filter}.xlsx"

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
    if epa_list_dir is not None:
        print(f"\nNote: Results are filtered by EPA list ({epa_list_dir})")
    print("=" * 80)

    # Print summary statistics
    print("\nSummary:")
    print(f"  Total input files: {len(chromatogram_files)}")
    print(f"  Library filtering: {'EPA list' if epa_list_dir is not None else 'None (full library)'}")
    print(f"  Total matches: {len(combined_results)}")
    print(f"  Unique peaks matched: {combined_results['Peak ID'].n_unique()}")
    print(
        f"  Unique compounds (base_inchikey): {combined_results['base_inchikey'].n_unique()}"
    )

    similarity_stats = combined_results.select(
        pl.col("similarity_score").mean().alias("mean_similarity"),
        pl.col("similarity_score").min().alias("min_similarity"),
        pl.col("similarity_score").max().alias("max_similarity"),
        pl.col("spectral_information_score").mean().alias("mean_info_score"),
    )

    print(f"  Mean similarity score: {similarity_stats['mean_similarity'][0]:.3f}")
    print(
        f"  Similarity range: {similarity_stats['min_similarity'][0]:.3f} - {similarity_stats['max_similarity'][0]:.3f}"
    )
    print(f"  Mean information score: {similarity_stats['mean_info_score'][0]:.3f}")

    print("\n" + "=" * 80)
    print("Statistical Analysis: Informativity vs False Positive Rate")
    print("=" * 80)

    if not all_candidates:
        print("  No candidate data available for statistical analysis")
    else:
        # Combine all candidates and write to temporary parquet for streaming
        candidates_temp_path = output_dir / "candidates_temp.parquet"
        print(f"  Combining {len(all_candidates)} candidate batches...")

        # Concatenate all candidates at once (more efficient than incremental)
        combined_candidates = pl.concat(all_candidates, how="diagonal_relaxed")
        print(
            f"  Writing {len(combined_candidates)} total candidates to temporary file..."
        )
        combined_candidates.write_parquet(candidates_temp_path)

        # Scan as lazyframe for streaming processing
        candidates_lf = pl.scan_parquet(candidates_temp_path)

        # Get count for reporting (using streaming collect)
        candidate_count = (
            candidates_lf.select(pl.len()).collect(engine="streaming").item()
        )
        print(f"  Total formula candidates: {candidate_count}")

        aligned_pairs_lf = align_features_across_samples_v2(
            candidates_lf,
            ms1_tolerance_ppm=args.ms1_tolerance_ppm,
            rt_tolerance_min=args.rt_tolerance_min,
        )

        # Check if we have any aligned pairs
        aligned_count = (
            aligned_pairs_lf.select(pl.len()).collect(engine="streaming").item()
        )

        if aligned_count > 0:
            print(f"  Found {aligned_count} aligned pairs")

            # Select best formula per feature (using SUM of explained intensity)
            feature_assignments, best_formula_per_feature = select_best_formula_per_feature(
                aligned_pairs_lf, candidates_lf
            )

            if len(feature_assignments) > 0 and len(best_formula_per_feature) > 0:
                print(f"  Processing {len(best_formula_per_feature)} features with best formulas")

                # Compute spectral info scores and FP counts per threshold
                spectra_df = compute_fp_counts_per_threshold(
                    feature_assignments,
                    best_formula_per_feature,
                    candidates_lf,
                    library_lf,
                    args,
                    args.fpr_thresholds,
                )

                if len(spectra_df) > 0:
                    print(f"  Computing comparisons for {len(spectra_df)} unique spectra")

                    # Compare all pairs within features
                    pairs_df = compute_pair_comparisons_v2(
                        spectra_df, feature_assignments, args.fpr_thresholds
                    )

                    # Compare vs most informative spectrum per feature
                    vs_best_df = compute_vs_most_informative_v2(
                        spectra_df, feature_assignments, args.fpr_thresholds
                    )

                    # Write statistical analysis results
                    n_features = spectra_df["feature_id"].n_unique()
                    write_statistical_analysis_v2(
                        pairs_df,
                        vs_best_df,
                        output_dir,
                        args.output_prefix,
                        args.fpr_thresholds,
                        args.ms1_tolerance_ppm,
                        args.rt_tolerance_min,
                        n_features=n_features,
                        n_spectra=len(spectra_df),
                    )
                else:
                    print("  No spectra found with library matches")
            else:
                print("  No features with best formulas found")
        else:
            print("  No aligned feature pairs found - skipping statistical analysis")

        # Clean up temporary file
        candidates_temp_path.unlink(missing_ok=True)
        print(f"  Cleaned up temporary file: {candidates_temp_path}")

    print("\n" + "=" * 80)
    print("Cache Information:")
    print(f"  Processed chromatogram data cached in: {cache_dir}")
    print(f"  Use --recompute to force recomputation and ignore cache")
    print("=" * 80)


if __name__ == "__main__":
    main()
