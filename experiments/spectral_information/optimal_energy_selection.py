"""
Find optimal collision energy points to maximize compound coverage.

For each molecule, we find the range of collision energies where at least one spectrum
achieves informativity >= threshold × max_informativity. Then we find the optimal
energy points that fall within the most molecules' valid ranges.

Algorithm:
1. Per molecule: compute min/max energy where informativity_fraction >= threshold
2. Find optimal energy points that cover the most molecule ranges (greedy)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class OptimalEnergyConfig:
    """
    Configuration for optimal energy selection analysis.

    Attributes:
        parquet_path: Path to input parquet file with spectral data.
        output_dir: Directory for output parquet files. If None, no files written.
        threshold: Minimum informativity fraction (default 2/3).
        max_combinations: Maximum number of energy points to select (default 7).
        collision_energy_column: Column name for collision energy values.
        molecule_id_column: Column name for molecule identifier.
        info_column: Column name for informativity score.
        ion_mode_column: Column name for ion mode (P/N).
        energy_tolerance: Tolerance for treating energies as equivalent (default 1.0 eV).
        collision_energy_nce_column: Column name for NCE collision energy.
        precursor_mz_column: Column name for precursor m/z.
    """

    parquet_path: Path
    output_dir: Optional[Path] = None
    threshold: float = 2.0 / 3.0
    max_combinations: int = 7
    collision_energy_column: str = "collision_energy_ev"
    molecule_id_column: str = "base_inchikey"
    info_column: str = "spectral_information_score"
    ion_mode_column: str = "ion_mode"
    energy_tolerance: float = 1.0
    collision_energy_nce_column: str = "collision_energy_NCE"
    precursor_mz_column: str = "precursor_mz"


@dataclass
class EnergyCombinationResult:
    """Result for a single energy combination."""

    n_energies: int
    energies: List[float]
    n_compounds_covered: int
    total_compounds: int
    coverage_fraction: float


@dataclass
class MoleculeEnergyRange:
    """Energy range where a molecule has sufficient informativity."""

    molecule_id: str
    min_energy: float
    max_energy: float


def interpolate_threshold_crossing(
    energies: np.ndarray, fractions: np.ndarray, threshold: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find the min and max energy where informativity crosses threshold via linear interpolation.

    Args:
        energies: Sorted array of collision energies.
        fractions: Corresponding informativity fractions.
        threshold: The threshold value to find crossings.

    Returns:
        Tuple of (min_energy, max_energy) where the curve crosses threshold.
        Returns (None, None) if the curve never crosses threshold.
    """
    assert len(energies) == len(fractions), "Energies and fractions must have same length"

    if len(energies) == 0:
        return None, None

    above = fractions >= threshold
    if not np.any(above):
        return None, None

    if np.all(above):
        return float(energies[0]), float(energies[-1])

    min_energy: Optional[float] = None
    max_energy: Optional[float] = None

    for i in range(len(energies) - 1):
        e1, e2 = energies[i], energies[i + 1]
        f1, f2 = fractions[i], fractions[i + 1]

        if f1 >= threshold and min_energy is None:
            min_energy = float(e1)
        elif f1 < threshold and f2 >= threshold:
            crossing = e1 + (e2 - e1) * (threshold - f1) / (f2 - f1)
            if min_energy is None:
                min_energy = float(crossing)

        if f2 >= threshold:
            max_energy = float(e2)
        elif f1 >= threshold and f2 < threshold:
            crossing = e1 + (e2 - e1) * (threshold - f1) / (f2 - f1)
            max_energy = float(crossing)

    return min_energy, max_energy


def compute_molecule_energy_ranges(
    df: pl.DataFrame,
    config: OptimalEnergyConfig,
) -> Tuple[List[MoleculeEnergyRange], int, int]:
    """
    For each molecule, compute the min/max energy where informativity >= threshold
    using linear interpolation between data points.

    Molecules with any null collision energy are excluded.

    Args:
        df: DataFrame with spectral data for one ion mode.
        config: Configuration with threshold and column names.

    Returns:
        Tuple of:
        - List of MoleculeEnergyRange objects for molecules with valid ranges
        - Count of molecules excluded due to null collision energy
        - Count of molecules with valid energy but no range meeting threshold
    """
    required_cols = {
        config.molecule_id_column,
        config.info_column,
        config.collision_energy_column,
    }
    missing = required_cols.difference(set(df.columns))
    assert not missing, f"DataFrame missing required columns: {sorted(missing)}"

    # If eV is null but NCE and precursor_mz are available, convert NCE to eV
    ev_col = config.collision_energy_column
    nce_col = config.collision_energy_nce_column
    mz_col = config.precursor_mz_column

    if nce_col in df.columns and mz_col in df.columns:
        df = df.with_columns(
            pl.when(pl.col(ev_col).is_null() & pl.col(nce_col).is_not_null() & pl.col(mz_col).is_not_null())
            .then(pl.col(nce_col) * pl.col(mz_col) / 500.0)
            .otherwise(pl.col(ev_col))
            .alias(ev_col)
        )

    df_clean = df.filter(pl.col(ev_col).is_not_null())

    molecules_with_null_energy = df.select(
        pl.col(config.molecule_id_column).n_unique()
    ).item() - df_clean.select(pl.col(config.molecule_id_column).n_unique()).item()

    df_with_fraction = df_clean.with_columns(
        max_info_per_mol=pl.col(config.info_column)
        .max()
        .over(config.molecule_id_column)
    ).with_columns(
        informativity_fraction=pl.col(config.info_column) / pl.col("max_info_per_mol")
    )

    per_mol = df_with_fraction.group_by(config.molecule_id_column).agg(
        energies=pl.col(config.collision_energy_column).sort(),
        fractions=pl.col("informativity_fraction").sort_by(
            pl.col(config.collision_energy_column)
        ),
    )

    ranges: List[MoleculeEnergyRange] = []
    molecules_no_valid_range = 0

    for row in per_mol.iter_rows(named=True):
        mol_id = row[config.molecule_id_column]
        energies = np.array(row["energies"], dtype=np.float64)
        fractions = np.array(row["fractions"], dtype=np.float64)

        min_e, max_e = interpolate_threshold_crossing(
            energies, fractions, config.threshold
        )

        if min_e is not None and max_e is not None:
            ranges.append(
                MoleculeEnergyRange(
                    molecule_id=mol_id,
                    min_energy=min_e,
                    max_energy=max_e,
                )
            )
        else:
            molecules_no_valid_range += 1

    return ranges, molecules_with_null_energy, molecules_no_valid_range


def discretize_energies(
    ranges: List[MoleculeEnergyRange],
    tolerance: float = 1.0,
) -> List[float]:
    """
    Create a set of candidate energy points from all range boundaries.

    We create candidate points at the min/max of each range, rounded to tolerance.
    This gives us a discrete set of energy values to test.

    Args:
        ranges: List of molecule energy ranges.
        tolerance: Rounding granularity for energies.

    Returns:
        Sorted list of unique candidate energy points.
    """
    if not ranges:
        return []

    all_energies: Set[float] = set()
    for r in ranges:
        rounded_min = round(r.min_energy / tolerance) * tolerance
        rounded_max = round(r.max_energy / tolerance) * tolerance
        all_energies.add(rounded_min)
        all_energies.add(rounded_max)

    return sorted(all_energies)


def find_optimal_energy_combinations_dp(
    ranges: List[MoleculeEnergyRange],
    candidate_energies: List[float],
    total_valid_molecules: int,
    max_combinations: int,
) -> List[EnergyCombinationResult]:
    """
    Exact dynamic programming algorithm to find optimal energy points.

    Finds the exact combination of up to `max_combinations` candidate energies
    that maximize the number of unique molecules covered. Exploits the fact that
    each molecule has a valid 1D continuous energy range.

    Args:
        ranges: List of molecule energy ranges.
        candidate_energies: Sorted discrete energy points to choose from.
        total_valid_molecules: Total number of valid molecules.
        max_combinations: Maximum number of energies to select.

    Returns:
        List of results for k=1, k=2, ..., up to max_combinations.
    """
    if not ranges or not candidate_energies:
        return []

    M = len(candidate_energies)
    max_k = min(max_combinations, M)

    if max_k == 0:
        return []

    # Vectorized calculation of intervals covering each candidate energy
    min_energies = np.array([r.min_energy for r in ranges], dtype=np.float64)
    max_energies = np.array([r.max_energy for r in ranges], dtype=np.float64)

    covered_by = []
    for p in candidate_energies:
        mask = (min_energies <= p) & (p <= max_energies)
        Ls = min_energies[mask]
        covered_by.append(np.sort(Ls))

    # dp[p][j] = max coverage using exactly p points, ending with candidate_energies[j].
    dp = np.full((max_k + 1, M), -1, dtype=np.int32)
    parent = np.full((max_k + 1, M), -1, dtype=np.int32)

    for j in range(M):
        dp[1][j] = len(covered_by[j])

    for p in range(2, max_k + 1):
        for j in range(M):
            Ls_sorted = covered_by[j]
            if len(Ls_sorted) == 0:
                best_i = -1
                best_val = -1
                for i in range(j):
                    if dp[p - 1][i] > best_val:
                        best_val = dp[p - 1][i]
                        best_i = i
                dp[p][j] = best_val
                parent[p][j] = best_i
                continue

            best_val = -1
            best_i = -1
            for i in range(j):
                if dp[p - 1][i] == -1:
                    continue

                # Count intervals covered by j that are NOT covered by i
                idx = np.searchsorted(Ls_sorted, candidate_energies[i], side="right")
                new_cov = len(Ls_sorted) - idx

                val = dp[p - 1][i] + new_cov
                if val > best_val:
                    best_val = val
                    best_i = i

            dp[p][j] = best_val
            parent[p][j] = best_i

    results: List[EnergyCombinationResult] = []
    prev_best_cov = -1

    for k in range(1, max_k + 1):
        best_j = int(np.argmax(dp[k]))
        best_cov = int(dp[k][best_j])

        if best_cov == -1 or best_cov <= prev_best_cov:
            break

        prev_best_cov = best_cov

        combo_indices = []
        curr_j = best_j
        for p in range(k, 0, -1):
            combo_indices.append(curr_j)
            curr_j = parent[p][curr_j]
        combo_indices.reverse()

        selected_energies = sorted([candidate_energies[idx] for idx in combo_indices])
        coverage_fraction = (
            best_cov / total_valid_molecules if total_valid_molecules > 0 else 0.0
        )

        results.append(
            EnergyCombinationResult(
                n_energies=k,
                energies=selected_energies,
                n_compounds_covered=best_cov,
                total_compounds=total_valid_molecules,
                coverage_fraction=coverage_fraction,
            )
        )

    return results


def run_analysis_for_ion_mode(
    df: pl.DataFrame,
    ion_mode: str,
    config: OptimalEnergyConfig,
) -> List[EnergyCombinationResult]:
    """
    Run the full analysis pipeline for a single ion mode.

    Args:
        df: Full DataFrame with all ion modes.
        ion_mode: Which ion mode to analyze.
        config: Configuration.

    Returns:
        List of optimal combination results.
    """
    assert config.ion_mode_column in df.columns, (
        f"DataFrame missing ion mode column: {config.ion_mode_column}"
    )

    df_filtered = df.filter(pl.col(config.ion_mode_column) == ion_mode)

    if df_filtered.height == 0:
        logger.warning("No data found for ion mode: %s", ion_mode)
        return []

    logger.info("Processing %s: %d spectra", ion_mode, df_filtered.height)

    total_compounds_all = df_filtered.select(
        pl.col(config.molecule_id_column).n_unique()
    ).item()
    logger.info("Total unique compounds in %s: %d", ion_mode, total_compounds_all)

    ranges, molecules_null_energy, molecules_no_range = compute_molecule_energy_ranges(
        df_filtered, config
    )
    molecules_with_valid_ranges = len(ranges)

    if molecules_null_energy > 0:
        logger.info(
            "%d molecules excluded due to null collision energy",
            molecules_null_energy,
        )
    if molecules_no_range > 0:
        logger.info(
            "%d molecules have NO energy range meeting threshold %.2f",
            molecules_no_range,
            config.threshold,
        )
    logger.info(
        "Found %d molecules with valid energy ranges in %s",
        molecules_with_valid_ranges,
        ion_mode,
    )

    if not ranges:
        return []

    candidate_energies = discretize_energies(ranges, config.energy_tolerance)
    logger.info(
        "Generated %d candidate energy points (tolerance=%.1f eV)",
        len(candidate_energies),
        config.energy_tolerance,
    )

    results = find_optimal_energy_combinations_dp(
        ranges, candidate_energies, molecules_with_valid_ranges, config.max_combinations
    )

    return results


def print_results(
    results: List[EnergyCombinationResult], ion_mode: str
) -> None:
    """Print results to console in human-readable format."""
    print(f"\n{'='*20} {ion_mode} Ion Mode {'='*20}")

    if not results:
        print("No results (no molecules meet threshold)")
        return

    for r in results:
        energies_str = ", ".join(f"{e:.1f}" for e in r.energies)
        print(
            f"Best {r.n_energies} energie(s): [{energies_str}] eV → "
            f"{r.n_compounds_covered:,} compounds ({r.coverage_fraction:.1%} coverage)"
        )


def save_results_to_parquet(
    results: List[EnergyCombinationResult],
    ion_mode: str,
    config: OptimalEnergyConfig,
) -> None:
    """Save results to parquet file."""
    if config.output_dir is None or not results:
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        rows.append(
            {
                "ion_mode": ion_mode,
                "n_energies": r.n_energies,
                "energies": r.energies,
                "n_compounds_covered": r.n_compounds_covered,
                "total_compounds": r.total_compounds,
                "coverage_fraction": r.coverage_fraction,
            }
        )

    df_results = pl.DataFrame(rows)
    output_path = config.output_dir / f"optimal_energies_{ion_mode.lower()}.parquet"
    df_results.write_parquet(output_path)
    logger.info("Saved results to %s", output_path)


def run_analysis(config: OptimalEnergyConfig) -> Dict[str, List[EnergyCombinationResult]]:
    """
    Main entry point - runs full analysis for both ion modes.

    Args:
        config: Configuration specifying input path, threshold, etc.

    Returns:
        Dict mapping ion mode ('P', 'N') to list of results.
    """
    assert config.parquet_path.exists(), (
        f"Parquet file not found: {config.parquet_path}"
    )
    assert 0.0 < config.threshold <= 1.0, (
        f"threshold must be in (0, 1], got {config.threshold}"
    )
    assert config.max_combinations >= 1, (
        f"max_combinations must be >= 1, got {config.max_combinations}"
    )

    required_cols = {
        config.molecule_id_column,
        config.info_column,
        config.collision_energy_column,
        config.ion_mode_column,
    }

    lf = pl.scan_parquet(config.parquet_path)

    available_cols = set(lf.collect_schema().names())
    missing = required_cols.difference(available_cols)
    assert not missing, (
        f"Parquet missing required columns: {sorted(missing)}. "
        f"Available: {sorted(available_cols)}"
    )

    # Optional columns for NCE to eV conversion
    optional_cols = {config.collision_energy_nce_column, config.precursor_mz_column}
    cols_to_select = required_cols.union(optional_cols.intersection(available_cols))

    df = lf.select(list(cols_to_select)).collect()
    logger.info(
        "Loaded %d total spectra from %s",
        df.height,
        config.parquet_path,
    )

    results: Dict[str, List[EnergyCombinationResult]] = {}

    for ion_mode in ["P", "N"]:
        ion_results = run_analysis_for_ion_mode(df, ion_mode, config)
        results[ion_mode] = ion_results

        print_results(ion_results, ion_mode)
        save_results_to_parquet(ion_results, ion_mode, config)

    return results


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Find optimal collision energy points for maximum compound coverage"
    )
    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Path to input parquet file with spectral data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output parquet files (default: no file output)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0 / 3.0,
        help="Minimum informativity fraction threshold (default: 0.667)",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=7,
        help="Maximum number of energies to combine (default: 7)",
    )
    parser.add_argument(
        "--collision-energy-column",
        type=str,
        default="collision_energy_ev",
        help="Column name for collision energy (default: collision_energy_ev)",
    )
    parser.add_argument(
        "--molecule-id-column",
        type=str,
        default="base_inchikey",
        help="Column name for molecule identifier (default: base_inchikey)",
    )
    parser.add_argument(
        "--collision-energy-nce-column",
        type=str,
        default="collision_energy_NCE",
        help="Column name for NCE collision energy (default: collision_energy_NCE)",
    )
    parser.add_argument(
        "--precursor-mz-column",
        type=str,
        default="precursor_mz",
        help="Column name for precursor m/z (default: precursor_mz)",
    )
    parser.add_argument(
        "--energy-tolerance",
        type=float,
        default=1.0,
        help="Rounding tolerance for energy values in eV (default: 1.0)",
    )

    args = parser.parse_args()

    config = OptimalEnergyConfig(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        max_combinations=args.max_combinations,
        collision_energy_column=args.collision_energy_column,
        molecule_id_column=args.molecule_id_column,
        collision_energy_nce_column=args.collision_energy_nce_column,
        precursor_mz_column=args.precursor_mz_column,
        energy_tolerance=args.energy_tolerance,
    )

    try:
        run_analysis(config)
    except AssertionError as e:
        logger.error("Validation error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)
