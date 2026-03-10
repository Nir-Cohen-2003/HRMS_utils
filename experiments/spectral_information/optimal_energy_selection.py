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
import numpy.lib.recfunctions as rfn
import polars as pl
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

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
    use_nce: bool = False
    plot_bin_size: float = 5.0
    max_energy: Optional[float] = None
    plot_only: bool = False


@dataclass
class EnergyCombinationResult:
    """Result for a single energy combination."""

    n_energies: int
    energies: List[float]
    n_compounds_covered: int
    total_compounds: int
    coverage_fraction: float


def get_coverage_matrix(
    df: pl.DataFrame,
    config: OptimalEnergyConfig,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    For each molecule, interpolate informativity to a discrete energy grid.
    
    Args:
        df: DataFrame with spectral data for one ion mode.
        config: Configuration with threshold and column names.
        
    Returns:
        Tuple of:
        - grid: Array of candidate energies
        - coverage_matrix: Boolean array (num_candidates, num_molecules)
        - Count of molecules excluded due to null collision energy
        - Count of molecules with no energy meeting threshold
    """
    ev_col = config.collision_energy_column
    nce_col = config.collision_energy_nce_column
    mz_col = config.precursor_mz_column

    if config.use_nce:
        primary_col = nce_col
        fallback_col = ev_col
    else:
        primary_col = ev_col
        fallback_col = nce_col

    if primary_col not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(primary_col))

    required_cols = {
        config.molecule_id_column,
        config.info_column,
        primary_col,
    }
    missing = required_cols.difference(set(df.columns))
    assert not missing, f"DataFrame missing required columns: {sorted(missing)}"

    if fallback_col in df.columns and mz_col in df.columns:
        if config.use_nce:
            conversion_expr = pl.col(fallback_col) * 500.0 / pl.col(mz_col)
        else:
            conversion_expr = pl.col(fallback_col) * pl.col(mz_col) / 500.0

        df = df.with_columns(
            pl.when(
                pl.col(primary_col).is_null()
                & pl.col(fallback_col).is_not_null()
                & pl.col(mz_col).is_not_null()
            )
            .then(conversion_expr)
            .otherwise(pl.col(primary_col))
            .alias(primary_col)
        )

    df_clean = df.filter(pl.col(primary_col).is_not_null())

    molecules_with_null_energy = (
        df.select(pl.col(config.molecule_id_column).n_unique()).item()
        - df_clean.select(pl.col(config.molecule_id_column).n_unique()).item()
    )

    df_with_fraction = df_clean.with_columns(
        max_info_per_mol=pl.col(config.info_column)
        .max()
        .over(config.molecule_id_column)
    ).with_columns(
        informativity_fraction=pl.col(config.info_column) / pl.col("max_info_per_mol")
    )

    per_mol = df_with_fraction.group_by(config.molecule_id_column).agg(
        energies=pl.col(primary_col).sort(),
        fractions=pl.col("informativity_fraction").sort_by(pl.col(primary_col)),
    )

    # Determine grid
    if config.max_energy is not None:
        hard_max = config.max_energy
    else:
        hard_max = 150.0 if config.use_nce else 100.0

    all_energies = df_clean[primary_col].to_numpy()
    if len(all_energies) == 0:
        return np.array([]), np.empty((0,0), dtype=bool), molecules_with_null_energy, 0
        
    min_e = max(0.0, float(np.min(all_energies)))
    max_e = min(hard_max, float(np.max(all_energies)))
    
    # We create a grid using energy_tolerance as the step size
    # To ensure we don't have too many candidates, if tolerance is very small, we might want to cap it.
    # But usually tolerance is >= 1.0
    grid = np.arange(np.floor(min_e), np.ceil(max_e) + config.energy_tolerance, config.energy_tolerance)
    
    num_candidates = len(grid)
    num_molecules = len(per_mol)
    
    coverage_matrix = np.zeros((num_candidates, num_molecules), dtype=bool)
    
    molecules_no_valid_range = 0
    valid_col_idx = 0
    
    for row in per_mol.iter_rows(named=True):
        energies = np.array(row["energies"], dtype=np.float64)
        fractions = np.array(row["fractions"], dtype=np.float64)
        
        if len(energies) == 0:
            continue
            
        # Interpolate fractions onto the grid
        # Molecules with high info at e.g. 20 and 80 but 0 at 45 will correctly dip below threshold
        interp_f = np.interp(grid, energies, fractions, left=0.0, right=0.0)
        mask = interp_f >= config.threshold
        
        if not np.any(mask):
            molecules_no_valid_range += 1
        else:
            coverage_matrix[:, valid_col_idx] = mask
            valid_col_idx += 1

    # Keep only valid molecules
    coverage_matrix = coverage_matrix[:, :valid_col_idx]
    
    return grid, coverage_matrix, molecules_with_null_energy, molecules_no_valid_range


def find_optimal_energy_combinations(
    grid: np.ndarray,
    coverage_matrix: np.ndarray,
    total_valid_molecules: int,
    max_combinations: int,
) -> List[EnergyCombinationResult]:
    """
    Exact optimization algorithm to find optimal energy points using MILP.

    Args:
        grid: Array of candidate energies.
        coverage_matrix: Boolean array (num_candidates, num_molecules).
        total_valid_molecules: Total number of valid molecules (number of columns).
        max_combinations: Maximum number of energies to select.

    Returns:
        List of results for k=1, k=2, ..., up to max_combinations.
    """
    if len(grid) == 0 or total_valid_molecules == 0:
        return []

    num_candidates, num_molecules = coverage_matrix.shape
    max_k = min(max_combinations, num_candidates)

    if max_k == 0:
        return []

    # 1. Prune completely dominated candidates to reduce problem size
    keep_candidates = np.ones(num_candidates, dtype=bool)
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j and keep_candidates[j] and keep_candidates[i]:
                # If candidate i is a subset of candidate j (i is dominated by j)
                if not np.any(coverage_matrix[i] & ~coverage_matrix[j]):
                    # Break ties by index
                    if np.array_equal(coverage_matrix[i], coverage_matrix[j]):
                        if i > j:
                            keep_candidates[i] = False
                    else:
                        keep_candidates[i] = False

    pruned_indices = np.where(keep_candidates)[0]
    pruned_cov = coverage_matrix[pruned_indices]
    M_pruned = len(pruned_indices)

    # 2. Merge identical molecules to reduce constraints
    packed = np.packbits(pruned_cov.T, axis=1)
    structured = rfn.unstructured_to_structured(packed)
    _, unique_cols, counts = np.unique(structured, return_index=True, return_counts=True)
    
    unique_cov = pruned_cov[:, unique_cols]
    U = len(counts)

    logger.info(
        "Optimization simplified: candidates %d -> %d, molecules %d -> %d",
        num_candidates, M_pruned, num_molecules, U
    )

    results: List[EnergyCombinationResult] = []
    prev_best_cov = -1

    for k in range(1, max_k + 1):
        if k == 1:
            # Simple max for k=1
            best_local_idx = int(np.argmax(pruned_cov.sum(axis=1)))
            best_cov = int(pruned_cov[best_local_idx].sum())
            best_combo_original = [int(pruned_indices[best_local_idx])]
        else:
            # Set up MILP
            # Variables: [x_1..x_M_pruned, y_1..y_U]
            # Objective: Maximize sum(count_j * y_j) -> Minimize -sum(count_j * y_j)
            c = np.zeros(M_pruned + U)
            c[M_pruned:] = -counts
            
            # Constraint 1: sum(x_i) <= k
            row_0 = np.zeros(M_pruned + U)
            row_0[:M_pruned] = 1
            
            # Constraint 2: For each unique molecule j: y_j - sum_i(cov[i,j] * x_i) <= 0
            A_ub_main = sp.hstack([ -unique_cov.T.astype(int), sp.eye(U) ])
            A_ub = sp.vstack([ row_0.reshape(1, -1), A_ub_main ])
            
            b_ub = np.zeros(1 + U)
            b_ub[0] = k
            
            constraints = LinearConstraint(A_ub, -np.inf, b_ub)
            integrality = np.ones(M_pruned + U)
            bounds = Bounds(0, 1)
            
            # Solve
            res = milp(
                c=c, 
                constraints=constraints, 
                integrality=integrality, 
                bounds=bounds,
                options={'time_limit': 120} # safety timeout
            )
            
            if res.success:
                x_sol = res.x[:M_pruned]
                selected_local = np.where(x_sol > 0.5)[0]
                # True coverage sum:
                mask = np.zeros(num_molecules, dtype=bool)
                for sl in selected_local:
                    mask |= pruned_cov[sl]
                best_cov = int(mask.sum())
                best_combo_original = [int(pruned_indices[sl]) for sl in selected_local]
            else:
                logger.warning("MILP failed for k=%d", k)
                break

        if best_cov <= prev_best_cov:
            break

        prev_best_cov = best_cov

        selected_energies = sorted([float(grid[idx]) for idx in best_combo_original])
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


def generate_plots(
    grid: np.ndarray,
    coverage_matrix: np.ndarray,
    total_valid_molecules: int,
    ion_mode: str,
    config: OptimalEnergyConfig,
) -> None:
    """Generate and save 1D and 2D coverage plots."""
    if len(grid) == 0 or total_valid_molecules == 0:
        return

    import matplotlib.pyplot as plt

    unit = "NCE" if config.use_nce else "eV"
    bin_size = config.plot_bin_size

    out_dir = config.output_dir if config.output_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calculate 1D coverage from the actual coverage matrix
    cov_1d_raw = coverage_matrix.sum(axis=1)
    cov_1d = cov_1d_raw / total_valid_molecules * 100.0

    # 1D plot
    fig, ax = plt.subplots()
    ax.plot(grid, cov_1d, marker="o")
    ax.set_xlabel(f"Collision Energy ({unit})")
    ax.set_ylabel("Coverage (%)")

    out_1d = (
        out_dir / f"coverage_1d_{ion_mode}_{bin_size}bin_{unit.lower()}.png"
    )
    fig.savefig(out_1d, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # 2D calculation using inclusion-exclusion principle: |A ∪ B| = |A| + |B| - |A ∩ B|
    # Computes pairwise intersection via matrix multiplication for massive memory savings
    masks_int = coverage_matrix.astype(np.int32)
    intersection = masks_int @ masks_int.T
    
    cov_2d_raw = cov_1d_raw[:, None] + cov_1d_raw[None, :] - intersection
    cov_2d = cov_2d_raw / total_valid_molecules * 100.0

    # 2D plot
    fig, ax = plt.subplots()
    c = ax.pcolormesh(grid, grid, cov_2d, shading="auto", cmap="viridis")
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Coverage (%)")
    ax.set_xlabel(f"Collision Energy 1 ({unit})")
    ax.set_ylabel(f"Collision Energy 2 ({unit})")

    out_2d = (
        out_dir / f"coverage_2d_{ion_mode}_{bin_size}bin_{unit.lower()}.png"
    )
    fig.savefig(out_2d, dpi=600, bbox_inches="tight")
    plt.close(fig)


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

    grid, cov_matrix, molecules_null_energy, molecules_no_range = get_coverage_matrix(
        df_filtered, config
    )
    molecules_with_valid_ranges = cov_matrix.shape[1] if len(grid) > 0 else 0

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

    if molecules_with_valid_ranges == 0:
        return []

    unit = "NCE" if config.use_nce else "eV"
    logger.info(
        "Generated %d candidate energy points (tolerance=%.1f %s)",
        len(grid),
        config.energy_tolerance,
        unit,
    )

    generate_plots(grid, cov_matrix, molecules_with_valid_ranges, ion_mode, config)

    if config.plot_only:
        return []

    results = find_optimal_energy_combinations(
        grid, cov_matrix, molecules_with_valid_ranges, config.max_combinations
    )

    return results


def print_results(
    results: List[EnergyCombinationResult], ion_mode: str, config: OptimalEnergyConfig, unit: str = "eV"
) -> None:
    """Print results to console in human-readable format."""
    print(f"\n{'=' * 20} {ion_mode} Ion Mode {'=' * 20}")

    if config.plot_only:
        print("Set search skipped (--plot-only is enabled).")
        return

    if not results:
        print("No results (no molecules meet threshold)")
        return

    for r in results:
        energies_str = ", ".join(f"{e:.1f}" for e in r.energies)
        print(
            f"Best {r.n_energies} energie(s): [{energies_str}] {unit} → "
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


def run_analysis(
    config: OptimalEnergyConfig,
) -> Dict[str, List[EnergyCombinationResult]]:
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

    primary_col = (
        config.collision_energy_nce_column
        if config.use_nce
        else config.collision_energy_column
    )
    fallback_col = (
        config.collision_energy_column
        if config.use_nce
        else config.collision_energy_nce_column
    )

    required_base_cols = {
        config.molecule_id_column,
        config.info_column,
        config.ion_mode_column,
    }

    lf = pl.scan_parquet(config.parquet_path)
    available_cols = set(lf.collect_schema().names())

    missing_base = required_base_cols.difference(available_cols)
    assert not missing_base, (
        f"Parquet missing base required columns: {sorted(missing_base)}. "
        f"Available: {sorted(available_cols)}"
    )

    has_primary = primary_col in available_cols
    has_fallback_and_mz = (fallback_col in available_cols) and (
        config.precursor_mz_column in available_cols
    )
    assert has_primary or has_fallback_and_mz, (
        f"Parquet must contain either the primary energy column ('{primary_col}') "
        f"OR both the fallback energy column ('{fallback_col}') and precursor mz column ('{config.precursor_mz_column}')."
    )

    cols_to_select = required_base_cols.copy()
    if has_primary:
        cols_to_select.add(primary_col)

    optional_cols = {fallback_col, config.precursor_mz_column}
    cols_to_select.update(optional_cols.intersection(available_cols))

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

        unit = "NCE" if config.use_nce else "eV"
        print_results(ion_results, ion_mode, config, unit=unit)
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
        help="Rounding tolerance for energy values (default: 1.0)",
    )
    parser.add_argument(
        "--use-nce",
        action="store_true",
        help="Optimize for NCE instead of eV. If NCE is missing, will estimate from eV.",
    )
    parser.add_argument(
        "--plot-bin-size",
        type=float,
        default=5.0,
        help="Energy bin size for plots (default: 5.0)",
    )
    parser.add_argument(
        "--max-energy",
        type=float,
        default=None,
        help="Maximum collision energy to consider. Defaults to 100 eV or 150 NCE.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Generate plots only without running set optimization.",
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
        use_nce=args.use_nce,
        plot_bin_size=args.plot_bin_size,
        max_energy=args.max_energy,
        plot_only=args.plot_only,
    )

    try:
        run_analysis(config)
    except AssertionError as e:
        logger.error("Validation error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)
