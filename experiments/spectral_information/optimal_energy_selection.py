"""
Find optimal collision energy points to maximize compound coverage.

For each molecule, we find the range of collision energies where at least one spectrum
achieves informativity >= threshold × max_informativity. Then we find the optimal
energy points that fall within the most molecules' valid ranges.

Algorithm:
1. Per molecule: compute min/max energy where informativity_fraction >= threshold
2. Find optimal energy points that cover the most molecule ranges (greedy)
"""

import concurrent.futures
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.lib.recfunctions as rfn
import polars as pl
import scipy.sparse as sp

logger = logging.getLogger(__name__)


@dataclass
class OptimalEnergyConfig:
    """
    Configuration for optimal energy selection analysis.

    Attributes:
        parquet_path: Path to input parquet file with spectral data.
        output_dir: Directory for output parquet files. If None, no files written.
        threshold: Minimum informativity fraction (default 2/3).
        max_combinations: Maximum number of energy points to select (default 6).
        collision_energy_column: Column name for collision energy values.
        molecule_id_column: Column name for molecule identifier.
        info_column: Column name for informativity score.
        ion_mode_column: Column name for ion mode (P/N).
        bin_size: Bin size / step size for energy grid (default 2.0 eV).
        collision_energy_nce_column: Column name for NCE collision energy.
        precursor_mz_column: Column name for precursor m/z.
    """

    parquet_path: Path
    output_dir: Optional[Path] = None
    threshold: float = 2.0 / 3.0
    max_combinations: int = 6
    collision_energy_column: str = "collision_energy_ev"
    molecule_id_column: str = "base_inchikey"
    info_column: str = "spectral_information_score"
    ion_mode_column: str = "ion_mode"
    bin_size: float = 2.0
    collision_energy_nce_column: str = "collision_energy_NCE"
    precursor_mz_column: str = "precursor_mz"
    use_nce: bool = False
    max_energy: Optional[float] = None
    plot_only: bool = False
    molecular_ion_intensity_column: str = "molecular_ion_intensity"


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
    return_informativity: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, int, Optional[np.ndarray]]:
    """
    For each molecule, interpolate informativity to a discrete energy grid.

    Args:
        df: DataFrame with spectral data for one ion mode.
        config: Configuration with threshold and column names.
        return_informativity: If True, also return the interpolated informativity matrix.

    Returns:
        Tuple of:
        - grid: Array of candidate energies
        - coverage_matrix: Boolean array (num_candidates, num_molecules)
        - Count of molecules excluded due to null collision energy
        - Count of molecules with no energy meeting threshold
        - informativity_matrix: Float array (num_candidates, num_molecules) with interpolated
          informativity values, or None if return_informativity is False
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
        hard_max = 100.0

    all_energies = df_clean[primary_col].to_numpy()
    if len(all_energies) == 0:
        empty_inf_matrix = np.empty((0, 0), dtype=np.float64) if return_informativity else None
        return np.array([]), np.empty((0, 0), dtype=bool), molecules_with_null_energy, 0, empty_inf_matrix

    min_e = max(0.0, float(np.min(all_energies)))
    max_e = min(hard_max, float(np.max(all_energies)))

    # We create a grid using bin_size as the step size
    grid = np.arange(np.floor(min_e), np.ceil(max_e) + config.bin_size, config.bin_size)

    num_candidates = len(grid)
    num_molecules = len(per_mol)

    coverage_matrix = np.zeros((num_candidates, num_molecules), dtype=bool)
    informativity_matrix = np.zeros((num_candidates, num_molecules), dtype=np.float64) if return_informativity else None

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
            if return_informativity:
                informativity_matrix[:, valid_col_idx] = interp_f
            valid_col_idx += 1

    # Keep only valid molecules
    coverage_matrix = coverage_matrix[:, :valid_col_idx]
    if return_informativity:
        informativity_matrix = informativity_matrix[:, :valid_col_idx]

    return grid, coverage_matrix, molecules_with_null_energy, molecules_no_valid_range, informativity_matrix


def _evaluate_first_item(args):
    start_idx, k, M, cands = args

    start_cov = cands[start_idx]

    if k == 1:
        return start_cov.bit_count(), (start_idx,)

    best_score = -1
    best_combo = tuple(range(start_idx, start_idx + k))  # fallback

    for combo in itertools.combinations(range(start_idx + 1, M), k - 1):
        cov = start_cov
        for idx in combo:
            cov |= cands[idx]
        score = cov.bit_count()
        if score > best_score:
            best_score = score
            best_combo = (start_idx,) + combo

    return best_score, best_combo


def find_optimal_energy_combinations(
    grid: np.ndarray,
    coverage_matrix: np.ndarray,
    total_valid_molecules: int,
    max_combinations: int,
) -> List[EnergyCombinationResult]:
    """
    Exact optimization algorithm to find optimal energy points using bitpacking and parallel combinations.

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

    logger.info(
        "Optimization simplified: candidates %d -> %d, molecules %d",
        num_candidates,
        M_pruned,
        num_molecules,
    )

    # 2. Bitpack the coverage matrix
    # Convert each candidate's boolean row into a single Python integer representing a bitset
    packed_cov = np.packbits(pruned_cov, axis=1)
    candidates_int = [int.from_bytes(row.tobytes(), "little") for row in packed_cov]

    results: List[EnergyCombinationResult] = []
    prev_best_cov = -1

    for k in range(1, max_k + 1):
        best_cov = -1
        best_combo_original = []

        tasks = [(i, k, M_pruned, candidates_int) for i in range(M_pruned - k + 1)]

        # Parallel evaluate over start_idx
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for score, combo in executor.map(_evaluate_first_item, tasks):
                if score > best_cov:
                    best_cov = score
                    best_combo_original = [int(pruned_indices[idx]) for idx in combo]

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
    bin_size = config.bin_size

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

    out_1d = out_dir / f"coverage_1d_{ion_mode}_{bin_size}bin_{unit.lower()}.png"
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

    out_2d = out_dir / f"coverage_2d_{ion_mode}_{bin_size}bin_{unit.lower()}.png"
    fig.savefig(out_2d, dpi=600, bbox_inches="tight")
    plt.close(fig)


def generate_combined_plots(
    plot_data: Dict[str, Tuple[np.ndarray, np.ndarray, int]],
    config: OptimalEnergyConfig,
) -> None:
    """
    Generate combined plots with both ionization modes.
    
    Args:
        plot_data: Dict mapping ion_mode ('P', 'N') to (grid, coverage_matrix, total_valid_molecules).
        config: Configuration.
    """
    import matplotlib.pyplot as plt

    unit = "NCE" if config.use_nce else "eV"
    bin_size = config.bin_size

    out_dir = config.output_dir if config.output_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check which modes have data
    has_p = "P" in plot_data and len(plot_data["P"][0]) > 0
    has_n = "N" in plot_data and len(plot_data["N"][0]) > 0

    if not has_p and not has_n:
        return

    # Calculate 1D coverage for both modes
    data_1d = {}
    for mode in ["P", "N"]:
        if mode in plot_data:
            grid, cov_matrix, total_mol = plot_data[mode]
            if len(grid) > 0 and total_mol > 0:
                cov_1d_raw = cov_matrix.sum(axis=1)
                cov_1d = cov_1d_raw / total_mol * 100.0
                data_1d[mode] = (grid, cov_1d)

    # === 1D Combined: Stacked subplots ===
    if len(data_1d) > 0:
        modes = [m for m in ["P", "N"] if m in data_1d]
        n_modes = len(modes)
        
        fig, axes = plt.subplots(n_modes, 1, sharex=True, figsize=(8, 3 * n_modes))
        if n_modes == 1:
            axes = [axes]
        
        for idx, mode in enumerate(modes):
            ax = axes[idx]
            grid, cov_1d = data_1d[mode]
            ax.plot(grid, cov_1d, marker="o")
            ax.set_ylabel(f"{mode}\nCoverage (%)")
        
        axes[-1].set_xlabel(f"Collision Energy ({unit})")
        
        out_combined_1d = out_dir / f"coverage_1d_combined_{bin_size}bin_{unit.lower()}.png"
        fig.savefig(out_combined_1d, dpi=600, bbox_inches="tight")
        plt.close(fig)

    # === 1D Combined: Both traces on same axes ===
    if len(data_1d) > 0:
        fig, ax = plt.subplots()
        
        for mode in ["P", "N"]:
            if mode in data_1d:
                grid, cov_1d = data_1d[mode]
                ax.plot(grid, cov_1d, marker="o")
        
        ax.set_xlabel(f"Collision Energy ({unit})")
        ax.set_ylabel("Coverage (%)")
        
        out_overlay_1d = out_dir / f"coverage_1d_overlay_{bin_size}bin_{unit.lower()}.png"
        fig.savefig(out_overlay_1d, dpi=600, bbox_inches="tight")
        plt.close(fig)

    # Calculate 2D coverage for both modes
    data_2d = {}
    for mode in ["P", "N"]:
        if mode in plot_data:
            grid, cov_matrix, total_mol = plot_data[mode]
            if len(grid) > 0 and total_mol > 0:
                cov_1d_raw = cov_matrix.sum(axis=1)
                masks_int = cov_matrix.astype(np.int32)
                intersection = masks_int @ masks_int.T
                cov_2d_raw = cov_1d_raw[:, None] + cov_1d_raw[None, :] - intersection
                cov_2d = cov_2d_raw / total_mol * 100.0
                data_2d[mode] = (grid, cov_2d)

    # === 2D Combined: Stacked subplots with shared color scale ===
    if len(data_2d) > 0:
        modes = [m for m in ["P", "N"] if m in data_2d]
        n_modes = len(modes)
        
        # Find global min/max for shared color scale
        vmin = min(data_2d[m][1].min() for m in modes)
        vmax = max(data_2d[m][1].max() for m in modes)
        
        fig, axes = plt.subplots(n_modes, 1, sharex=True, sharey=True, figsize=(8, 4 * n_modes))
        if n_modes == 1:
            axes = [axes]
        
        # P at top, N at bottom
        for idx, mode in enumerate(modes):
            ax = axes[idx]
            grid, cov_2d = data_2d[mode]
            c = ax.pcolormesh(grid, grid, cov_2d, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_ylabel(f"{mode}\nCollision Energy 2 ({unit})")
            ax.set_aspect("equal")
        
        axes[-1].set_xlabel(f"Collision Energy 1 ({unit})")
        
        # Shared colorbar
        cbar = fig.colorbar(c, ax=axes, orientation="vertical", pad=0.02)
        cbar.set_label("Coverage (%)")
        
        out_combined_2d = out_dir / f"coverage_2d_combined_{bin_size}bin_{unit.lower()}.png"
        fig.savefig(out_combined_2d, dpi=600, bbox_inches="tight")
        plt.close(fig)


def generate_optimal_energy_histogram(
    grid: np.ndarray,
    informativity_matrix: np.ndarray,
    ion_mode: str,
    config: OptimalEnergyConfig,
) -> None:
    """
    Generate and save a histogram of the optimal energy per molecule.
    
    The optimal energy for each molecule is defined as the energy point where
    the interpolated informativity is maximized.
    
    Args:
        grid: Array of candidate energies (energy bins).
        informativity_matrix: Float array (num_candidates, num_molecules) with 
            interpolated informativity values for each molecule at each energy point.
        ion_mode: Ion mode being analyzed ('P' or 'N').
        config: Configuration with output directory and bin size.
    """
    if len(grid) == 0 or informativity_matrix is None or informativity_matrix.shape[1] == 0:
        return

    import matplotlib.pyplot as plt

    unit = "NCE" if config.use_nce else "eV"
    bin_size = config.bin_size

    out_dir = config.output_dir if config.output_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find the optimal energy for each molecule (where informativity is maximized)
    # informativity_matrix has shape (num_candidates, num_molecules)
    optimal_indices = np.argmax(informativity_matrix, axis=0)
    optimal_energies = grid[optimal_indices]

    # Always use a max_energy of 100 by default, or the configured value
    # Filter out molecules with optimal energy above max_energy for display
    # but keep them in the total count for fraction calculation
    effective_max_energy = config.max_energy if config.max_energy is not None else 100.0
    valid_mask = optimal_energies <= effective_max_energy
    filtered_energies = optimal_energies[valid_mask]

    total_count = len(optimal_energies)  # Total for fraction calculation (includes filtered)

    if len(filtered_energies) == 0:
        logger.warning("No molecules with optimal energy <= %s for %s mode", effective_max_energy, ion_mode)
        return

    # Create histogram with fraction (sum of all bars = 1)
    fig, ax = plt.subplots()

    # Use the grid edges for histogram bins
    # Filter grid to only include values <= max_energy for bin edges
    grid_for_bins = grid[grid <= effective_max_energy]

    # Create bin edges based on filtered grid
    bin_edges = np.arange(
        grid_for_bins[0] - bin_size / 2,
        grid_for_bins[-1] + bin_size,
        bin_size
    )

    n, bins, patches = ax.hist(filtered_energies, bins=bin_edges, density=False, alpha=0.7, edgecolor='black')
    # Normalize to fraction using total count (including filtered out molecules)
    n_fraction = n / total_count
    ax.clear()
    ax.bar(bins[:-1], n_fraction, width=np.diff(bins), alpha=0.7, edgecolor='black', align='edge')
    ax.set_xlabel(f"Optimal Collision Energy ({unit})")
    ax.set_ylabel("Fraction of Molecules")

    out_hist = out_dir / f"optimal_energy_histogram_{ion_mode}_{bin_size}bin_{unit.lower()}.png"
    fig.savefig(out_hist, dpi=600, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "Saved optimal energy histogram for %s mode to %s (%d molecules shown, %d total)",
        ion_mode,
        out_hist,
        len(filtered_energies),
        total_count,
    )


def generate_molecular_ion_intensity_histogram(
    df: pl.DataFrame,
    grid: np.ndarray,
    informativity_matrix: np.ndarray,
    ion_mode: str,
    config: OptimalEnergyConfig,
) -> None:
    """
    Generate and save a histogram of molecular ion intensity for maximally informative spectra.
    
    For each molecule, identifies the spectrum with maximum informativity and plots
    the distribution of molecular ion intensities for those maximally informative spectra.
    
    Args:
        df: DataFrame with spectral data including molecular_ion_intensity column.
        grid: Array of candidate energies (energy bins).
        informativity_matrix: Float array (num_candidates, num_molecules) with 
            interpolated informativity values for each molecule at each energy point.
        ion_mode: Ion mode being analyzed ('P' or 'N').
        config: Configuration with output directory and column names.
    """
    if len(grid) == 0 or informativity_matrix is None or informativity_matrix.shape[1] == 0:
        return

    mol_ion_col = config.molecular_ion_intensity_column
    if mol_ion_col not in df.columns:
        logger.warning(
            "Column '%s' not found in DataFrame, skipping molecular ion intensity histogram",
            mol_ion_col
        )
        return

    import matplotlib.pyplot as plt

    unit = "NCE" if config.use_nce else "eV"
    bin_size = config.bin_size

    out_dir = config.output_dir if config.output_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find the optimal energy index for each molecule
    # informativity_matrix has shape (num_candidates, num_molecules)
    optimal_indices = np.argmax(informativity_matrix, axis=0)
    optimal_energies = grid[optimal_indices]

    # Get molecular IDs that have valid ranges
    # We need to match these with the original df
    ev_col = config.collision_energy_column
    nce_col = config.collision_energy_nce_column
    
    if config.use_nce:
        primary_col = nce_col
    else:
        primary_col = ev_col

    # Filter df to molecules with valid ranges and get their molecular ion intensities
    # For each molecule, find the spectrum closest to its optimal energy
    df_clean = df.filter(pl.col(primary_col).is_not_null())
    
    # Create a mapping from molecule to its optimal energy
    # We need to get the unique molecules in the same order as informativity_matrix columns
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

    # Build a list of (molecule_id, optimal_energy) pairs
    molecule_optimal_energies = []
    mol_idx = 0
    for row in per_mol.iter_rows(named=True):
        if mol_idx < len(optimal_energies):
            molecule_optimal_energies.append((row[config.molecule_id_column], optimal_energies[mol_idx]))
            mol_idx += 1

    # For each molecule, find the spectrum closest to the optimal energy and get its molecular ion intensity
    molecular_ion_intensities = []
    
    for mol_id, opt_energy in molecule_optimal_energies:
        # Get spectra for this molecule
        mol_spectra = df_clean.filter(pl.col(config.molecule_id_column) == mol_id)
        
        if mol_spectra.height == 0:
            continue
            
        # Find the spectrum closest to the optimal energy
        energy_diffs = np.abs(mol_spectra[primary_col].to_numpy() - opt_energy)
        closest_idx = np.argmin(energy_diffs)
        
        # Get molecular ion intensity for this spectrum
        mol_intensity = mol_spectra[mol_ion_col].to_numpy()[closest_idx]
        if mol_intensity is not None and not np.isnan(mol_intensity):
            molecular_ion_intensities.append(mol_intensity)

    if len(molecular_ion_intensities) == 0:
        logger.warning("No valid molecular ion intensities found for %s mode", ion_mode)
        return

    molecular_ion_intensities = np.array(molecular_ion_intensities)

    # Create histogram with fraction (sum of all bars = 1)
    fig, ax = plt.subplots()
    
    # Use automatic binning for the intensity values
    n, bins, patches = ax.hist(molecular_ion_intensities, bins=50, density=False, alpha=0.7, edgecolor='black')
    # Normalize to fraction
    n_fraction = n / len(molecular_ion_intensities)
    ax.clear()
    ax.bar(bins[:-1], n_fraction, width=np.diff(bins), alpha=0.7, edgecolor='black', align='edge')
    ax.set_xlabel("Molecular Ion Intensity")
    ax.set_ylabel("Fraction of Molecules")

    out_hist = out_dir / f"molecular_ion_intensity_histogram_{ion_mode}_{bin_size}bin_{unit.lower()}.png"
    fig.savefig(out_hist, dpi=600, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "Saved molecular ion intensity histogram for %s mode to %s (%d spectra, mean=%.2e, median=%.2e)",
        ion_mode,
        out_hist,
        len(molecular_ion_intensities),
        np.mean(molecular_ion_intensities),
        np.median(molecular_ion_intensities),
    )


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
    results, _, _, _ = run_analysis_for_ion_mode_with_data(df, ion_mode, config)
    return results


def run_analysis_for_ion_mode_with_data(
    df: pl.DataFrame,
    ion_mode: str,
    config: OptimalEnergyConfig,
) -> Tuple[List[EnergyCombinationResult], np.ndarray, np.ndarray, int]:
    """
    Run the full analysis pipeline for a single ion mode and return data for plotting.

    Args:
        df: Full DataFrame with all ion modes.
        ion_mode: Which ion mode to analyze.
        config: Configuration.

    Returns:
        Tuple of:
        - List of optimal combination results
        - Energy grid array
        - Coverage matrix
        - Number of valid molecules
    """
    assert config.ion_mode_column in df.columns, (
        f"DataFrame missing ion mode column: {config.ion_mode_column}"
    )

    df_filtered = df.filter(pl.col(config.ion_mode_column) == ion_mode)

    if df_filtered.height == 0:
        logger.warning("No data found for ion mode: %s", ion_mode)
        return [], np.array([]), np.empty((0, 0), dtype=bool), 0

    logger.info("Processing %s: %d spectra", ion_mode, df_filtered.height)

    total_compounds_all = df_filtered.select(
        pl.col(config.molecule_id_column).n_unique()
    ).item()
    logger.info("Total unique compounds in %s: %d", ion_mode, total_compounds_all)

    grid, cov_matrix, molecules_null_energy, molecules_no_range, informativity_matrix = get_coverage_matrix(
        df_filtered, config, return_informativity=True
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
        return [], grid, cov_matrix, 0

    unit = "NCE" if config.use_nce else "eV"
    logger.info(
        "Generated %d candidate energy points (bin_size=%.1f %s)",
        len(grid),
        config.bin_size,
        unit,
    )

    generate_plots(grid, cov_matrix, molecules_with_valid_ranges, ion_mode, config)
    generate_optimal_energy_histogram(grid, informativity_matrix, ion_mode, config)
    generate_molecular_ion_intensity_histogram(df_filtered, grid, informativity_matrix, ion_mode, config)

    if config.plot_only:
        return [], grid, cov_matrix, molecules_with_valid_ranges

    results = find_optimal_energy_combinations(
        grid, cov_matrix, molecules_with_valid_ranges, config.max_combinations
    )

    return results, grid, cov_matrix, molecules_with_valid_ranges


def print_results(
    results: List[EnergyCombinationResult],
    ion_mode: str,
    config: OptimalEnergyConfig,
    unit: str = "eV",
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

    optional_cols = {fallback_col, config.precursor_mz_column, config.molecular_ion_intensity_column}
    cols_to_select.update(optional_cols.intersection(available_cols))

    df = lf.select(list(cols_to_select)).collect()
    logger.info(
        "Loaded %d total spectra from %s",
        df.height,
        config.parquet_path,
    )

    results: Dict[str, List[EnergyCombinationResult]] = {}
    plot_data: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}

    for ion_mode in ["P", "N"]:
        ion_results, grid, cov_matrix, valid_molecules = run_analysis_for_ion_mode_with_data(
            df, ion_mode, config
        )
        results[ion_mode] = ion_results
        
        # Store plot data for combined figures
        if len(grid) > 0 and valid_molecules > 0:
            plot_data[ion_mode] = (grid, cov_matrix, valid_molecules)

        unit = "NCE" if config.use_nce else "eV"
        print_results(ion_results, ion_mode, config, unit=unit)
        save_results_to_parquet(ion_results, ion_mode, config)

    # Generate combined plots after processing both ion modes
    if len(plot_data) > 0:
        generate_combined_plots(plot_data, config)

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
        default=6,
        help="Maximum number of energies to combine (default: 6)",
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
        "--bin-size",
        type=float,
        default=2.0,
        help="Step size for energy grid during optimization (default: 2.0)",
    )
    parser.add_argument(
        "--use-nce",
        action="store_true",
        help="Optimize for NCE instead of eV. If NCE is missing, will estimate from eV.",
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
    parser.add_argument(
        "--molecular-ion-intensity-column",
        type=str,
        default="molecular_ion_intensity",
        help="Column name for molecular ion intensity (default: molecular_ion_intensity)",
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
        bin_size=args.bin_size,
        use_nce=args.use_nce,
        max_energy=args.max_energy,
        plot_only=args.plot_only,
        molecular_ion_intensity_column=args.molecular_ion_intensity_column,
    )

    try:
        run_analysis(config)
    except AssertionError as e:
        logger.error("Validation error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)
