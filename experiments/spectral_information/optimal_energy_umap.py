"""
Find the optimal collision energy per molecule and plot a UMAP of Morgan fingerprints colored by this energy.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl
import umap
from parallel_rdkit.fingerprint import FingerprintParams, get_fp_list

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Plot UMAP of Morgan fingerprints colored by optimal collision energy."
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
        help="Directory for output plot (default: current directory)",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="smiles",
        help="Column name for SMILES (default: smiles)",
    )
    parser.add_argument(
        "--info-column",
        type=str,
        default="spectral_information_score",
        help="Column name for informativity score (default: spectral_information_score)",
    )
    parser.add_argument(
        "--collision-energy-column",
        type=str,
        default="collision_energy_ev",
        help="Column name for collision energy (default: collision_energy_ev)",
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
        "--use-nce",
        action="store_true",
        help="Optimize for NCE instead of eV. If NCE is missing, will estimate from eV.",
    )
    parser.add_argument(
        "--max-energy",
        type=float,
        default=100.0,
        help="Maximum collision energy to consider (default: 100.0).",
    )

    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_col = (
        args.collision_energy_nce_column
        if args.use_nce
        else args.collision_energy_column
    )
    fallback_col = (
        args.collision_energy_column
        if args.use_nce
        else args.collision_energy_nce_column
    )

    lf = pl.scan_parquet(args.parquet_path)
    available_cols = set(lf.collect_schema().names())

    required_cols = {args.smiles_column, args.info_column}
    missing = required_cols.difference(available_cols)
    assert not missing, f"Missing required columns: {missing}"

    has_primary = primary_col in available_cols
    has_fallback_and_mz = (fallback_col in available_cols) and (
        args.precursor_mz_column in available_cols
    )
    assert has_primary or has_fallback_and_mz, (
        f"Parquet must contain either '{primary_col}' "
        f"OR both '{fallback_col}' and '{args.precursor_mz_column}'."
    )

    cols_to_select = list(required_cols)
    if has_primary:
        cols_to_select.append(primary_col)
    if fallback_col in available_cols:
        cols_to_select.append(fallback_col)
    if args.precursor_mz_column in available_cols:
        cols_to_select.append(args.precursor_mz_column)

    df = lf.select(cols_to_select).collect()
    logger.info("Loaded %d spectra from %s", df.height, args.parquet_path)

    if primary_col not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(primary_col))

    if fallback_col in df.columns and args.precursor_mz_column in df.columns:
        if args.use_nce:
            conversion_expr = pl.col(fallback_col) * 500.0 / pl.col(args.precursor_mz_column)
        else:
            conversion_expr = pl.col(fallback_col) * pl.col(args.precursor_mz_column) / 500.0

        df = df.with_columns(
            pl.when(
                pl.col(primary_col).is_null()
                & pl.col(fallback_col).is_not_null()
                & pl.col(args.precursor_mz_column).is_not_null()
            )
            .then(conversion_expr)
            .otherwise(pl.col(primary_col))
            .alias(primary_col)
        )

    df_clean = df.filter(
        pl.col(primary_col).is_not_null()
        & pl.col(args.smiles_column).is_not_null()
        & (pl.col(primary_col) <= args.max_energy)
    )
    logger.info("Retained %d spectra with valid energy (<= %.1f) and SMILES", df_clean.height, args.max_energy)

    if df_clean.height == 0:
        logger.error("No valid data to process.")
        sys.exit(1)

    # For each molecule, find the energy that maximizes the info score
    # We sort by info_score and take the last element
    agg_df = df_clean.group_by(args.smiles_column).agg(
        optimal_energy=pl.col(primary_col).sort_by(args.info_column).last()
    )
    
    logger.info("Found optimal energy for %d unique molecules", agg_df.height)

    smiles_list = agg_df[args.smiles_column].to_list()
    optimal_energies = agg_df["optimal_energy"].to_numpy()

    logger.info("Calculating Morgan fingerprints (2048 bit, radius 2)...")
    params = FingerprintParams(fpSize=2048, radius=2)
    fp_matrix, valid_mask = get_fp_list(smiles_list, params=params, return_numpy=True)

    n_failed = (~valid_mask).sum()
    if n_failed > 0:
        logger.warning("Failed to calculate fingerprints for %d molecules", n_failed)
    
    fp_matrix_valid = fp_matrix[valid_mask]
    optimal_energies_valid = optimal_energies[valid_mask]

    if len(fp_matrix_valid) == 0:
        logger.error("No valid fingerprints were calculated. Exiting.")
        sys.exit(1)

    logger.info("Computing UMAP embedding for %d molecules using 16 workers...", len(fp_matrix_valid))
    reducer = umap.UMAP(n_jobs=16)
    embedding = reducer.fit_transform(fp_matrix_valid)

    logger.info("Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unit = "NCE" if args.use_nce else "eV"
    
    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=optimal_energies_valid,
        cmap="viridis",
        s=5,
        alpha=0.8,
        edgecolors="none"
    )
    
    # Remove axis titles, titles, and ticks per requirements
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add color scale to the right with numbers and indication of eV/NCE
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f"Optimal Energy ({unit})")
    
    out_file = out_dir / f"optimal_energy_umap_{unit.lower()}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    logger.info("Plot saved to %s", out_file)


if __name__ == "__main__":
    main()
