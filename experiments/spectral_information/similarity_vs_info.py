import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

import polars as pl
from fast_cosine_sim import GPUApproximateConfig, batched_approximate_similarity_gpu

# Import hrms_core to register the spectral_similarity and spectral_info plugins
import hrms_utils.hrms_core  # noqa: F401

from utils import compute_and_save_tanimoto_scores  # type: ignore

os.environ["RUST_BACKTRACE"] = "full"


if __name__ == "__main__":
    LIBRARY_PATH = Path(
        "file:///home/analytit_admin/Data/spectral_libs/info_score/combined_library.parquet"
    )
    PAIRS_PATH = Path(
        "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_pairs_260121.parquet"
    )
    APPROX_PAIRS_PATH = PAIRS_PATH.with_suffix(".approx.parquet")
    LOG_PATH = PAIRS_PATH.with_suffix(".log")
    SNAPSHOT_PATH = PAIRS_PATH.with_suffix(".left_library.parquet")
    PAIRS_WITH_TANIMOTO_PATH = PAIRS_PATH.with_suffix(".with_tanimoto.parquet")

    pass


LIBRARY_PATH = Path("experiments/spectral_information/data.parquet")
PAIRS_PATH = Path("experiments/spectral_information/data_pairs_260311.parquet")
APPROX_PAIRS_PATH = PAIRS_PATH.with_suffix(".approx.parquet")
PAIRS_WITH_TANIMOTO_PATH = PAIRS_PATH.with_stem(PAIRS_PATH.stem + "_with_tanimoto")
LOG_PATH = PAIRS_PATH.with_suffix(".log")

# Threshold configuration
# Why: Approximate threshold is lower to ensure we don't miss candidates that might
# pass the exact threshold after proper peak matching. The gap accounts for binning
# artifacts and tolerance window expansion effects.
EXACT_THRESHOLD = 0.5
APPROX_THRESHOLD = 0.35

    # =========================================================================
    # Step 0: Load library, assign indices, and recompute info scores
    # =========================================================================
    # Why: Downstream tools (tanimoto, plotting) expect integer idx/mol_idx and
    # correct spectral_information_score values. We recompute the score here using
    # the Rust plugin to ensure correctness.

    logging.info(f"Loading library from {LIBRARY_PATH}")
    library_lf = pl.scan_parquet(str(LIBRARY_PATH))

    # Drop rows with missing or placeholder SMILES so they never enter the pipeline.
    if "smiles" in library_lf.collect_schema().names():
        library_lf = library_lf.filter(
            pl.col("smiles").is_not_null() & (pl.col("smiles") != "NOT FOUND")
        )

    # Assign integer idx and molecule index for downstream compatibility
    library_lf = library_lf.with_row_index("idx")
    if "base_inchikey" in library_lf.collect_schema().names() and "ion_mode" in library_lf.collect_schema().names():
        library_lf = library_lf.with_columns(
            mol_idx=pl.col("idx").min().over(["base_inchikey", "ion_mode"])
        )
    else:
        logging.warning(
            "base_inchikey or ion_mode not found in library; using idx as mol_idx"
        )
        library_lf = library_lf.with_columns(mol_idx=pl.col("idx"))

    # Recompute spectral_information_score if formula columns are available
    schema_names = library_lf.collect_schema().names()
    if (
        "precursor_formula_array" in schema_names
        and "cleaned_fragment_formulas" in schema_names
    ):
        logging.info("Recomputing spectral_information_score (ignore_hydrogens=True)")
        cols_to_drop = [
            c
            for c in [
                "spectral_information_score",
                "spectral_information_score_with_hydrogens",
            ]
            if c in schema_names
        ]
        if cols_to_drop:
            library_lf = library_lf.drop(cols_to_drop)

        library_lf = library_lf.with_columns(
            pl.struct(
                [
                    pl.col("precursor_formula_array").alias("precursor_formula"),
                    pl.col("cleaned_fragment_formulas").alias("fragment_formulas"),
                ]
            )
            .spectral_info.spectral_info_score(distance_metric="l2", ignore_hydrogens=True)
            .alias("spectral_information_score")
        )
    else:
        logging.warning(
            "Missing precursor_formula_array or cleaned_fragment_formulas; "
            "using existing spectral_information_score from library file"
        )

    logging.info("Collecting full library for writing...")
    library_df = library_lf.collect(engine="streaming")

    FULL_LIBRARY_PATH = PAIRS_PATH.with_suffix(".left_library_full.parquet")
    logging.info("Writing full library to %s", FULL_LIBRARY_PATH)
    library_df.write_parquet(FULL_LIBRARY_PATH)

    snapshot_cols = [
        "idx",
        "mol_idx",
        "base_inchikey",
        "ion_mode",
        "smiles",
        "precursor_mz",
        "spectral_information_score",
    ]
    available_snapshot_cols = [c for c in snapshot_cols if c in library_df.columns]
    logging.info("Writing library snapshot to %s", SNAPSHOT_PATH)
    library_df.select(available_snapshot_cols).write_parquet(SNAPSHOT_PATH)

    # =========================================================================
    # Step 1: Generate approximate similarity candidate pairs
    # =========================================================================
    # Why: GPU-batched approximate similarity uses binned sparse matrices to quickly
    # identify candidate pairs above a lower threshold. This is much faster than exact
    # computation but less accurate due to binning artifacts.
    #
    # Output: `idx_left`, `idx_right`, `similarity` columns written to parquet

    approx_cfg = GPUApproximateConfig(
        # Binning parameters
        upper_mass_bound=1000.0,
        bin_size=0.001,
        ms2_tolerance_ppm=5.0,
        intensity_power=0.5,
        weight_col="info_scores",
        weight_power=1.0,
        approx_threshold=APPROX_THRESHOLD,
        # Comparison mode
        comparison_mode="self",  # Self-comparison with upper triangular optimization
        # Memory management
        target_gpu_mem_ratio=0.1,  # Use 10% of free GPU memory (conservative)
        safety_factor=0.5,  # Additional safety margin for memory estimation
        write_buffer_batches=1000,  # Flush to parquet every 1000 GPU batches
        # Column names (match the library schema)
        spectrum_id_col="idx",
        mz_col="cleaned_normalized_mz",
        intensity_col="cleaned_normalized_intensity",
    )

    logging.info("Running approximate similarity computation on GPU...")
    t_start = perf_counter()

    # Run approximate similarity and write directly to parquet
    # Why: Writing to file allows us to stream the results through the exact computation
    # stage without holding everything in memory.
    batched_approximate_similarity_gpu(
        library_lf,
        config=approx_cfg,
        output_path=APPROX_PAIRS_PATH,
        logger=logging.getLogger(__name__),
    )

    t_end = perf_counter()
    logging.info(f"Approximate similarity complete in {t_end - t_start:.3f}s")

    # =========================================================================
    # Step 2: Compute exact similarity on candidate pairs (CPU streaming)
    # =========================================================================
    # Why: Exact similarity uses unbinned m/z arrays and precise peak matching with
    # tolerance windows. This is slower but more accurate. We stream from the approximate
    # results file to avoid loading everything into memory.
    #
    # Output: `idx`, `idx_right`, `mol_idx`, `mol_idx_right`, `approx_similarity`, `dotprod_similarity`

    logging.info("Computing exact similarity on candidates (CPU streaming)...")
    t_start = perf_counter()

    # Load library
    library_lf = pl.scan_parquet(str(LIBRARY_PATH))
    library_lf = library_lf.rename(
        {
            "precursor_formula_array": "precursor_formula",
            "cleaned_fragment_formulas": "fragment_formulas",
        }
    ).with_columns(
        pl.struct("precursor_formula", "fragment_formulas")
        .spectral_info.spectral_info_score_per_fragment(
            distance_metric="l2", ignore_hydrogens=True
        )
        .alias("info_scores")
    )

    # Load approximate pairs from file (lazy)
    approx_pairs_lf = pl.scan_parquet(str(APPROX_PAIRS_PATH))

    # Rename `similarity` to `approx_similarity` for clarity in final output
    approx_pairs_lf = approx_pairs_lf.rename({"similarity": "approx_similarity"})

    logging.info(f"Loaded approximate pairs from {APPROX_PAIRS_PATH}")

    # Join left and right spectra
    # Note: 'idx_left' and 'idx_right' in pairs now correspond to 'idx' in library
    pairs_with_spectra = (
        approx_pairs_lf.join(
            library_lf,
            left_on="idx_left",
            right_on="idx",
        )
        .rename(
            {
                "cleaned_normalized_mz": "mz1",
                "cleaned_normalized_intensity": "intensities1",
                "precursor_mz": "precursor_mz1",
                "info_scores": "weights1",
            }
        )
        .join(
            library_lf,
            left_on="idx_right",
            right_on="idx",
            suffix="_right",
        )
        .rename(
            {
                "cleaned_normalized_mz": "mz2",
                "cleaned_normalized_intensity": "intensities2",
                "precursor_mz": "precursor_mz2",
                "info_scores": "weights2",
            }
        )
    )

    # Compute exact dotprod and filter
    # Why: Use Polars spectral_similarity plugin for CPU-based exact cosine similarity.
    # This is slower than GPU but simpler and doesn't require complex batching logic.
    pairs_exact = (
        pairs_with_spectra.with_columns(
            pl.struct(
                mz1=pl.col("mz1"),
                intensities1=pl.col("intensities1"),
                weights1=pl.col("weights1"),
                mz2=pl.col("mz2"),
                intensities2=pl.col("intensities2"),
                weights2=pl.col("weights2"),
                precursor_mz1=pl.col("precursor_mz1"),
                precursor_mz2=pl.col("precursor_mz2"),
            )
            .spectral_similarity.dotprod_similarity(
                ms2_tolerance_in_ppm=5.0,
                clean_spectra_first=False,
                ignore_precursor=True,
            )
            .alias("dotprod_similarity")
        )
        .filter(pl.col("dotprod_similarity") >= EXACT_THRESHOLD)
        .select(
            pl.col("idx_left").alias("idx"),
            pl.col("idx_right").alias("idx_right"),
            pl.col("mol_idx").alias("mol_idx"),
            pl.col("mol_idx_right").alias("mol_idx_right"),
            "approx_similarity",
            "dotprod_similarity",
        )
    )

    # Collect and write final results
    # Why: Streaming sink for memory efficiency
    logging.info("Streaming exact results to final output...")
    pairs_exact.sink_parquet(PAIRS_PATH, maintain_order=False)

    t_end = perf_counter()
    logging.info(f"Exact similarity complete in {t_end - t_start:.3f}s")

    # Read final count (lightweight, just metadata)
    final_count = pl.scan_parquet(PAIRS_PATH).select(pl.len()).collect().item()
    logging.info(f"Final results: {final_count} pairs written to {PAIRS_PATH}")

    # =========================================================================
    # Step 3: Compute Tanimoto similarities
    # =========================================================================
    # Why: Molecular Tanimoto similarity is needed for downstream analysis and
    # plotting. We join the pairs with library SMILES and compute fingerprints.

    logging.info("Computing Tanimoto similarities...")
    t_tanimoto_start = perf_counter()

    compute_and_save_tanimoto_scores(
        input_parquet_path=PAIRS_PATH,
        output_path=PAIRS_WITH_TANIMOTO_PATH,
        left_library_parquet_path=SNAPSHOT_PATH,
        right_library_parquet_path=None,
        batch_size=100_000,
    )

    t_tanimoto_end = perf_counter()
    logging.info(
        f"Tanimoto computation complete in {t_tanimoto_end - t_tanimoto_start:.3f}s"
    )

    # =========================================================================
    # Step 4: Generate plots
    # =========================================================================
    # Why: Produce the standard set of similarity-vs-information figures from the
    # freshly computed pairs and library snapshot.

    logging.info("Generating plots...")
    t_plot_start = perf_counter()

    from plot_similarity_vs_info import (  # type: ignore
        InfoMetric,
        SimilarityVsInfoConfig,
        plot_heatmap_avg_tanimoto_vs_info_and_dotprod,
        run_global_line_plots,
        run_per_molecule_analysis,
    )

    plot_output_dir = PAIRS_PATH.parent / f"sim_vs_info_analysis_{PAIRS_PATH.stem.split('_')[-1]}"
    FULL_LIBRARY_PATH = PAIRS_PATH.with_suffix(".left_library_full.parquet")
    cfg = SimilarityVsInfoConfig(
        pairs_parquet_path=PAIRS_WITH_TANIMOTO_PATH,
        left_library_parquet_path=SNAPSHOT_PATH,
        right_library_parquet_path=None,
        left_library_full_parquet_path=FULL_LIBRARY_PATH,
        right_library_full_parquet_path=None,
        info_metric=InfoMetric.SPECTRAL_INFORMATION,
        tanimoto_col="tanimoto_similarity",
        left_idx_col="idx",
        right_idx_col="idx_right",
        left_mol_col="mol_idx",
        right_mol_col="mol_idx_right",
        output_dir=plot_output_dir,
        dotprod_thresholds=(0.8, 0.9),
        dotprod_bin_size=0.1,
        use_avg_info=True,
    )

    run_per_molecule_analysis(cfg)
    run_global_line_plots(cfg)
    plot_heatmap_avg_tanimoto_vs_info_and_dotprod(cfg)

    t_plot_end = perf_counter()
    logging.info(f"Plotting complete in {t_plot_end - t_plot_start:.3f}s")

    total_time = t_plot_end - t_approx_start
    logging.info(
        f"Total pipeline time: {total_time:.3f}s "
        f"(approx: {t_approx_end - t_approx_start:.3f}s, "
        f"exact: {t_exact_end - t_exact_start:.3f}s, "
        f"tanimoto: {t_tanimoto_end - t_tanimoto_start:.3f}s, "
        f"plots: {t_plot_end - t_plot_start:.3f}s)"
    )

    # Get unique molecules with their SMILES
    unique_mols = (
        library_with_mol_idx
        .filter(
            pl.col("smiles").is_not_null() & pl.col("mol_idx").is_not_null()
        )
        .select(["mol_idx", "smiles"])
        .unique(subset=["mol_idx"])
        .sort("mol_idx")
        .collect()
    )

    # Create lookup from mol_idx to smiles
    mol_idx_to_smiles = dict(zip(
        unique_mols["mol_idx"].to_list(),
        unique_mols["smiles"].to_list()
    ))

    # Build spectrum to molecule mapping
    spectrum_to_mol = (
        library_with_mol_idx
        .select(["msp_index", "mol_idx"])
        .collect()
    )
    spec_idx_to_mol_idx = dict(zip(
        spectrum_to_mol["msp_index"].to_list(),
        spectrum_to_mol["mol_idx"].to_list()
    ))

    # Open parquet file for chunked reading with pyarrow
    # Why: pyarrow supports true chunked reading without loading entire file
    parquet_file = pq.ParquetFile(str(PAIRS_PATH))
    total_rows = parquet_file.metadata.num_rows
    logging.info(f"Total pairs to process: {total_rows:,}")

    # Process in chunks and write incrementally
    n_chunks = (total_rows + config.chunk_size - 1) // config.chunk_size
    logging.info(f"Processing in {n_chunks} chunks...")

    # Remove output file if exists
    if PAIRS_WITH_TANIMOTO_PATH.exists():
        PAIRS_WITH_TANIMOTO_PATH.unlink()

    total_processed = 0
    chunk_idx = 0

    # Iterate over batches using pyarrow
    for batch in parquet_file.iter_batches(batch_size=config.chunk_size):
        chunk_idx += 1
        chunk_start_time = perf_counter()

        logging.info(f"Processing chunk {chunk_idx}/{n_chunks} (batch size: {len(batch):,})...")

        # Convert batch to polars DataFrame
        chunk = pl.from_arrow(batch)

        if len(chunk) == 0:
            break

        # Add mol_idx columns using the pre-built mapping
        chunk = chunk.with_columns([
            pl.col("idx_left").replace_strict(spec_idx_to_mol_idx, default=None).alias("mol_idx_left"),
            pl.col("idx_right").replace_strict(spec_idx_to_mol_idx, default=None).alias("mol_idx_right"),
        ])

        # Get unique molecule indices in this chunk
        mol_idx_left_list = chunk["mol_idx_left"].to_numpy()
        mol_idx_right_list = chunk["mol_idx_right"].to_numpy()
        unique_mol_indices = np.unique(np.concatenate([mol_idx_left_list, mol_idx_right_list]))
        # Filter out None and NaN values (Polars nulls become np.nan when converted to numpy)
        unique_mol_indices = unique_mol_indices[~np.isnan(unique_mol_indices.astype(float))]

        if len(unique_mol_indices) == 0:
            # No valid molecules in this chunk, add NaN column
            chunk = chunk.with_columns(pl.lit(np.nan).alias("tanimoto_similarity"))
        else:
            # Get SMILES for unique molecules
            chunk_smiles = [mol_idx_to_smiles.get(int(idx), "") for idx in unique_mol_indices]

            # Create mapping from mol_idx to position in chunk_smiles
            mol_idx_to_pos = {int(idx): i for i, idx in enumerate(unique_mol_indices)}

            # Map pairs to positions in chunk_smiles
            pairs_indices = np.array([
                [mol_idx_to_pos.get(int(left), -1), mol_idx_to_pos.get(int(right), -1)]
                for left, right in zip(mol_idx_left_list, mol_idx_right_list)
            ])

            # Compute Tanimoto for these pairs
            similarities = _compute_tanimoto_for_pairs(
                chunk_smiles,
                pairs_indices,
                config.fp_params,
            )

            # Add similarities to chunk
            chunk = chunk.with_columns(pl.Series("tanimoto_similarity", similarities))

        # Filter out rows with null tanimoto similarity
        chunk = chunk.filter(pl.col("tanimoto_similarity").is_not_null())

        # Keep mol_idx columns for downstream analysis (plot_similarity_vs_info.py)

        # Write chunk to output
        if chunk_idx == 1:
            chunk.write_parquet(PAIRS_WITH_TANIMOTO_PATH)
        else:
            # Append mode: read existing, concat, write
            existing = pl.read_parquet(PAIRS_WITH_TANIMOTO_PATH)
            combined = pl.concat([existing, chunk])
            combined.write_parquet(PAIRS_WITH_TANIMOTO_PATH)

        chunk_time = perf_counter() - chunk_start_time
        total_processed += len(chunk)
        logging.info(f"Chunk {chunk_idx} complete: {len(chunk):,} rows in {chunk_time:.2f}s")

    t_end = perf_counter()
    logging.info(f"Tanimoto similarity computation complete: {total_processed:,} pairs in {t_end - t_start:.3f}s")

    return t_end - t_start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute spectral and molecular similarities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python similarity_vs_info.py --approx          # Run only approximate similarity
  python similarity_vs_info.py --exact           # Run only exact similarity
  python similarity_vs_info.py --tanimoto        # Run only Tanimoto similarity
  python similarity_vs_info.py --approx --exact --tanimoto  # Run all steps
        """,
    )
    parser.add_argument(
        "--approx",
        action="store_true",
        help="Run approximate similarity computation (GPU). Requires data.parquet."
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Run exact similarity computation (CPU). Requires data_pairs_260311.approx.parquet."
    )
    parser.add_argument(
        "--tanimoto",
        action="store_true",
        help="Run Tanimoto similarity computation (GPU). Requires data_pairs_260311.parquet."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000_000,
        help="Number of rows to process per chunk for Tanimoto computation (default: 5,000,000)."
    )

    args = parser.parse_args()

    # If no arguments provided, print usage explanation
    if not (args.approx or args.exact or args.tanimoto):
        print("""
Usage: python similarity_vs_info.py [--approx] [--exact] [--tanimoto]

This script computes spectral and molecular similarities in three steps:

  --approx    Run approximate spectral similarity on GPU.
              Input:  data.parquet
              Output: data_pairs_260311.approx.parquet

  --exact     Run exact spectral similarity on CPU using approximate candidates.
              Input:  data_pairs_260311.approx.parquet, data.parquet
              Output: data_pairs_260311.parquet

  --tanimoto  Run Tanimoto molecular similarity on GPU using Morgan fingerprints.
              Input:  data_pairs_260311.parquet, data.parquet (for SMILES)
              Output: data_pairs_260311_with_tanimoto.parquet

  --chunk-size N  Number of rows per chunk for Tanimoto computation (default: 5000000).

You can run multiple steps at once, e.g.: --approx --exact --tanimoto
Run with -h for more details.
        """.strip())
        sys.exit(0)

    # Setup logging
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    total_start = perf_counter()
    timings = {}

    try:
        if args.approx:
            logging.info("=" * 60)
            logging.info("Starting Step 1: Approximate Similarity")
            logging.info("=" * 60)
            timings["approx"] = run_approximate_similarity()

        if args.exact:
            logging.info("=" * 60)
            logging.info("Starting Step 2: Exact Similarity")
            logging.info("=" * 60)
            timings["exact"] = run_exact_similarity()

        if args.tanimoto:
            logging.info("=" * 60)
            logging.info("Starting Step 3: Tanimoto Similarity")
            logging.info("=" * 60)
            tanimoto_config = TanimotoConfig(chunk_size=args.chunk_size)
            timings["tanimoto"] = run_tanimoto_similarity(config=tanimoto_config)

    except MissingFileError as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.exception("Unexpected error occurred")
        sys.exit(1)

    total_end = perf_counter()

    # Print summary
    logging.info("=" * 60)
    logging.info("Computation Summary")
    logging.info("=" * 60)
    for step, duration in timings.items():
        logging.info(f"{step.capitalize()}: {duration:.3f}s")
    logging.info(f"Total: {total_end - total_start:.3f}s")
