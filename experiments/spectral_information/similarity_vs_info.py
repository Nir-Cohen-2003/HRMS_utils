import logging
import os
import sys
from pathlib import Path
from time import perf_counter

import polars as pl

# Add the src directory to sys.path so hrms_utils can be imported
sys.path.append(str(Path(__file__).parents[2] / "src"))

# Use the packaged fast_cosine_sim implementation (packages/fast_cosine_sim) instead of
# experiments-local modules.
#
# This experiment should be runnable without installing the wheel/conda package, so we
# add the local packages source tree (`packages/fast_cosine_sim/src`) to sys.path.
_FAST_COSINE_SIM_SRC = (
    Path(__file__).parents[2] / "packages" / "fast_cosine_sim" / "src"
)
assert _FAST_COSINE_SIM_SRC.exists(), (
    f"Expected fast_cosine_sim source tree at {_FAST_COSINE_SIM_SRC}. "
    "If you moved the package, update this path."
)
sys.path.insert(0, str(_FAST_COSINE_SIM_SRC))

from fast_cosine_sim import (  # noqa: E402
    GPUApproximateConfig,
    batched_approximate_similarity_gpu,
)

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

    # Why: Remove existing log file to start fresh and avoid appending to old logs
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    # Setup logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    # Threshold configuration
    # Why: Approximate threshold is lower to ensure we don't miss candidates that might
    # pass the exact threshold after proper peak matching. The gap accounts for binning
    # artifacts and tolerance window expansion effects.
    EXACT_THRESHOLD = 0.5
    APPROX_THRESHOLD = 0.35

    logging.info(
        f"Starting similarity calculation. Approx threshold: {APPROX_THRESHOLD}, Exact threshold: {EXACT_THRESHOLD}"
    )

    # =========================================================================
    # Step 0: Load library, assign indices, and recompute info scores
    # =========================================================================
    # Why: Downstream tools (tanimoto, plotting) expect integer idx/mol_idx and
    # correct spectral_information_score values. We recompute the score here using
    # the Rust plugin to ensure correctness.

    logging.info(f"Loading library from {LIBRARY_PATH}")
    library_lf = pl.scan_parquet(str(LIBRARY_PATH))

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

    # Persist minimal library snapshot for downstream joins (idx/mol_idx + metadata)
    snapshot_cols = [
        "idx",
        "mol_idx",
        "base_inchikey",
        "ion_mode",
        "smiles",
        "precursor_mz",
        "spectral_information_score",
    ]
    available_snapshot_cols = [
        c for c in snapshot_cols if c in library_lf.collect_schema().names()
    ]
    logging.info("Writing library snapshot to %s", SNAPSHOT_PATH)
    library_lf.select(available_snapshot_cols).collect(engine="streaming").write_parquet(
        SNAPSHOT_PATH
    )

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
    t_approx_start = perf_counter()

    # Run approximate similarity and write directly to parquet
    # Why: Writing to file allows us to stream the results through the exact computation
    # stage without holding everything in memory.
    result_lf = batched_approximate_similarity_gpu(
        library_lf,
        config=approx_cfg,
        output_path=APPROX_PAIRS_PATH,
        logger=logging.getLogger(__name__),
    )

    t_approx_end = perf_counter()
    logging.info(
        f"Approximate similarity complete in {t_approx_end - t_approx_start:.3f}s"
    )

    # =========================================================================
    # Step 2: Compute exact similarity on candidate pairs (CPU streaming)
    # =========================================================================
    # Why: Exact similarity uses unbinned m/z arrays and precise peak matching with
    # tolerance windows. This is slower but more accurate. We stream from the approximate
    # results file to avoid loading everything into memory.
    #
    # Output: `idx`, `idx_right`, `mol_idx`, `mol_idx_right`, `approx_similarity`, `dotprod_similarity`

    logging.info("Computing exact similarity on candidates (CPU streaming)...")
    t_exact_start = perf_counter()

    # Load approximate pairs from file (lazy)
    # Note: The package outputs columns as `idx_left`, `idx_right`, `similarity`
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
                mz2=pl.col("mz2"),
                intensities2=pl.col("intensities2"),
                precursor_mz1=pl.col("precursor_mz1"),
                precursor_mz2=pl.col("precursor_mz2"),
            )
            .spectral_similarity.dotprod_similarity(
                ms2_tolerance_in_ppm=approx_cfg.ms2_tolerance_ppm,
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

    t_exact_end = perf_counter()
    logging.info(f"Exact similarity complete in {t_exact_end - t_exact_start:.3f}s")

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
    cfg = SimilarityVsInfoConfig(
        pairs_parquet_path=PAIRS_WITH_TANIMOTO_PATH,
        left_library_parquet_path=SNAPSHOT_PATH,
        right_library_parquet_path=None,
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
