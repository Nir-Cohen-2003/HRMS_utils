import argparse
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from parallel_rdkit.matrix_tanimoto import calculate_tanimoto_matrix

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

# Import hrms_core to register the spectral_similarity plugin
import hrms_utils.hrms_core  # noqa: F401

os.environ["RUST_BACKTRACE"] = "full"


class MissingFileError(FileNotFoundError):
    """Raised when a required input file is missing for a computation step."""

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


def run_approximate_similarity():
    """Step 1: Generate approximate similarity candidate pairs on GPU."""
    if not LIBRARY_PATH.exists():
        raise MissingFileError(
            f"Missing required input file for approximate similarity: {LIBRARY_PATH}\n"
            f"This file should contain the spectral library data. "
            f"Please ensure the library parquet is available at this location."
        )

    logging.info(f"Loading library from {LIBRARY_PATH}")
    library_lf = pl.scan_parquet(str(LIBRARY_PATH))

    # Calculate information scores per fragment
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

    # Configure approximate similarity computation
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
        write_buffer_batches=100,  # Flush to parquet every 100 GPU batches
        # Column names (match the library schema)
        spectrum_id_col="msp_index",
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

    return t_end - t_start


def run_exact_similarity():
    """Step 2: Compute exact similarity on candidate pairs (CPU streaming)."""
    if not APPROX_PAIRS_PATH.exists():
        raise MissingFileError(
            f"Missing required input file for exact similarity: {APPROX_PAIRS_PATH}\n"
            f"This file is produced by the approximate similarity step. "
            f"Run with --approx first to generate it."
        )

    if not LIBRARY_PATH.exists():
        raise MissingFileError(
            f"Missing required input file for exact similarity: {LIBRARY_PATH}\n"
            f"This file should contain the spectral library data. "
            f"Please ensure the library parquet is available at this location."
        )

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
    # Note: 'idx_left' and 'idx_right' in pairs correspond to 'msp_index' in library
    pairs_with_spectra = (
        approx_pairs_lf.join(
            library_lf,
            left_on="idx_left",
            right_on="msp_index",
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
            right_on="msp_index",
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
            "idx_left",
            "idx_right",
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

    return t_end - t_start


def run_tanimoto_similarity():
    """Step 3: Compute Tanimoto similarity on GPU for molecule pairs."""
    if not PAIRS_PATH.exists():
        raise MissingFileError(
            f"Missing required input file for Tanimoto similarity: {PAIRS_PATH}\n"
            f"This file is produced by the exact similarity step. "
            f"Run with --exact first to generate it."
        )

    if not LIBRARY_PATH.exists():
        raise MissingFileError(
            f"Missing required input file for Tanimoto similarity: {LIBRARY_PATH}\n"
            f"This file should contain the spectral library data with SMILES. "
            f"Please ensure the library parquet is available at this location."
        )

    logging.info("Computing Tanimoto similarity on GPU...")
    t_start = perf_counter()

    # Load the exact pairs and library
    pairs_df = pl.read_parquet(str(PAIRS_PATH))
    library_df = pl.read_parquet(str(LIBRARY_PATH))

    # Create mol_idx mapping based on base_inchikey
    # Why: base_inchikey is the canonical molecule identifier; spectra from the same
    # molecule share the same base_inchikey
    library_with_mol_idx = library_df.with_columns(
        pl.col("base_inchikey").hash(seed=42).cast(pl.Int64).alias("mol_idx")
    )

    # Get unique molecules (base_inchikey + smiles + mol_idx)
    # Why: We only need one SMILES per molecule for Tanimoto computation
    unique_mols = (
        library_with_mol_idx
        .filter(pl.col("smiles").is_not_null())
        .select(["mol_idx", "smiles", "base_inchikey"])
        .unique(subset=["mol_idx"])
        .sort("mol_idx")
    )

    smiles_list = unique_mols["smiles"].to_list()
    mol_idx_list = unique_mols["mol_idx"].to_list()

    logging.info(f"Computing Tanimoto matrix for {len(smiles_list)} unique molecules...")

    # Compute Tanimoto similarity matrix on GPU
    tanimoto_matrix = calculate_tanimoto_matrix(
        smiles=smiles_list,
        fp_radius=2,
        fp_size=2048,
        save_path=None,  # We process the matrix in memory
        log_path=None,
    )

    # Create a mapping from molecule index pair to Tanimoto similarity
    # Why: The matrix is symmetric; we only need to store upper triangular
    mol_idx_to_tanimoto = {}
    for i, mol_i in enumerate(mol_idx_list):
        for j, mol_j in enumerate(mol_idx_list):
            if i <= j:  # Upper triangular including diagonal
                mol_idx_to_tanimoto[(mol_i, mol_j)] = float(tanimoto_matrix[i, j])
                mol_idx_to_tanimoto[(mol_j, mol_i)] = float(tanimoto_matrix[i, j])

    # Join mol_idx to pairs based on spectrum indices
    # Why: idx_left and idx_right in pairs correspond to msp_index in library
    spectrum_to_mol = dict(zip(
        library_with_mol_idx["msp_index"].to_list(),
        library_with_mol_idx["mol_idx"].to_list()
    ))

    # Add mol_idx columns to pairs
    pairs_with_mol = pairs_df.with_columns([
        pl.col("idx_left").map_elements(
            lambda x: spectrum_to_mol.get(x, None),
            return_dtype=pl.Int64
        ).alias("mol_idx"),
        pl.col("idx_right").map_elements(
            lambda x: spectrum_to_mol.get(x, None),
            return_dtype=pl.Int64
        ).alias("mol_idx_right")
    ])

    # Add Tanimoto similarity by looking up molecule pairs
    pairs_with_tanimoto = pairs_with_mol.with_columns(
        pl.struct(["mol_idx", "mol_idx_right"])
        .map_elements(
            lambda row: mol_idx_to_tanimoto.get((row["mol_idx"], row["mol_idx_right"]), 0.0),
            return_dtype=pl.Float64
        )
        .alias("tanimoto_similarity")
    )

    # Write final results with Tanimoto similarity
    logging.info(f"Writing final pairs with Tanimoto to {PAIRS_WITH_TANIMOTO_PATH}")
    pairs_with_tanimoto.write_parquet(PAIRS_WITH_TANIMOTO_PATH)

    t_end = perf_counter()
    logging.info(f"Tanimoto similarity computation complete in {t_end - t_start:.3f}s")

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
            timings["tanimoto"] = run_tanimoto_similarity()

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
