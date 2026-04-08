import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
import polars as pl

from parallel_rdkit.fingerprint import FingerprintParams, get_fp_list
from parallel_rdkit.mol import sanitize_smiles

import pyarrow.parquet as pq


@dataclass
class TanimotoConfig:
    """Configuration for Tanimoto similarity computation.
    
    Why: Chunked processing allows memory-efficient computation on large pair files
    without loading everything into memory at once.
    """
    chunk_size: int = 5_000_000  # Default 5M rows per chunk
    fp_params: FingerprintParams = None
    
    def __post_init__(self):
        if self.fp_params is None:
            self.fp_params = FingerprintParams(fp_type="morgan", radius=2, fpSize=2048)


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


def _compute_tanimoto_for_pairs(
    smiles_list: List[str],
    pairs_indices: np.ndarray,
    fp_params: FingerprintParams,
) -> np.ndarray:
    """Compute Tanimoto similarity for specific pairs of molecules.
    
    Why: Instead of computing full all-vs-all matrix, we only compute similarities
    for the specific molecule pairs we need, using in-memory fingerprint generation.
    
    Args:
        smiles_list: List of SMILES strings for all molecules in chunk
        pairs_indices: Array of shape (n_pairs, 2) with [idx_left, idx_right] pairs
        fp_params: Fingerprint parameters
        
    Returns:
        Array of Tanimoto similarities for each pair
    """
    # Generate fingerprints for all molecules in chunk
    fps, valid_mask = get_fp_list(smiles_list, fp_params, return_numpy=True)
    
    # fps is 2D array (n_mols, fp_size) if return_numpy=True
    # valid_mask is 1D boolean array
    
    # Compute Tanimoto for each pair
    # Tanimoto = (A & B).sum() / (A | B).sum()
    n_pairs = len(pairs_indices)
    similarities = np.zeros(n_pairs, dtype=np.float32)
    
    for i, (idx_left, idx_right) in enumerate(pairs_indices):
        if not valid_mask[idx_left] or not valid_mask[idx_right]:
            similarities[i] = np.nan
            continue
            
        fp_left = fps[idx_left]
        fp_right = fps[idx_right]
        
        # Compute intersection and union
        intersection = np.logical_and(fp_left, fp_right).sum()
        union = np.logical_or(fp_left, fp_right).sum()
        
        if union > 0:
            similarities[i] = intersection / union
        else:
            similarities[i] = 0.0
    
    return similarities


def run_tanimoto_similarity(config: Optional[TanimotoConfig] = None):
    """Step 3: Compute Tanimoto similarity on GPU for molecule pairs.
    
    This implementation reads pairs in chunks, extracts the unique molecules needed
    for each chunk, and computes Tanimoto similarity only for those specific pairs
    using in-memory fingerprint generation.
    
    Output includes spectrum indices (idx_left, idx_right), molecule indices
    (mol_idx_left, mol_idx_right), dot-product similarity, and Tanimoto similarity.
    The molecule indices enable downstream analysis without reconstruction.
    
    Args:
        config: TanimotoConfig with chunk_size and fp_params. Uses defaults if None.
    """
    if config is None:
        config = TanimotoConfig()
    
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

    logging.info("Computing Tanimoto similarity using chunked in-memory processing...")
    logging.info(f"Chunk size: {config.chunk_size:,} rows")
    t_start = perf_counter()

    # Load library and create mol_idx mapping
    # Why: We need to map spectrum indices to molecule indices
    library_lf = pl.scan_parquet(str(LIBRARY_PATH))
    
    library_with_mol_idx = library_lf.with_columns(
        pl.col("base_inchikey").rank(method="dense").cast(pl.Int64).alias("mol_idx")
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
