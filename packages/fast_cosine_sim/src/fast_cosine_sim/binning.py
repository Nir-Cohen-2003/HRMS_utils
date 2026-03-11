"""
CPU-side preprocessing: Polars DataFrame to sparse CSR matrix.

This module handles the conversion from Polars DataFrames with list-valued
spectrum columns to scipy sparse CSR matrices suitable for GPU transfer.

Pipeline:
1. select_and_collect: Handle LazyFrame/DataFrame input
2. _flatten_spectra_to_numpy: Explode list columns to flat numpy arrays
3. _sparse_bin_flat_spectra_to_csr: Bin peaks into sparse matrix
4. sparse_bin_spectra_df_to_csr: Main entry point with optional centroiding

Why separate module:
- CPU preprocessing is independent of GPU operations
- scipy.sparse doesn't require GPU imports
- Can be tested independently
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray

from .config import (
    CSR_DATA_DTYPE_CPU,
    INDEX_DTYPE_NP,
    INDEX_DTYPE_PL,
    GPUApproximateConfig,
)


def select_and_collect(
    frame: pl.DataFrame | pl.LazyFrame, config: GPUApproximateConfig
) -> pl.DataFrame:
    """
    Collect a LazyFrame if needed, otherwise return DataFrame with only necessary columns.

    Why: Accept both DataFrame and LazyFrame inputs for flexibility.
    Selecting only needed columns reduces memory before processing.
    """
    cols = [config.mz_col, config.intensity_col, config.spectrum_id_col]
    if config.weight_col is not None:
        cols.append(config.weight_col)
    frame = frame.select(cols)
    return (
        frame.collect(engine="streaming") if isinstance(frame, pl.LazyFrame) else frame
    )


def _flatten_spectra_to_numpy(
    df: pl.DataFrame, mz_col: str, int_col: str, weight_col: str | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.int32], NDArray[np.float64] | None, int]:
    """
    Flatten list-valued spectrum columns from DataFrame into NumPy arrays.

    Why: Polars list columns are efficient for storage but need to be exploded
    for binning. This function explodes and converts to numpy arrays in one pass.

    Args:
        df: DataFrame with list columns
        mz_col: Name of m/z column (list of floats)
        int_col: Name of intensity column (list of floats)
        weight_col: Optional name of weight column (list of floats)

    Returns:
        (flat_mzs, flat_ints, spec_idx, flat_weights, n_spec)
        - flat_mzs: np.ndarray[np.float64] of all m/z values
        - flat_ints: np.ndarray[np.float32] of all intensities
        - spec_idx: np.ndarray[np.int32] mapping each peak to its spectrum index
        - flat_weights: np.ndarray[np.float64] of all weights, or None
        - n_spec: number of spectra
    """
    n_spec = len(df)
    if n_spec == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float64) if weight_col else None,
            0,
        )

    # Add temporary row index and explode
    df_idx = df.with_row_index("__spec_idx")
    explode_cols = [mz_col, int_col]
    if weight_col:
        explode_cols.append(weight_col)
    exploded = df_idx.explode(explode_cols)

    if len(exploded) == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float64) if weight_col else None,
            n_spec,
        )

    # Cast and extract
    cast_cols = [
        pl.col(mz_col).cast(pl.Float32),
        pl.col(int_col).cast(pl.Float32),
        pl.col("__spec_idx").cast(INDEX_DTYPE_PL),
    ]
    if weight_col:
        cast_cols.append(pl.col(weight_col).cast(pl.Float32))
    exploded = exploded.with_columns(cast_cols)

    flat_mzs = exploded.get_column(mz_col).to_numpy()
    flat_ints = exploded.get_column(int_col).to_numpy()
    spec_idx = exploded.get_column("__spec_idx").to_numpy()
    flat_weights = exploded.get_column(weight_col).to_numpy() if weight_col else None

    return flat_mzs, flat_ints, spec_idx, flat_weights, n_spec


def _sparse_bin_flat_spectra_to_csr(
    flat_mzs: NDArray[np.float64],
    flat_ints: NDArray[np.float32],
    spec_idx: NDArray[np.int32],
    flat_weights: NDArray[np.float64] | None,
    n_spec: int,
    upper_bound: float,
    intensity_power: float,
    weight_power: float,
    bin_size: float,
) -> sp.csr_matrix:
    """
    Turn flattened arrays into a sparse CSR matrix (n_spec, nbins).

    Why: Binning reduces dimensionality and enables fast sparse matmul for
    approximate similarity. COO construction with duplicates summed is the
    most efficient path in SciPy.

    Binning uses: bin = np.rint(mz / bin_size)
    Duplicates are summed via COO -> CSR conversion.

    Args:
        flat_mzs: All m/z values
        flat_ints: All intensity values
        spec_idx: Spectrum index for each peak
        flat_weights: Optional array of weight values
        n_spec: Total number of spectra
        upper_bound: Maximum m/z
        intensity_power: Power to apply to intensities
        weight_power: Power to apply to weights
        bin_size: Bin width

    Returns:
        scipy.sparse.csr_matrix of shape (n_spec, nbins)
    """
    nbins = int(np.floor(upper_bound / float(bin_size))) + 1

    if n_spec == 0 or flat_mzs.size == 0 or flat_ints.size == 0:
        return sp.csr_matrix((n_spec, nbins), dtype=CSR_DATA_DTYPE_CPU)

    # Bin m/z values
    mass_bins = np.rint(flat_mzs / float(bin_size)).astype(np.int32)

    # Filter valid bins
    valid_mask = (mass_bins >= 0) & (mass_bins < nbins) & (flat_ints > 0)
    if not np.any(valid_mask):
        return sp.csr_matrix((n_spec, nbins), dtype=CSR_DATA_DTYPE_CPU)

    mass_bins = mass_bins[valid_mask].astype(np.int32)
    spec_idx = spec_idx[valid_mask].astype(np.int32)
    
    weights = np.asarray(flat_ints[valid_mask], dtype=np.float32) ** float(
        intensity_power
    )
    if flat_weights is not None:
        weight_vals = np.asarray(flat_weights[valid_mask], dtype=np.float32) ** float(weight_power)
        weights = weights * weight_vals

    # Build COO matrix (duplicates are summed in tocsr())
    coo = sp.coo_matrix(
        (weights.astype(CSR_DATA_DTYPE_CPU, copy=False), (spec_idx, mass_bins)),
        shape=(n_spec, nbins),
        dtype=CSR_DATA_DTYPE_CPU,
    )

    # Convert to CSR (SciPy sums duplicates automatically)
    return sp.csr_matrix(coo.tocsr())


def sparse_bin_spectra_df_to_csr(
    df: pl.DataFrame,
    mz_col: str,
    int_col: str,
    upper_bound: float,
    intensity_power: float,
    bin_size: float,
    *,
    weight_col: str | None = None,
    weight_power: float = 0.0,
    apply_centroiding: bool = False,
    tolerance_ppm: float = 10.0,
    mass_tolerance_cutoff_mz: float = 200.0,
) -> sp.csr_matrix:
    """
    Explode list-valued spectra and bin into a sparse CSR matrix.

    Why: This is the main entry point for converting a Polars DataFrame of
    spectra into a binned sparse matrix ready for GPU transfer.

    Optionally applies centroiding before binning to prevent one-to-many
    peak matching (which causes similarities > 1.0).

    Args:
        df: DataFrame with list columns
        mz_col: Name of m/z column
        int_col: Name of intensity column
        upper_bound: Maximum m/z
        intensity_power: Power to apply to intensities
        bin_size: Bin width
        weight_col: Optional name of weight column
        weight_power: Power to apply to weights
        apply_centroiding: If True, centroid peaks before binning
        tolerance_ppm: PPM tolerance for centroiding (if enabled)
        mass_tolerance_cutoff_mz: m/z cutoff for centroiding (if enabled)

    Returns:
        scipy.sparse.csr_matrix of shape (len(df), nbins)
    """
    nbins = int(np.floor(upper_bound / float(bin_size))) + 1

    flat_mzs, flat_ints, spec_idx, flat_weights, n_spec = _flatten_spectra_to_numpy(
        df, mz_col, int_col, weight_col
    )

    if n_spec == 0:
        return sp.csr_matrix((0, nbins), dtype=CSR_DATA_DTYPE_CPU)
    if flat_mzs.size == 0 or flat_ints.size == 0:
        return sp.csr_matrix((n_spec, nbins), dtype=CSR_DATA_DTYPE_CPU)

    # Apply centroiding if enabled
    # Why: Prevents one-to-many peak matching which causes similarities > 1.0
    if apply_centroiding:
        from .centroiding import centroid_flat_spectra

        flat_mzs, flat_ints, spec_idx, flat_weights, n_spec = centroid_flat_spectra(
            flat_mzs,
            flat_ints,
            spec_idx,
            flat_weights,
            n_spec,
            tolerance_ppm=tolerance_ppm,
            mass_tolerance_cutoff_mz=mass_tolerance_cutoff_mz,
        )

    return _sparse_bin_flat_spectra_to_csr(
        flat_mzs, flat_ints, spec_idx, flat_weights, n_spec, upper_bound, intensity_power, weight_power, bin_size
    )
