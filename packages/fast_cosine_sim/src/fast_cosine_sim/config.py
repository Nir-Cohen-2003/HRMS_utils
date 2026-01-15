from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

# Why: below this m/z cutoff, ppm-based tolerances become unrealistically small; the
# experiments version uses the same cutoff for adaptive expansion windows.
MASS_TOLERANCE_CUTOFF: float = 200.0


@dataclass(frozen=True, slots=True)
class ApproximateGpuDtypesConfig:
    """Dtype configuration for the approximate GPU-batched similarity stage.

    Notes:
      - `index_dtype` is used for spectrum ids in the output pairs.
      - `csr_index_dtype` is used for CSR `indices` and `indptr` (CuPy sparse generally expects int32).
      - `intensity_dtype` is the dtype for CSR `.data`.
      - `similarity_dtype` is the dtype for similarity values emitted.
    """

    index_dtype: np.dtype = np.dtype(np.int64)
    # Why: CuPy sparse is most compatible/performant with int32 indices/indptr.
    csr_index_dtype: np.dtype = np.dtype(np.int32)

    intensity_dtype: np.dtype = np.dtype(np.float32)
    similarity_dtype: np.dtype = np.dtype(np.float32)

    def __post_init__(self) -> None:
        for field_name, field_value in (
            ("index_dtype", self.index_dtype),
            ("csr_index_dtype", self.csr_index_dtype),
            ("intensity_dtype", self.intensity_dtype),
            ("similarity_dtype", self.similarity_dtype),
        ):
            object.__setattr__(self, field_name, np.dtype(field_value))

        if np.dtype(self.index_dtype).kind not in ("i", "u"):
            raise TypeError(
                f"index_dtype must be an integer dtype, got {self.index_dtype}"
            )
        if np.dtype(self.csr_index_dtype).kind not in ("i", "u"):
            raise TypeError(
                f"csr_index_dtype must be an integer dtype, got {self.csr_index_dtype}"
            )
        if np.dtype(self.intensity_dtype).kind != "f":
            raise TypeError(
                f"intensity_dtype must be a float dtype, got {self.intensity_dtype}"
            )
        if np.dtype(self.similarity_dtype).kind != "f":
            raise TypeError(
                f"similarity_dtype must be a float dtype, got {self.similarity_dtype}"
            )


@dataclass(frozen=True, slots=True)
class IntensityTransformConfig:
    """Controls intensity transformation before similarity.

    Requirement:
      - allow dictating the power to which intensity will be raised
      - special-case power==1.0 to skip extra work
      - if power==0.0 then set all (non-zero) intensities to 1.0 (presence-only)

    Semantics in this package:
      - power == 1.0 => identity (skip transform)
      - power == 0.0 => presence-only (intensity treated as 1.0 for any non-zero entry)
      - otherwise => intensity ** power
    """

    power: float = 0.5

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.power)):
            raise ValueError(f"intensity power must be finite, got {self.power}")


@dataclass(frozen=True, slots=True)
class OutputParquetConfig:
    """Optional parquet write-and-scan output settings.

    Contract:
      - If `path` is None: do not write; return in-memory `pl.DataFrame`.
      - If `path` is not None: write parquet partitions and return `pl.scan_parquet(...)`.
    """

    path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.path is not None and not isinstance(self.path, Path):
            raise TypeError(
                f"path must be a pathlib.Path or None, got {type(self.path)}"
            )


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Optional logging/profiling output.

    If `log_path` is provided, the caller can supply a logger and write profiling
    info (timings, arguments, batch stats) to this file.
    """

    log_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.log_path is not None and not isinstance(self.log_path, Path):
            raise TypeError(
                f"log_path must be a pathlib.Path or None, got {type(self.log_path)}"
            )


@dataclass(frozen=True, slots=True)
class BatchSizingConfig:
    """Controls coarse batching behavior.

    - `target_gpu_memory_usage_ratio`: fraction of free GPU memory to target.
    - `min_spectra_per_batch`: enforced lower bound for batch size (when possible).
    - `max_peaks_per_batch`: optional clamp on dynamic peak-count-based batching.
    - `flush_to_parquet_every_n_batches`: if set, enables write mode and determines flush cadence.
    """

    target_gpu_memory_usage_ratio: float = 0.1
    min_spectra_per_batch: int = 256
    max_peaks_per_batch: Optional[int] = None
    flush_to_parquet_every_n_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if not (0.0 < float(self.target_gpu_memory_usage_ratio) <= 1.0):
            raise ValueError(
                "target_gpu_memory_usage_ratio must be in (0.0, 1.0], "
                f"got {self.target_gpu_memory_usage_ratio}"
            )
        if int(self.min_spectra_per_batch) <= 0:
            raise ValueError(
                f"min_spectra_per_batch must be positive, got {self.min_spectra_per_batch}"
            )
        if self.max_peaks_per_batch is not None and int(self.max_peaks_per_batch) <= 0:
            raise ValueError(
                f"max_peaks_per_batch must be positive if provided, got {self.max_peaks_per_batch}"
            )
        if (
            self.flush_to_parquet_every_n_batches is not None
            and int(self.flush_to_parquet_every_n_batches) <= 0
        ):
            raise ValueError(
                "flush_to_parquet_every_n_batches must be positive if provided, "
                f"got {self.flush_to_parquet_every_n_batches}"
            )


@dataclass(frozen=True, slots=True)
class ApproximateGpuBatchedSimilarityConfig:
    """Top-level config for the approximate GPU-batched similarity stage only.

    This package stops at approximate candidate generation and is intentionally
    not coupled to any exact-stage computation or `hrms_utils`.

    `comparison_mode`:
      - "self": one input frame => compute upper-triangular (i < j)
      - "cross": left & right frames => compute full cross product
    """

    # binning / threshold
    upper_mass_bound: float
    bin_size: float
    approx_threshold: float

    # adaptive (mass-dependent) right-matrix expansion
    # Why: candidate generation must account for MS2 fragment tolerance by dilating the RHS
    # across neighboring bins; this mirrors the original experiments implementation.
    ms2_tolerance_ppm: Optional[float] = None
    mass_tolerance_cutoff_mz: float = MASS_TOLERANCE_CUTOFF

    # knobs
    dtypes: ApproximateGpuDtypesConfig = ApproximateGpuDtypesConfig()
    intensity: IntensityTransformConfig = IntensityTransformConfig()
    batching: BatchSizingConfig = BatchSizingConfig()
    output_parquet: OutputParquetConfig = OutputParquetConfig()
    logging: LoggingConfig = LoggingConfig()

    comparison_mode: Literal["self", "cross"] = "self"

    # input schema
    spectrum_id_column: str = "idx"
    mz_column: str = "mz"
    intensity_column: str = "intensity"

    def __post_init__(self) -> None:
        assert self.upper_mass_bound > 0.0, "upper_mass_bound must be positive"
        assert self.bin_size > 0.0, "bin_size must be positive"
        assert 0.0 <= self.approx_threshold <= 1.0, "approx_threshold must be in [0, 1]"

        if self.ms2_tolerance_ppm is not None and float(self.ms2_tolerance_ppm) <= 0.0:
            raise ValueError(
                f"ms2_tolerance_ppm must be positive if provided, got {self.ms2_tolerance_ppm}"
            )
        assert float(self.mass_tolerance_cutoff_mz) > 0.0, (
            "mass_tolerance_cutoff_mz must be positive; "
            f"got {self.mass_tolerance_cutoff_mz}"
        )

        if self.comparison_mode not in ("self", "cross"):
            raise ValueError(
                f"comparison_mode must be 'self' or 'cross', got {self.comparison_mode}"
            )

        for name, value in (
            ("spectrum_id_column", self.spectrum_id_column),
            ("mz_column", self.mz_column),
            ("intensity_column", self.intensity_column),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
