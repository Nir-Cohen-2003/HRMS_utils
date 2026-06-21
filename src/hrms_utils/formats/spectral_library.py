import logging
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TextIO, TypeVar, Union, cast

import numpy as np
import polars as pl
from parallel_rdkit import (
    inchi_to_smiles_parallel,
    msready_inchi_inchikey_parallel,
    smiles_to_formula,  # type: ignore[attr-defined]
    smiles_to_inchikey_parallel,
)

from ..formula_annotation.element_table import (
    ADDUCT_MASSES,
    ELEMENT_MASSES,
    ELEMENT_SYMBOLS,
)
from ..formula_annotation.utils import (
    format_formula_string_to_array,
    formula_to_array,
    num_elements,
)
from ..hrms_core import *
from . import mgf, nist_mspec

polarsFrame = TypeVar("polarsFrame", pl.DataFrame, pl.LazyFrame)

# Default Polars execution engine to streaming for the pipeline in this module.
pl.Config.set_engine_affinity("streaming")


# Final output columns emitted by both ``process_single_file`` and
# ``process_spectral_library``. Kept as a module-level constant so the two
# entry points stay in lock-step.
_OUTPUT_COLUMNS: List[str] = [
    "name",
    # "db_id",
    "instrument_type",
    "instrument",
    "ionization",
    "ion_mode",
    "mslevel",
    "collision_energy_NCE",
    "collision_energy_ev",
    "collision_energy_list",
    "multiple_collision_energies",
    "collision_energy_mean",
    "cas",
    "inchikey",
    "base_inchikey",
    "smiles",
    "inchi",
    "is_orbitrap",
    "is_TOF",
    "is_ESI",
    "is_LC",
    "precursor_type",
    "precursor_mz",
    "molecular_formula",
    "molecular_formula_array",
    "precursor_formula_array",
    "clean_precursor",
    "exact_mass",
    "raw_spectrum_mz",
    "raw_spectrum_intensity",
    "cleaned_normalized_mz",
    "cleaned_normalized_intensity",
    "cleaned_fragment_formulas",
    "cleaned_fragment_formulas_str",
    "cleaned_fragment_errors_ppm",
    "explained_intensity",
    "molecular_ion_intensity",
    "spectral_information_score",
    "spectral_information_score_with_hydrogens",
    "source_file",
    "rt_seconds",
    "precursor_intensity",
    "pubchem_cid",
    "ccs",
    "charge",
    "adduct",
    "precursor_im",
    "sample_inlet",
    "column_type",
    "spectral_entropy",
    "num_explained_peaks",
    "explained_intensity_raw",
    "enamine_catalog_id",
    "iupac_name",
    "adduct_formula",
]


def _select_output_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Project ``lf`` onto the canonical output schema used by both pipelines."""
    existing_cols = lf.collect_schema().names()
    cols_to_select = [c for c in _OUTPUT_COLUMNS if c in existing_cols]
    # Always include MSn columns when present so callers that requested them
    # still receive them.
    msn_cols = [c for c in existing_cols if c.startswith("msn_")]
    return lf.select(cols_to_select + msn_cols)


def _deduce_output_path(files: list[Path | str]) -> Path:
    """Deduce the default output Parquet path from the input file list.

    - One file  -> ``<file>.parquet``
    - Multiple  -> ``<parent_dir>/<parent_dir>.parquet``
    """
    first = Path(files[0])
    if len(files) == 1:
        return first.with_suffix(".parquet")
    return first.parent / f"{first.parent.name}.parquet"


def _log(logger: logging.Logger | TextIO | None, msg: str) -> None:
    """Emit ``msg`` to ``logger`` with a timestamp, flushing immediately.

    Accepts a :class:`logging.Logger` (configured by the CLI) or a file-like
    object such as ``sys.stdout`` for backwards compatibility.
    """
    if logger is None:
        return
    timestamp = datetime.now().isoformat(timespec="seconds")
    if isinstance(logger, logging.Logger):
        logger.info(msg)
    else:
        logger.write(f"{timestamp} {msg}\n")
        try:
            logger.flush()
        except Exception:
            pass


def _collect_logged(lf: pl.LazyFrame, logger: logging.Logger | TextIO | None, label: str) -> pl.DataFrame:
    """Call ``lf.collect()`` and log ``label`` immediately before and after."""
    _log(logger, f"before {label}")
    result = lf.collect()
    _log(logger, f"after {label}")
    return result


def _run_logged(logger: logging.Logger | TextIO | None, label: str, func, *args, **kwargs):
    """Call ``func(*args, **kwargs)`` and log ``label`` immediately before and after."""
    _log(logger, f"before {label}")
    result = func(*args, **kwargs)
    _log(logger, f"after {label}")
    return result


def _get_stats_str(lf: pl.LazyFrame, prefix: str = "") -> str:
    """Return a compact statistics string for ``lf`` without materializing wide columns."""
    total = lf.select(pl.len()).collect().item()
    total_molecules = lf.select(pl.col("base_inchikey").n_unique()).collect().item()
    stats_str = f"{prefix}Total: {total} spectra, {total_molecules} molecules\n"
    file_stats = (
        lf.group_by("source_file")
        .agg(pl.len().alias("spectra"), pl.col("base_inchikey").n_unique().alias("molecules"))
        .sort("source_file")
        .collect()
    )
    for row in file_stats.iter_rows(named=True):
        stats_str += f"  {row['source_file']}: {row['spectra']} spectra, {row['molecules']} molecules\n"
    return stats_str


def _formula_array_to_strings(formula_array: np.ndarray) -> list[str]:
    """Convert a 2D formula count array to molecular formula strings.

    Parameters
    ----------
    formula_array:
        Array of shape ``(n, 12)`` with counts in ``ELEMENT_SYMBOLS`` order.

    Returns
    -------
    List of formula strings following Hill notation: when carbon is present
    it is written first, followed by hydrogen, then the remaining elements in
    alphabetical order. Carbon-free formulas are written in alphabetical
    order. All-zero rows become ``""``.
    """
    strings = []
    for row in formula_array:
        counts = {sym: int(c) for sym, c in zip(ELEMENT_SYMBOLS, row) if int(c) > 0}
        if not counts:
            strings.append("")
            continue
        parts: list[str] = []
        if "C" in counts:
            c = counts.pop("C")
            parts.append("C" if c == 1 else f"C{c}")
            if "H" in counts:
                h = counts.pop("H")
                parts.append("H" if h == 1 else f"H{h}")
        for sym in sorted(counts):
            c = counts[sym]
            parts.append(sym if c == 1 else f"{sym}{c}")
        strings.append("".join(parts))
    return strings


def process_single_file(
    path: Path | str,
    raw_fragment_tolerance_ppm: float = 10.0,
    normalized_fragment_tolerance_ppm: float = 5.0,
    molecular_ion_tolerance_ppm: float = 5.0,
    includes_MSn: bool = False,
    lazy: bool = False,
    clean_identifiers: bool = True,
    annotate: bool = True,
    logger: logging.Logger | TextIO | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Unified entry point for processing a single spectral library file (MSP, MSPEC, MGF).

    When ``annotate`` is ``False`` the pipeline only normalizes metadata and
    returns a frame that is not yet annotated. ``process_spectral_library``
    uses this mode so it can apply MS-Ready standardization and
    precursor-type inference before running spectral annotation.
    """
    path = Path(path)
    lf = _parse_file(path, includes_MSn=includes_MSn)

    # Add source file metadata
    lf = lf.with_columns(pl.lit(path.name).alias("source_file"))

    # Run processing pipeline
    lf = _process_pipeline(
        lf,
        raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
        normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
        molecular_ion_tolerance_ppm=molecular_ion_tolerance_ppm,
        clean_identifiers=clean_identifiers,
        annotate=annotate,
        logger=logger,
    )

    if not lazy:
        return _collect_logged(
            lf, logger, f"collecting processed single file ({path.name})"
        )
    return lf


def process_spectral_library(
    files: list[Path | str],
    raw_fragment_tolerance_ppm: float = 10.0,
    normalized_fragment_tolerance_ppm: float = 5.0,
    molecular_ion_tolerance_ppm: float = 5.0,
    includes_MSn: bool = False,
    pubchem_path: Path | None = None,
    pubchem_fill_missing_only: bool = True,
    min_explained_intensity: float | None = None,
    dedup_threshold: float = 0.99,
    deduplicate: bool = True,
    clean_identifiers: bool = True,
    inchikey_changes_path: Path | None = None,
    output_path: Path | None = None,
    logger: logging.Logger | TextIO | None = None,
) -> pl.LazyFrame:
    """Unified entry point for processing multiple spectral library files with deduplication and enrichment.

    The annotated (and, if requested, deduplicated) frame is always streamed
    to a Parquet file on disk and a :class:`pl.LazyFrame` scanning the
    resulting Parquet is returned. When ``output_path`` is not provided it
    is deduced from the input file list: a single file yields
    ``<file>.parquet``; multiple files yield
    ``<parent_dir>/<parent_dir>.parquet``.

    The pipeline writes an intermediate ``.temp.parquet`` next to the final
    output, drops the in-memory frame, and rescans the intermediate file
    before deduplication/sinking to keep peak memory low.
    """

    t0 = time.perf_counter()
    if output_path is None:
        output_path = _deduce_output_path(files)
    lazyframes = []
    for f in files:
        ti = time.perf_counter()
        # Parse the file with metadata-only normalization; annotation is
        # performed later, after MS-Ready standardization and precursor-type
        # inference, so it can use the corrected structures.
        lf = process_single_file(
            f,
            raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
            normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
            molecular_ion_tolerance_ppm=molecular_ion_tolerance_ppm,
            includes_MSn=includes_MSn,
            lazy=True,
            clean_identifiers=clean_identifiers,
            annotate=False,
            logger=logger,
        )
        lazyframes.append(lf)
        _log(logger, f"[{time.perf_counter() - ti:.2f}s] Parsed {f}")

    combined_lf = pl.concat(lazyframes, how="diagonal_relaxed")
    # Workaround for polars schema lengths differ bug
    df = _collect_logged(combined_lf, logger, "concatenating parsed files")
    combined_lf = df.lazy()
    _log(logger, f"[{time.perf_counter() - t0:.2f}s] Parsed all files and collected")

    t1 = time.perf_counter()

    # Optional PubChem enrichment
    intermediate_path: Path | None = None
    if pubchem_path is not None:
        # Persist the current frame and re-scan to keep the rest of the pipeline
        # streaming-friendly and to give PubChem enrichment a stable schema.
        fd, intermediate_path_str = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)
        intermediate_path = Path(intermediate_path_str)
        try:
            _run_logged(
                logger,
                "writing intermediate Parquet for PubChem enrichment",
                combined_lf.sink_parquet,
                intermediate_path,
                engine="streaming",
            )
            combined_lf = pl.scan_parquet(intermediate_path)
            _log(logger, "Materialized intermediate Parquet for PubChem enrichment")

            # Report identifier status before enrichment
            stats = _collect_logged(
                combined_lf.select(
                    (pl.col("smiles").is_null() & pl.col("inchi").is_null())
                    .sum()
                    .alias("missing_both_count"),
                    (pl.col("smiles").is_not_null() & pl.col("inchi").is_not_null())
                    .sum()
                    .alias("have_both_count"),
                ),
                logger,
                "collecting identifier status before PubChem enrichment",
            )
            missing_both_count = stats["missing_both_count"].item()
            have_both_count = stats["have_both_count"].item()
            _log(logger, 
                f"Identifier status before PubChem: {missing_both_count} rows missing both SMILES and InChI, "
                f"{have_both_count} rows have both"
            )
            combined_lf = _enrich_with_pubchem(
                combined_lf, pubchem_path, fill_missing_only=pubchem_fill_missing_only
            )
            _log(logger, "Built PubChem enrichment")
        except Exception:
            if intermediate_path is not None:
                intermediate_path.unlink(missing_ok=True)
            raise

    # MS-ready standardization
    try:
        df = _collect_logged(
            combined_lf, logger, "collecting after optional PubChem enrichment"
        )
        _log(logger, f"[{time.perf_counter() - t1:.2f}s] collected after enriching with pubchem")
    finally:
        if intermediate_path is not None:
            intermediate_path.unlink(missing_ok=True)

    if clean_identifiers:
        t5 = time.perf_counter()
        # Preserve pre-standardization identifier columns for change tracking/CSV output
        df = df.with_columns(
            pl.col("base_inchikey").alias("original_base_inchikey"),
            pl.col("inchikey").alias("original_inchikey"),
            pl.col("smiles").alias("original_smiles"),
            pl.col("inchi").alias("original_inchi"),
        )
        df = _run_logged(logger, "standardizing structures", _standardize_structures, df)

        # Count rows whose base InChIKey changed during standardization
        n_changed = df.filter(
            pl.col("base_inchikey") != pl.col("original_base_inchikey")
        ).height
        _log(logger, f"{n_changed} rows had final base InChIKey differ from original")

        # Optionally write a CSV of changed rows next to the input
        if inchikey_changes_path is not None and n_changed > 0:
            changed_df = df.filter(
                pl.col("base_inchikey") != pl.col("original_base_inchikey")
            ).select(
                "name",
                pl.col("original_inchikey"),
                pl.col("original_base_inchikey"),
                pl.col("original_smiles"),
                pl.col("original_inchi"),
                pl.col("inchikey").alias("new_inchikey"),
                pl.col("base_inchikey").alias("new_base_inchikey"),
                pl.col("smiles").alias("new_smiles"),
                pl.col("inchi").alias("new_inchi"),
            )
            changed_df.write_csv(inchikey_changes_path)

        df = df.drop(
            [
                "original_base_inchikey",
                "original_inchikey",
                "original_smiles",
                "original_inchi",
            ]
        )
        _log(logger, f"[{time.perf_counter() - t5:.2f}s] Standardized structures")
    else:
        _log(logger, "Skipped identifier standardization (clean_identifiers=False)")

    # Drop rows whose standardization left them without any identifier.
    pre_filter = df.height
    df = _collect_logged(
        _filter_invalid_entries(df.lazy()), logger, "filtering invalid entries"
    )
    dropped = pre_filter - df.height
    if dropped > 0:
        _log(logger, 
            f"Dropped {dropped} rows with no molecular identifier after standardization"
        )

    # Fill missing molecular formulas and infer missing precursor types from
    # the MS-Ready SMILES produced by ``_standardize_structures``.
    t_infer = time.perf_counter()
    df = _run_logged(
        logger,
        "filling molecular formula and inferring precursor type",
        _fill_molecular_formula_and_infer_precursor_type,
        df,
        molecular_ion_tolerance_ppm,
    )
    _log(logger, 
        f"[{time.perf_counter() - t_infer:.2f}s] Filled molecular_formula and inferred precursor_type"
    )

    # Now run the annotation steps that were skipped in ``annotate=False``
    # mode. The pipeline still reuses the helper functions so behaviour is
    # consistent with ``process_single_file(annotate=True)``.
    df = _collect_logged(
        _compute_precursor_formula(df.lazy(), logger=logger),
        logger,
        "computing precursor formula for annotation",
    )
    df = _run_logged(
        logger, "adding precursor type indicators", _add_precursor_type_indicators, df
    )
    df = _run_logged(
        logger,
        "spectral formula annotation",
        _annotate_spectra,
        df,
        raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
        normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
    )
    df = _run_logged(
        logger,
        "adding molecular ion info",
        _add_molecular_ion_info,
        df,
        molecular_ion_tolerance_ppm,
    )
    df = _run_logged(
        logger,
        "adding spectral information score",
        _add_spectral_information_score,
        df,
    )

    # Apply the explained-intensity filter now that annotation is done.
    if min_explained_intensity is not None:
        pre_filter = df.height
        df = df.filter(pl.col("explained_intensity") >= min_explained_intensity)
        _log(logger, 
            f"Defined explained intensity filter (>= {min_explained_intensity}); "
            f"dropped {pre_filter - df.height} rows"
        )

    combined_lf = _select_output_columns(df.lazy())

    # Spill the annotated frame to a temp Parquet, drop the in-memory frame,
    # and rescan from disk. This keeps peak memory low before the (potentially
    # memory-hungry) deduplication step and the final sink.
    temp_path = output_path.with_suffix(".temp.parquet")
    try:
        _run_logged(
            logger,
            "writing intermediate Parquet",
            combined_lf.sink_parquet,
            temp_path,
            engine="streaming",
        )
        # Drop in-memory frames before rescanning from disk
        df = None  # type: ignore[assignment]
        del df
        combined_lf = pl.scan_parquet(temp_path)
        _log(logger, "Rescanned intermediate Parquet from disk")

        if deduplicate:
            t4 = time.perf_counter()

            stats_lf = combined_lf.select(["source_file", "base_inchikey", "clean_precursor"])
            _log(logger, "Library Statistics (Pre-deduplication):")
            _log(logger, _get_stats_str(stats_lf))
            _log(logger, "Library Statistics (Clean Precursors Only):")
            _log(logger, _get_stats_str(stats_lf.filter(pl.col("clean_precursor"))))

            combined_lf = _deduplicate_spectra(
                combined_lf,
                fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
                molecular_ion_tolerance_ppm=molecular_ion_tolerance_ppm,
                threshold=dedup_threshold,
                logger=logger,
            )
            _log(logger, f"[{time.perf_counter() - t4:.2f}s] Built deduplication plan")

            t6 = time.perf_counter()
            _run_logged(
                logger,
                "sinking deduplicated library to Parquet",
                combined_lf.sink_parquet,
                output_path,
                engine="streaming",
            )
            _log(logger, f"[{time.perf_counter() - t6:.2f}s] Sank deduplicated library to {output_path}")
        else:
            _log(logger, "Skipped deduplication (deduplicate=False)")
            t6 = time.perf_counter()
            _run_logged(
                logger,
                "sinking final library to Parquet",
                combined_lf.sink_parquet,
                output_path,
                engine="streaming",
            )
            _log(logger, f"[{time.perf_counter() - t6:.2f}s] Sank final library to {output_path}")

        result_lf = pl.scan_parquet(output_path)
        if deduplicate:
            _log(logger, "Library Statistics (Post-deduplication):")
            _log(logger, _get_stats_str(result_lf))
            _log(logger, "Library Statistics (Post-deduplication, Clean Precursors Only):")
            _log(logger, _get_stats_str(result_lf.filter(pl.col("clean_precursor"))))
        else:
            _log(logger, "Library Statistics (Final):")
            _log(logger, _get_stats_str(result_lf))

        _log(logger, f"[{time.perf_counter() - t0:.2f}s] Total execution time")
        return result_lf
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _parse_file(path: Path, includes_MSn: bool = False) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix in [".mspec", ".msp"]:
        return nist_mspec.parse_mspec(path)
    elif suffix == ".mgf":
        # Note: we use the new parse_mgf which returns a LazyFrame and has unified columns
        return mgf.parse_mgf(path, includes_MSn=includes_MSn)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _process_pipeline(
    lf: pl.LazyFrame,
    raw_fragment_tolerance_ppm: float,
    normalized_fragment_tolerance_ppm: float,
    molecular_ion_tolerance_ppm: float,
    clean_identifiers: bool = True,
    annotate: bool = True,
    logger: logging.Logger | TextIO | None = None,
) -> pl.LazyFrame:
    """
    Core processing pipeline.
    including organizing of metadata, spectrum annotation

    When ``annotate`` is ``False`` the pipeline only normalizes metadata
    (precursor-type strings, base InChIKey, instrument flags, collision
    energy extraction, optional InChIKey filling). Precursor-formula
    computation and spectral annotation are deferred to the caller so that
    :func:`process_spectral_library` can run them after MS-Ready
    standardization and precursor-type inference.
    """
    if clean_identifiers:
        lf = _fill_missing_inchikeys(lf, logger=logger)
    lf = _add_base_inchikey(lf)
    lf = _normalize_precursor_type_strings(lf)
    lf = _annotate_and_filter_metadata(lf)
    lf = _extract_collision_energy_values(lf)
    lf = _fill_missing_mslevel(lf)
    lf = _remove_electronic_noise(lf)
    if annotate:
        lf = _compute_precursor_formula(lf, logger=logger)
        lf = _run_logged(
            logger,
            "spectral formula annotation",
            _annotate_spectra,
            lf,
            raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
            normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
        )
        lf = _run_logged(
            logger, "adding precursor type indicators", _add_precursor_type_indicators, lf
        )
        lf = _run_logged(
            logger,
            "adding molecular ion info",
            _add_molecular_ion_info,
            lf,
            molecular_ion_tolerance_ppm,
        )
        lf = _run_logged(
            logger,
            "adding spectral information score",
            _add_spectral_information_score,
            lf,
        )

    return _select_output_columns(lf)


def _filter_invalid_entries(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Filters out entries lacking all molecular identifiers."""

    def _present(col: str) -> pl.Expr:
        return pl.col(col).is_not_null() & (pl.col(col) != "")

    return lf.filter(_present("inchikey") | _present("smiles") | _present("inchi"))


def _fill_missing_inchikeys(
    lf: pl.LazyFrame, logger: logging.Logger | TextIO | None = None
) -> pl.LazyFrame:
    """Fill missing inchikeys from inchi or smiles using parallel_rdkit."""
    df = _collect_logged(lf, logger, "filling missing InChIKeys")

    # Find rows where inchikey is missing but inchi or smiles exist
    missing_inchikey = df.filter(
        pl.col("inchikey").is_null()
        & (pl.col("inchi").is_not_null() | pl.col("smiles").is_not_null())
    )

    if missing_inchikey.is_empty():
        return lf

    result_df = df.clone()

    # First convert inchi to smiles if inchi exists but smiles doesn't
    inchi_only_mask = (
        pl.col("inchikey").is_null()
        & pl.col("inchi").is_not_null()
        & pl.col("smiles").is_null()
    )

    if result_df.filter(inchi_only_mask).height > 0:
        inchi_only_df = result_df.filter(inchi_only_mask)
        inchi_list = inchi_only_df["inchi"].unique().to_list()
        smiles_from_inchi = inchi_to_smiles_parallel(inchi_list)
        inchi_to_smiles_df = pl.DataFrame(
            {"inchi": inchi_list, "smiles_mapped": smiles_from_inchi}
        )

        result_df = (
            result_df.join(inchi_to_smiles_df, on="inchi", how="left")
            .with_columns(
                pl.when(inchi_only_mask)
                .then(pl.col("smiles_mapped"))
                .otherwise(pl.col("smiles"))
                .alias("smiles")
            )
            .drop("smiles_mapped")
        )

    # Then convert smiles to inchikey
    smiles_but_no_inchikey_mask = (
        pl.col("inchikey").is_null() & pl.col("smiles").is_not_null()
    )

    if result_df.filter(smiles_but_no_inchikey_mask).height > 0:
        smiles_df = result_df.filter(smiles_but_no_inchikey_mask)
        smiles_list = smiles_df["smiles"].unique().to_list()
        inchikeys = smiles_to_inchikey_parallel(smiles_list)
        smiles_to_inchikey_df = pl.DataFrame(
            {"smiles": smiles_list, "inchikey_mapped": inchikeys}
        )

        result_df = (
            result_df.join(smiles_to_inchikey_df, on="smiles", how="left")
            .with_columns(
                pl.when(smiles_but_no_inchikey_mask)
                .then(pl.col("inchikey_mapped"))
                .otherwise(pl.col("inchikey"))
                .alias("inchikey")
            )
            .drop("inchikey_mapped")
        )

    return result_df.lazy()


def _add_base_inchikey(lf: pl.LazyFrame) -> pl.LazyFrame:
    if "base_inchikey" in lf.collect_schema().names():
        return lf
    return lf.with_columns(
        pl.col("inchikey").str.split(by="-").list.get(0).alias("base_inchikey")
    )


def _compute_precursor_formula(
    lf: pl.LazyFrame, logger: logging.Logger | TextIO | None = None
) -> pl.LazyFrame:
    """Vectorized precursor formula computation."""
    _log(logger, "before computing precursor formula")
    # First ensure we have molecular_formula_array if missing (e.g. from some MGFs)
    if "molecular_formula_array" not in lf.collect_schema().names():
        lf = formula_to_array(lf, "molecular_formula", "molecular_formula_array")

    # Precursor formula logic
    lf = lf.with_columns(
        pl.when(pl.col("precursor_type").is_not_null())
        .then(pl.col("precursor_type").str.replace(r"\[(M.*)\][+\\-]?\\d*", r"$1"))
        .otherwise(
            pl.when(pl.col("ion_mode").str.to_uppercase().eq("P"))
            .then(pl.lit("M+H"))
            .otherwise(pl.lit("M-H"))
        )
        .str.replace("M", pl.col("molecular_formula"))
        .alias("precursor_formula")
    )

    # Handle cases where formula is missing but identifiers exist
    # Raise NotImplementedError if we have InChIKey but no formula - user will fix in parallel_rdkit
    check_df = _collect_logged(
        lf.filter(
            pl.col("molecular_formula").is_null() & pl.col("inchikey").is_not_null()
        ).limit(1),
        logger,
        "checking for missing molecular formulas",
    )
    if not check_df.is_empty():
        raise NotImplementedError(
            "Molecular formula missing for entry with InChIKey. This will be handled by parallel_rdkit."
        )

    result = lf.with_columns(
        pl.col("precursor_formula")
        .map_elements(format_formula_string_to_array, return_dtype=pl.List(pl.Int32))
        .list.to_array(width=num_elements)
        .alias("precursor_formula_array")
    )
    _log(logger, "after computing precursor formula")
    return result


def _fill_molecular_formula_and_infer_precursor_type(
    df: pl.DataFrame, tolerance_ppm: float
) -> pl.DataFrame:
    """Fill missing ``molecular_formula`` from MS-Ready SMILES and infer missing ``precursor_type``.

    Only rows with a non-empty ``smiles`` are considered. Invalid SMILES
    (all-zero formula arrays from ``smiles_to_formula``) are ignored.

    For positive ion mode, candidate adducts are tested in order:
    ``[M+H]+`` then ``[M+Na]+``. For negative ion mode, ``[M-H]-`` is tested.
    The first match within ``tolerance_ppm`` is assigned.
    """
    needs_processing = (
        pl.col("smiles").is_not_null()
        & (pl.col("smiles") != "")
        & (
            pl.col("molecular_formula").is_null()
            | (
                pl.col("precursor_type").is_null()
                & pl.col("precursor_mz").is_not_null()
                & (pl.col("precursor_mz") != 0.0)
            )
        )
    )
    processing_df = df.filter(needs_processing)
    if processing_df.is_empty():
        return df

    formula_arrays = smiles_to_formula(processing_df["smiles"])
    # ``smiles_to_formula`` may return a polars Series or a 2D numpy array
    # depending on the backend. Normalize to a 2D numpy array.
    if isinstance(formula_arrays, pl.Series):
        formula_arrays = np.stack(formula_arrays.to_list())
    valid_formula_mask = np.any(formula_arrays != 0, axis=1)
    formula_strings = _formula_array_to_strings(formula_arrays)

    # Compute exact mass from formula arrays using numpy (only for inferred rows)
    element_masses = np.array(ELEMENT_MASSES, dtype=np.float64)
    exact_masses = formula_arrays.astype(np.float64).dot(element_masses)

    processing_df = processing_df.with_columns(
        pl.Series("inferred_formula_array", formula_arrays),
        pl.Series("inferred_molecular_formula", formula_strings),
        pl.Series("inferred_exact_mass", exact_masses),
    )

    missing_formula_mask = pl.col("molecular_formula").is_null()
    needs_inference = (
        pl.col("precursor_type").is_null()
        & pl.col("precursor_mz").is_not_null()
        & (pl.col("precursor_mz") != 0.0)
    )

    # Compute neutral mass via Polars expression (dot product). The expression
    # is built once and reused; ``pl.sum_horizontal`` materializes a column
    # aligned to the processing frame.
    mass_expr = pl.sum_horizontal(
        [
            pl.col("inferred_formula_array").arr.get(i).cast(pl.Float64) * mass
            for i, mass in enumerate(ELEMENT_MASSES)
        ]
    )
    processing_df = processing_df.with_columns(mass_expr.alias("inferred_neutral_mass"))

    h_mass = ADDUCT_MASSES["[M+H]+"]
    na_mass = ADDUCT_MASSES["[M+Na]+"]
    mh_mass = ADDUCT_MASSES["[M-H]-"]

    processing_df = processing_df.with_columns(
        (pl.col("inferred_neutral_mass") + h_mass).alias("mass_m_h"),
        (pl.col("inferred_neutral_mass") + na_mass).alias("mass_m_na"),
        (pl.col("inferred_neutral_mass") + mh_mass).alias("mass_m_minus_h"),
    )

    def ppm_error(cand_col: str) -> pl.Expr:
        return (
            (pl.col(cand_col) - pl.col("precursor_mz")).abs()
            / pl.col("precursor_mz").abs()
            * 1e6
        )

    processing_df = processing_df.with_columns(
        ppm_error("mass_m_h").alias("err_m_h"),
        ppm_error("mass_m_na").alias("err_m_na"),
        ppm_error("mass_m_minus_h").alias("err_m_minus_h"),
    )

    pos_mask = pl.col("ion_mode").str.to_uppercase().is_in(["P", "POSITIVE"])
    neg_mask = pl.col("ion_mode").str.to_uppercase().is_in(["N", "NEGATIVE"])

    inferred_type = (
        pl.when(pos_mask & (pl.col("err_m_h") <= tolerance_ppm))
        .then(pl.lit("[M+H]+"))
        .when(pos_mask & (pl.col("err_m_na") <= tolerance_ppm))
        .then(pl.lit("[M+Na]+"))
        .when(neg_mask & (pl.col("err_m_minus_h") <= tolerance_ppm))
        .then(pl.lit("[M-H]-"))
        .otherwise(pl.lit(None))
    )
    processing_df = processing_df.with_columns(
        inferred_type.alias("inferred_precursor_type")
    )

    df = df.with_row_index("_infer_idx")
    processing_df = processing_df.with_row_index("_infer_idx")

    n_filled = int(
        processing_df.filter(missing_formula_mask & valid_formula_mask).height
    )
    n_inferred = int(
        processing_df.filter(
            needs_inference & pl.col("inferred_precursor_type").is_not_null()
        ).height
    )
    if n_filled > 0 or n_inferred > 0:
        print(
            f"Inferred molecular_formula for {n_filled} rows and precursor_type "
            f"for {n_inferred} rows from MS-Ready SMILES"
        )

    has_exact_mass = "exact_mass" in df.columns
    join_cols = [
        "_infer_idx",
        "inferred_formula_array",
        "inferred_molecular_formula",
        "inferred_exact_mass",
        "inferred_precursor_type",
    ]

    joined = df.join(
        processing_df.select(join_cols),
        on="_infer_idx",
        how="left",
    )

    update_exprs = [
        pl.when(
            missing_formula_mask
            & pl.col("inferred_molecular_formula").is_not_null()
            & (pl.col("inferred_molecular_formula") != "")
        )
        .then(pl.col("inferred_molecular_formula"))
        .otherwise(pl.col("molecular_formula"))
        .alias("molecular_formula"),
        pl.when(needs_inference & pl.col("inferred_precursor_type").is_not_null())
        .then(pl.col("inferred_precursor_type"))
        .otherwise(pl.col("precursor_type"))
        .alias("precursor_type"),
    ]
    if has_exact_mass:
        update_exprs.append(
            pl.when(missing_formula_mask & pl.col("inferred_exact_mass").is_not_null())
            .then(pl.col("inferred_exact_mass"))
            .otherwise(pl.col("exact_mass"))
            .alias("exact_mass")
        )

    drop_cols = [
        "_infer_idx",
        "inferred_formula_array",
        "inferred_molecular_formula",
        "inferred_exact_mass",
        "inferred_precursor_type",
    ]

    return joined.with_columns(update_exprs).drop(drop_cols)


def _normalize_precursor_type_strings(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.when(pl.col("precursor_type").is_not_null())
        .then(
            pl.col("precursor_type")
            .str.strip_chars()
            .str.replace_all(r"\s+", "")
            .str.replace_all(r"[−–—]", "-")
        )
        .otherwise(None)
        .alias("precursor_type")
    )


def _annotate_and_filter_metadata(data: polarsFrame) -> polarsFrame:
    # Different formats expose instrument metadata under different columns,
    # and some formats omit fields like sample_inlet entirely. Only use the
    # columns that are actually present so the lazy plan stays valid across
    # MGF, MSP, and mspec inputs.
    possible_instrument_cols = [
        "instrument",
        "instrument_type",
        "ionization",
        "sample_inlet",
    ]
    existing_instrument_cols = [
        c for c in possible_instrument_cols if c in data.collect_schema().names()
    ]

    if existing_instrument_cols:
        instrument_data_columns = pl.selectors.by_name(existing_instrument_cols)
        qq_mask = pl.any_horizontal(
            instrument_data_columns.str.contains(r"(?i)QQ").fill_null(False)
        )
        is_lc = pl.any_horizontal(
            instrument_data_columns.str.contains(r"(?i)LC|HPLC").fill_null(False)
        )
        is_orbitrap = pl.any_horizontal(
            instrument_data_columns.str.contains(r"(?i)orbi(?:trap)?|HCD").fill_null(
                False
            )
            | instrument_data_columns.str.contains(r"(?i)thermo").fill_null(False)
            | (
                instrument_data_columns.str.contains(r"(?i)FT").fill_null(False)
                & instrument_data_columns.str.contains(r"(?i)ICR")
                .not_()
                .fill_null(True)
                & instrument_data_columns.str.contains(r"(?i)TOF")
                .not_()
                .fill_null(True)
            )
        )
        is_tof = pl.any_horizontal(
            instrument_data_columns.str.contains(r"(?i)TOF").fill_null(False)
        )
        is_esi = pl.any_horizontal(
            instrument_data_columns.str.contains(r"(?i)ESI|LC").fill_null(False)
        )
    else:
        qq_mask = pl.lit(False)
        is_lc = pl.lit(False)
        is_orbitrap = pl.lit(False)
        is_tof = pl.lit(False)
        is_esi = pl.lit(False)

    return cast(
        polarsFrame,
        data.filter(qq_mask.not_()).with_columns(
            is_lc.alias("is_LC"),
            is_orbitrap.alias("is_orbitrap"),
            is_tof.alias("is_TOF"),
            is_esi.alias("is_ESI"),
        ),
    )


def _extract_collision_energy_values(data: polarsFrame) -> polarsFrame:
    # Different formats represent collision energy differently. Detect what is
    # available and branch accordingly so formats that omit these fields do not
    # break the pipeline.
    cols = data.collect_schema().names()
    has_collision_energy_list = "collision_energy_list" in cols
    has_collision_energy_raw = "collision_energy_raw" in cols
    has_ev_raw = "_collision_energies_ev_raw" in cols

    if has_collision_energy_list:
        mean_expr = pl.col("collision_energy_list").list.mean()
        is_orbi = pl.col("is_orbitrap").fill_null(False)
        return cast(
            polarsFrame,
            data.with_columns(
                pl.col("collision_energy_list")
                .list.len()
                .ge(2)
                .fill_null(False)
                .alias("multiple_collision_energies"),
                mean_expr.alias("collision_energy_mean"),
                pl.when(is_orbi)
                .then(mean_expr)
                .otherwise(None)
                .alias("collision_energy_NCE"),
                pl.when(is_orbi.not_())
                .then(mean_expr)
                .otherwise(None)
                .alias("collision_energy_ev"),
            )
            .with_columns(
                pl.when(
                    pl.col("collision_energy_NCE").is_null()
                    & pl.col("collision_energy_ev").is_not_null()
                    & pl.col("precursor_mz").is_not_null()
                    & (pl.col("precursor_mz") != 0.0)
                )
                .then(
                    pl.col("collision_energy_ev")
                    .mul(500.0)
                    .truediv(pl.col("precursor_mz"))
                    .abs()
                )
                .otherwise(pl.col("collision_energy_NCE"))
                .alias("collision_energy_NCE"),
                pl.when(
                    pl.col("collision_energy_ev").is_null()
                    & pl.col("collision_energy_NCE").is_not_null()
                    & pl.col("precursor_mz").is_not_null()
                    & (pl.col("precursor_mz") != 0.0)
                )
                .then(
                    pl.col("collision_energy_NCE")
                    .mul(pl.col("precursor_mz"))
                    .truediv(500.0)
                    .abs()
                )
                .otherwise(pl.col("collision_energy_ev"))
                .alias("collision_energy_ev"),
            ),
        )

    if not has_collision_energy_raw:
        # No usable collision energy metadata at all; create the expected columns
        # as nulls so downstream consumers can still select them.
        return cast(
            polarsFrame,
            data.with_columns(
                pl.lit(None).cast(pl.Float64).alias("collision_energy_NCE"),
                pl.lit(None).cast(pl.Float64).alias("collision_energy_ev"),
                pl.lit(None).cast(pl.List(pl.Float64)).alias("collision_energy_list"),
                pl.lit(False).alias("multiple_collision_energies"),
                pl.lit(None).cast(pl.Float64).alias("collision_energy_mean"),
            ),
        )

    pat_nce = r"(?i)(?:NCE\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)|([0-9]+(?:\.[0-9]+)?)\s*(?:%|(?:\(?NCE\)?)))"
    pat_ev = r"(?i)([0-9]+(?:\.[0-9]+)?)\s*e?V"
    pat_num = r"([0-9]+(?:\.[0-9]+)?)"
    pat_list_content = r"\[(.*?)\]"

    result = cast(
        polarsFrame,
        data.with_columns(
            pl.col("collision_energy_raw")
            .str.extract(pat_nce, group_index=1)
            .fill_null(
                pl.col("collision_energy_raw").str.extract(pat_nce, group_index=2)
            )
            .cast(pl.Float64, strict=False)
            .abs()
            .alias("collision_energy_NCE"),
            pl.col("collision_energy_raw")
            .str.extract(pat_ev, group_index=1)
            .cast(pl.Float64, strict=False)
            .abs()
            .alias("collision_energy_ev"),
            pl.col("collision_energy_raw")
            .str.extract(pat_list_content, group_index=1)
            .str.extract_all(r"\d+(?:\.\d+)?")
            .list.eval(pl.element().cast(pl.Float64, strict=False).abs())
            .alias("collision_energy_list"),
        )
        .with_columns(
            pl.when(
                pl.col("collision_energy_NCE").is_null()
                & pl.col("collision_energy_ev").is_null()
                & pl.col("collision_energy_list").is_null()
            )
            .then(
                pl.col("collision_energy_raw")
                .str.extract(pat_num, group_index=1)
                .cast(pl.Float64, strict=False)
                .abs()
            )
            .otherwise(None)
            .alias("_bare_energy"),
            pl.col("collision_energy_list").list.mean().alias("_list_mean"),
        )
        .with_columns(
            pl.when(pl.col("collision_energy_NCE").is_null())
            .then(
                pl.when(pl.col("is_orbitrap"))
                .then(pl.coalesce([pl.col("_list_mean"), pl.col("_bare_energy")]))
                .otherwise(None)
            )
            .otherwise(pl.col("collision_energy_NCE"))
            .alias("collision_energy_NCE"),
            pl.when(pl.col("collision_energy_ev").is_null())
            .then(
                pl.when(pl.col("is_orbitrap").not_())
                .then(pl.coalesce([pl.col("_list_mean"), pl.col("_bare_energy")]))
                .otherwise(None)
            )
            .otherwise(pl.col("collision_energy_ev"))
            .alias("collision_energy_ev"),
        ),
    )

    if has_ev_raw:
        result = result.with_columns(
            pl.when(pl.col("_collision_energies_ev_raw").is_not_null())
            .then(
                pl.col("_collision_energies_ev_raw")
                .str.extract(pat_num, group_index=1)
                .cast(pl.Float64, strict=False)
                .abs()
            )
            .otherwise(pl.col("collision_energy_ev"))
            .alias("collision_energy_ev")
        )

    # Derive the missing unit from the known one using a 500 Da normalization.
    # Values are always kept positive.
    result = cast(
        polarsFrame,
        result.with_columns(
            pl.when(
                pl.col("collision_energy_NCE").is_null()
                & pl.col("collision_energy_ev").is_not_null()
                & pl.col("precursor_mz").is_not_null()
                & (pl.col("precursor_mz") != 0.0)
            )
            .then(
                pl.col("collision_energy_ev")
                .mul(500.0)
                .truediv(pl.col("precursor_mz"))
                .abs()
            )
            .otherwise(pl.col("collision_energy_NCE"))
            .alias("collision_energy_NCE"),
            pl.when(
                pl.col("collision_energy_ev").is_null()
                & pl.col("collision_energy_NCE").is_not_null()
                & pl.col("precursor_mz").is_not_null()
                & (pl.col("precursor_mz") != 0.0)
            )
            .then(
                pl.col("collision_energy_NCE")
                .mul(pl.col("precursor_mz"))
                .truediv(500.0)
                .abs()
            )
            .otherwise(pl.col("collision_energy_ev"))
            .alias("collision_energy_ev"),
        ).with_columns(
            pl.col("collision_energy_list")
            .list.len()
            .ge(2)
            .fill_null(False)
            .alias("multiple_collision_energies"),
            pl.coalesce(
                [
                    pl.col("_list_mean"),
                    pl.col("collision_energy_NCE"),
                    pl.col("collision_energy_ev"),
                ]
            ).alias("collision_energy_mean"),
        ),
    )

    drop_cols = ["_bare_energy", "_list_mean"]
    if has_ev_raw:
        drop_cols.append("_collision_energies_ev_raw")
    return cast(polarsFrame, result.drop(drop_cols))


def _fill_missing_mslevel(data: polarsFrame) -> polarsFrame:
    """Fill absent or null ``mslevel`` values with 2 (MS2)."""
    cols = data.collect_schema().names()
    if "mslevel" not in cols:
        return cast(
            polarsFrame,
            data.with_columns(pl.lit(2).cast(pl.Int64).alias("mslevel")),
        )
    return cast(
        polarsFrame,
        data.with_columns(
            pl.when(pl.col("mslevel").is_null())
            .then(pl.lit(2).cast(pl.Int64))
            .otherwise(pl.col("mslevel"))
            .alias("mslevel")
        ),
    )


def _remove_electronic_noise(data: polarsFrame) -> polarsFrame:
    """Drop electronic-noise floor peaks from raw spectra.

    For each spectrum, if the lowest intensity fragment is below 1% of the
    most intense fragment, every fragment whose intensity equals that minimum
    (within a 1e-2 relative tolerance) is removed from both the mass and
    intensity lists.
    """
    cols = data.collect_schema().names()
    if "raw_spectrum_mz" not in cols or "raw_spectrum_intensity" not in cols:
        return cast(polarsFrame, data)

    idx = "_noise_row_index"
    indexed = data.with_row_index(idx)
    cleaned = (
        indexed.explode(["raw_spectrum_mz", "raw_spectrum_intensity"])
        .with_columns(
            pl.col("raw_spectrum_intensity").min().over(idx).alias("_min_intensity"),
            pl.col("raw_spectrum_intensity").max().over(idx).alias("_max_intensity"),
        )
        .filter(
            (pl.col("_min_intensity") >= 0.01 * pl.col("_max_intensity"))
            | (
                (pl.col("raw_spectrum_intensity") - pl.col("_min_intensity")).abs()
                > 1e-2 * pl.col("_min_intensity")
            )
        )
        .group_by(idx)
        .agg(
            pl.col("raw_spectrum_mz"),
            pl.col("raw_spectrum_intensity"),
        )
    )

    return cast(
        polarsFrame,
        indexed.drop(["raw_spectrum_mz", "raw_spectrum_intensity"])
        .join(cleaned, on=idx, how="left")
        .drop(idx),
    )


def _annotate_spectra(
    data: polarsFrame,
    raw_fragment_tolerance_ppm: float,
    normalized_fragment_tolerance_ppm: float,
) -> polarsFrame:
    return cast(
        polarsFrame,
        data.with_columns(
            pl.col("raw_spectrum_intensity")
            .truediv(pl.col("raw_spectrum_intensity").list.sum())
            .alias("raw_spectrum_intensity")
        )
        .with_columns(
            pl.struct(
                [
                    pl.col("precursor_formula_array").alias("precursor_formula"),
                    pl.col("raw_spectrum_mz").alias("mz"),
                    pl.col("raw_spectrum_intensity").alias("intensities"),
                ]
            )
            .mass_decomposition.clean_and_normalize_spectrum(
                raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
                normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
                min_dbe=-0.5,
                max_dbe=40,
                dbe_mode="half_integer",
                water_absorption=True,
            )
            .alias("cleaned_normalized_spectra")
        )
        .with_columns(
            pl.col("cleaned_normalized_spectra")
            .struct.field("normalized_masses")
            .alias("cleaned_normalized_mz"),
            pl.col("cleaned_normalized_spectra")
            .struct.field("intensities")
            .alias("cleaned_normalized_intensity"),
            pl.col("cleaned_normalized_spectra")
            .struct.field("formulas")
            .alias("cleaned_fragment_formulas"),
            pl.col("cleaned_normalized_spectra")
            .struct.field("formulas_str")
            .alias("cleaned_fragment_formulas_str"),
            pl.col("cleaned_normalized_spectra")
            .struct.field("errors_ppm")
            .alias("cleaned_fragment_errors_ppm"),
        )
        .drop("cleaned_normalized_spectra")
        .with_columns(
            pl.col("cleaned_normalized_intensity")
            .list.sum()
            .truediv(pl.col("raw_spectrum_intensity").list.sum())
            .alias("explained_intensity")
        ),
    )


def _add_precursor_type_indicators(data: polarsFrame) -> polarsFrame:
    fragment_pattern = (
        r"-\d*"
        + r"((H(\d+|[A-Z]|[a-z]))|([A-G]|[I-Z])[a-z]?\d*)"
        + r"(([A-Z][a-z]?\d*))*"
    )
    return cast(
        polarsFrame,
        data.with_columns(
            pl.col("precursor_type").str.contains("i").alias("Isotope"),
            pl.col("precursor_type").str.contains("Cat").alias("Cation"),
            pl.col("precursor_type").str.contains("[0-9]M").alias("Multimer"),
            pl.col("precursor_type").str.contains("][0-9]").alias("MultiCharge"),
            pl.col("precursor_type").str.contains(fragment_pattern).alias("Fragment"),
        ).with_columns(
            (
                pl.col("Isotope")
                | pl.col("Cation")
                | pl.col("Multimer")
                | pl.col("MultiCharge")
                | pl.col("Fragment")
                | pl.col("precursor_type").str.contains("M").not_()
            )
            .not_()
            .alias("clean_precursor")
        ),
    )


def _add_molecular_ion_info(
    NIST: polarsFrame, tolerance_ppm: float = 10.0
) -> polarsFrame:
    lf = NIST.lazy().with_columns(
        molecular_ion_intensity=pl.when(
            pl.col("cleaned_normalized_mz")
            .list.last()
            .is_close(
                pl.col("precursor_mz"),
                rel_tol=tolerance_ppm * 1e-6,
                abs_tol=200.0 * tolerance_ppm * 1e-6,
            )
        )
        .then(pl.col("cleaned_normalized_intensity").list.last())
        .otherwise(None)
    )
    return cast(
        polarsFrame,
        lf if isinstance(NIST, pl.LazyFrame) else lf.collect(),
    )


def _add_spectral_information_score(data: polarsFrame) -> polarsFrame:
    return cast(
        polarsFrame,
        data.with_columns(
            pl.struct(
                [
                    pl.col("precursor_formula_array").alias("precursor_formula"),
                    pl.col("cleaned_fragment_formulas").alias("fragment_formulas"),
                ]
            ).alias("spectra_for_spectral_info")
        ).with_columns(
            pl.col("spectra_for_spectral_info")
            .spectral_info.spectral_info_score(
                distance_metric="l2", ignore_hydrogens=True
            )
            .alias("spectral_information_score"),
            pl.col("spectra_for_spectral_info")
            .spectral_info.spectral_info_score(
                distance_metric="l2", ignore_hydrogens=False
            )
            .alias("spectral_information_score_with_hydrogens"),
        ),
    )


def _deduplicate_spectra(
    lf: pl.LazyFrame,
    fragment_tolerance_ppm: float,
    molecular_ion_tolerance_ppm: float,
    threshold: float,
    logger: logging.Logger | TextIO | None = None,
) -> pl.LazyFrame:
    """Pairwise deduplication based on explained intensity, restricted to same base_inchikey and precursor_mz."""
    lf = lf.with_row_index("_dedup_idx")

    valid_bases = lf.filter(
        pl.col("base_inchikey").is_not_null() & (pl.col("base_inchikey") != "")
    )

    # We only want to deduplicate among valid base_inchikeys
    base_lf = valid_bases.select(
        [
            "_dedup_idx",
            "base_inchikey",
            "cleaned_normalized_mz",
            "cleaned_normalized_intensity",
            "precursor_mz",
            "source_file",
        ]
    )

    # Bin precursor_mz to 0.1 Da bins to avoid Cartesian explosion during join
    base_lf = base_lf.with_columns(
        pl.col("precursor_mz").round(decimals=0).cast(pl.Int64).alias("_mz_bin")
    )
    # Join on base_inchikey AND the expanded mz bin
    pairs = base_lf.join(
        base_lf,
        on=["base_inchikey", "_mz_bin"],
        suffix="_right",
    ).filter(pl.col("_dedup_idx") < pl.col("_dedup_idx_right"))

    # Now exact precursor_mz tolerance filter
    # ppm = abs(mz1 - mz2) / mz2 * 1e6
    pairs = pairs.with_columns(
        (
            (pl.col("precursor_mz") - pl.col("precursor_mz_right")).abs()
            / pl.col("precursor_mz_right")
            * 1e6
        ).alias("_ppm_diff")
    ).filter(pl.col("_ppm_diff") <= molecular_ion_tolerance_ppm)

    # Forward similarity
    pairs = pairs.with_columns(
        pl.struct(
            [
                pl.col("cleaned_normalized_mz").alias("mz1"),
                pl.col("cleaned_normalized_intensity").alias("intensities1"),
                pl.col("cleaned_normalized_mz_right").alias("mz2"),
                pl.col("cleaned_normalized_intensity_right").alias("intensities2"),
                pl.col("precursor_mz").alias("precursor_mz1"),
                pl.col("precursor_mz_right").alias("precursor_mz2"),
            ]
        )
        .spectral_similarity.explained_intensity(
            ms2_tolerance_in_ppm=fragment_tolerance_ppm, permissive=True
        )
        .alias("sim_forward"),
        # Reverse similarity
        pl.struct(
            [
                pl.col("cleaned_normalized_mz_right").alias("mz1"),
                pl.col("cleaned_normalized_intensity_right").alias("intensities1"),
                pl.col("cleaned_normalized_mz").alias("mz2"),
                pl.col("cleaned_normalized_intensity").alias("intensities2"),
                pl.col("precursor_mz_right").alias("precursor_mz1"),
                pl.col("precursor_mz").alias("precursor_mz2"),
            ]
        )
        .spectral_similarity.explained_intensity(
            ms2_tolerance_in_ppm=fragment_tolerance_ppm, permissive=True
        )
        .alias("sim_reverse"),
    )

    duplicate_pairs = _collect_logged(
        pairs.filter(
            (pl.col("sim_forward") >= threshold) & (pl.col("sim_reverse") >= threshold)
        ),
        logger,
        "collecting duplicate candidate pairs",
    )

    if not duplicate_pairs.is_empty():
        # Log total duplicates
        total_dups = duplicate_pairs.select("_dedup_idx_right").n_unique()
        _log(logger, f"Deduplication: found {total_dups} duplicate spectra")

        # Log duplicates per file
        dups_per_file = duplicate_pairs.group_by("source_file_right").agg(
            pl.col("_dedup_idx_right").n_unique().alias("num_duplicates")
        )
        for row in dups_per_file.iter_rows(named=True):
            _log(
                logger,
                f"  {row['source_file_right']}: {row['num_duplicates']} duplicates",
            )

        # Overlap matrix
        overlap_matrix = (
            duplicate_pairs.group_by(["source_file", "source_file_right"])
            .agg(pl.len().alias("count"))
            .pivot(on="source_file_right", index="source_file", values="count")
            .fill_null(0)
        )

        with open("overlap_matrix.txt", "w") as f:
            f.write(str(overlap_matrix))

    to_remove = duplicate_pairs.select("_dedup_idx_right").unique()

    # Join back with original lf and filter out the duplicates
    return (
        lf.join(
            to_remove.lazy().with_columns(pl.lit(True).alias("_is_dup")),
            left_on="_dedup_idx",
            right_on="_dedup_idx_right",
            how="left",
        )
        .filter(pl.col("_is_dup").is_null())
        .drop(["_dedup_idx", "_is_dup"])
    )


def _enrich_with_pubchem(
    lf: pl.LazyFrame, pubchem_path: Path, fill_missing_only: bool = False
) -> pl.LazyFrame:
    """get smiles from pubchem based on matching base_inchikey

    If fill_missing_only is True, PubChem values are only used to fill rows where
    BOTH 'smiles' and 'inchi' are null. Otherwise PubChem identifiers are
    coalesced over the existing ones for every row.

    To keep memory low, only a narrow slice of columns (the join key and the
    identifier columns) is passed through the PubChem join. A row index is used
    to merge the enrichment result back into the full frame so no wide columns
    (spectra, arrays, etc.) are expanded during the join.
    """
    pubchem_lf = (
        pl.scan_parquet(str(pubchem_path), low_memory=True)
        .select(
            [
                pl.col("CID").alias("cid"),
                pl.col("InChIKey").alias("inchikey_pubchem"),
                pl.col("SMILES").alias("smiles_pubchem"),
                pl.col("InChI").alias("inchi_pubchem"),
            ]
        )
        .with_columns(
            pl.col("inchikey_pubchem").str.split("-").list.get(0).alias("base_inchikey")
        )
        # Deduplicate pubchem by base_inchikey
        .unique(subset="base_inchikey")
    )

    # Use a row index to merge enrichment back without carrying wide columns.
    lf = lf.with_row_index("_enrich_idx")

    missing_mask = pl.col("smiles").is_null() & pl.col("inchi").is_null()
    if fill_missing_only:
        enrich_lf = lf.filter(missing_mask)
    else:
        enrich_lf = lf

    # Only the columns needed for matching and coalescing enter the join.
    enrich_lf = (
        enrich_lf.select(
            [
                "_enrich_idx",
                "base_inchikey",
                "smiles",
                "inchi",
                "inchikey",
            ]
        )
        .filter(
            pl.col("base_inchikey").is_not_null() & (pl.col("base_inchikey") != "")
        )
        .join(pubchem_lf, on="base_inchikey", how="left")
    )

    enriched = enrich_lf.with_columns(
        pl.coalesce([pl.col("smiles_pubchem"), pl.col("smiles")]).alias(
            "smiles_enriched"
        ),
        pl.coalesce([pl.col("inchi_pubchem"), pl.col("inchi")]).alias(
            "inchi_enriched"
        ),
        pl.coalesce([pl.col("inchikey_pubchem"), pl.col("inchikey")]).alias(
            "inchikey_enriched"
        ),
    ).select(
        [
            "_enrich_idx",
            "smiles_enriched",
            "inchi_enriched",
            "inchikey_enriched",
        ]
    )

    # Merge the narrow enrichment result back into the full frame.
    return (
        lf.join(enriched, on="_enrich_idx", how="left")
        .with_columns(
            pl.coalesce([pl.col("smiles_enriched"), pl.col("smiles")]).alias("smiles"),
            pl.coalesce([pl.col("inchi_enriched"), pl.col("inchi")]).alias("inchi"),
            pl.coalesce([pl.col("inchikey_enriched"), pl.col("inchikey")]).alias(
                "inchikey"
            ),
        )
        .drop(
            [
                "_enrich_idx",
                "smiles_enriched",
                "inchi_enriched",
                "inchikey_enriched",
            ]
        )
    )


def _standardize_structures(df: pl.DataFrame) -> pl.DataFrame:

    # Get unique SMILES per base_inchikey
    mapping_df = df.group_by("base_inchikey").agg(pl.col("smiles").drop_nulls().first())

    smiles_list = [s if s is not None else "" for s in mapping_df["smiles"].to_list()]
    msready_smiles, msready_inchi, msready_inchikey = msready_inchi_inchikey_parallel(
        smiles_list
    )

    results_df = pl.DataFrame(
        {
            "base_inchikey": mapping_df["base_inchikey"],
            "msready_smiles": msready_smiles,
            "msready_inchi": msready_inchi,
            "msready_inchikey": msready_inchikey,
        }
    )

    # Join back and replace
    df = df.join(results_df, on="base_inchikey", how="left")

    # Identify failures: No msready_inchikey (None or empty string)
    failed_mask = pl.col("msready_inchikey").is_null() | (
        pl.col("msready_inchikey") == ""
    )
    failed_df = df.filter(failed_mask)

    if not failed_df.is_empty():
        failed_df.write_parquet("failed_structures.parquet")
        num_failed_keys = failed_df["base_inchikey"].n_unique()
        num_failed_rows = failed_df.height
        print(
            f"Standardization failed for {num_failed_keys} unique InChIKeys ({num_failed_rows} total rows). Saved to 'failed_structures.parquet'."
        )

    return df.with_columns(
        pl.col("msready_smiles").alias("smiles"),
        pl.col("msready_inchi").alias("inchi"),
        pl.col("msready_inchikey").alias("inchikey"),
        pl.col("msready_inchikey").str.extract(r"(.+?)-").alias("base_inchikey"),
    ).drop(["msready_smiles", "msready_inchi", "msready_inchikey"])
