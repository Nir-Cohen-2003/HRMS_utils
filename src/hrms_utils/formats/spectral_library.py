import re
import time
from pathlib import Path
from typing import List, Optional, TypeVar, Union, cast

import polars as pl
import polars.selectors as cs
from parallel_rdkit import (
    inchi_to_smiles_parallel,
    msready_inchi_inchikey_parallel,
    smiles_to_inchikey_parallel,
)

from ..formula_annotation.utils import (
    format_formula_string_to_array,
    formula_to_array,
    num_elements,
)
from ..hrms_core import *
from . import mgf, nist_mspec

polarsFrame = TypeVar("polarsFrame", pl.DataFrame, pl.LazyFrame)


def process_single_file(
    path: Path | str,
    raw_fragment_tolerance_ppm: float = 10.0,
    normalized_fragment_tolerance_ppm: float = 5.0,
    molecular_ion_tolerance_ppm: float = 5.0,
    includes_MSn: bool = False,
    lazy: bool = False,
    clean_identifiers: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    """Unified entry point for processing a single spectral library file (MSP, MSPEC, MGF)."""
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
    )

    if not lazy:
        return lf.collect(engine="streaming")
    return lf


def process_spectral_library(
    files: list[Path | str],
    raw_fragment_tolerance_ppm: float = 10.0,
    normalized_fragment_tolerance_ppm: float = 5.0,
    molecular_ion_tolerance_ppm: float = 5.0,
    includes_MSn: bool = False,
    pubchem_path: Path | None = None,
    min_explained_intensity: float | None = None,
    dedup_threshold: float = 0.99,
    deduplicate: bool = True,
    clean_identifiers: bool = True,
    logger=None,
) -> pl.DataFrame:
    """Unified entry point for processing multiple spectral library files with deduplication and enrichment."""

    def log(msg):
        if logger is not None:
            print(msg, file=logger)

    t0 = time.perf_counter()
    lazyframes = []
    for f in files:
        ti = time.perf_counter()
        lf = process_single_file(
            f,
            raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
            normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
            molecular_ion_tolerance_ppm=molecular_ion_tolerance_ppm,
            includes_MSn=includes_MSn,
            lazy=True,
            clean_identifiers=clean_identifiers,
        )
        lazyframes.append(lf)
        log(f"[{time.perf_counter() - ti:.2f}s] Parsed {f}")

    combined_lf = pl.concat(lazyframes, how="diagonal_relaxed")
    # Filter by explained intensity
    if min_explained_intensity is not None:
        combined_lf = _filter_invalid_entries(combined_lf)
        # Filter rows with no molecular info early
        combined_lf = combined_lf.filter(
            pl.col("explained_intensity") >= min_explained_intensity
        )
        log(
            f"Defined explained intensity filter (>= {min_explained_intensity}) and any molecular info presence filter "
        )
    # Workaround for polars schema lengths differ bug
    df = combined_lf.collect(engine="streaming")
    combined_lf = df.lazy()
    log(f"[{time.perf_counter() - t0:.2f}s] Parsed all files and collected")

    t1 = time.perf_counter()

    # Optional PubChem enrichment
    if pubchem_path is not None:
        combined_lf = _enrich_with_pubchem(combined_lf, pubchem_path)
        log("Built PubChem enrichment")

    # MS-ready standardization
    df = combined_lf.collect(engine="streaming")
    log(f"[{time.perf_counter() - t1:.2f}s] collected after enriching with pubchem")

    if clean_identifiers:
        t5 = time.perf_counter()
        df = _standardize_structures(df)
        combined_lf = df.lazy()
        log(f"[{time.perf_counter() - t5:.2f}s] Standardized structures")
    else:
        combined_lf = df.lazy()
        log("Skipped identifier standardization (clean_identifiers=False)")

    # Pairwise deduplication
    if deduplicate:
        t4 = time.perf_counter()

        # Pre-deduplication stats
        stats_df = combined_lf.select(
            ["source_file", "base_inchikey", "clean_precursor"]
        ).collect()

        def get_stats(df, prefix=""):
            total_spectra = df.height
            total_molecules = df["base_inchikey"].n_unique()
            stats_str = (
                f"{prefix}Total: {total_spectra} spectra, {total_molecules} molecules\n"
            )

            file_stats = (
                df.group_by("source_file")
                .agg(
                    pl.len().alias("spectra"),
                    pl.col("base_inchikey").n_unique().alias("molecules"),
                )
                .sort("source_file")
            )
            for row in file_stats.iter_rows(named=True):
                stats_str += f"  {row['source_file']}: {row['spectra']} spectra, {row['molecules']} molecules\n"
            return stats_str

        log("Library Statistics (Pre-deduplication):")
        log(get_stats(stats_df))

        clean_df = stats_df.filter(pl.col("clean_precursor"))
        log("Library Statistics (Clean Precursors Only):")
        log(get_stats(clean_df))

        combined_lf = _deduplicate_spectra(
            combined_lf,
            fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
            molecular_ion_tolerance_ppm=molecular_ion_tolerance_ppm,
            threshold=dedup_threshold,
        )
        log(f"[{time.perf_counter() - t4:.2f}s] Built deduplication plan")

        t6 = time.perf_counter()
        df = combined_lf.collect(engine="streaming")
        log(f"[{time.perf_counter() - t6:.2f}s] Collected deduplicated library")

        log("Library Statistics (Post-deduplication):")
        log(get_stats(df))
        log("Library Statistics (Post-deduplication, Clean Precursors Only):")
        log(get_stats(df.filter(pl.col("clean_precursor"))))
    else:
        df = combined_lf.collect(engine="streaming")
        log("Skipped deduplication (deduplicate=False)")

    log(f"[{time.perf_counter() - t0:.2f}s] Total execution time")

    return df


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
) -> pl.LazyFrame:
    """
    Core processing pipeline.
    including organizing of metadata, spectrum annotation
    """
    if clean_identifiers:
        lf = _fill_missing_inchikeys(lf)
    lf = _add_base_inchikey(lf)
    lf = _normalize_precursor_type_strings(lf)
    lf = _compute_precursor_formula(lf)
    lf = _annotate_and_filter_metadata(lf)
    lf = _extract_collision_energy_values(lf)
    lf = _annotate_spectra(
        lf,
        raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
        normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
    )
    lf = _add_precursor_type_indicators(lf)
    lf = _add_molecular_ion_info(lf, molecular_ion_tolerance_ppm)
    lf = _add_spectral_information_score(lf)

    # Column selection
    existing_cols = lf.collect_schema().names()
    target_cols = [
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

    cols_to_select = [c for c in target_cols if c in existing_cols]

    # Add MSn columns if they exist
    msn_cols = [c for c in existing_cols if c.startswith("msn_")]
    return lf.select(cols_to_select + msn_cols)


def _filter_invalid_entries(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Filters out entries lacking all molecular idetifiers identifiers."""
    return lf.filter(
        pl.col("inchikey").is_not_null()
        | pl.col("smiles").is_not_null()
        | pl.col("inchi").is_not_null()
    )


def _fill_missing_inchikeys(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fill missing inchikeys from inchi or smiles using parallel_rdkit."""
    df = lf.collect()

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


def _compute_precursor_formula(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Vectorized precursor formula computation."""
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
    check_df = (
        lf.filter(
            pl.col("molecular_formula").is_null() & pl.col("inchikey").is_not_null()
        )
        .limit(1)
        .collect()
    )
    if not check_df.is_empty():
        raise NotImplementedError(
            "Molecular formula missing for entry with InChIKey. This will be handled by parallel_rdkit."
        )

    return lf.with_columns(
        pl.col("precursor_formula")
        .map_elements(format_formula_string_to_array, return_dtype=pl.List(pl.Int32))
        .list.to_array(width=num_elements)
        .alias("precursor_formula_array")
    )


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
    instrument_data_columns = pl.selectors.by_name(
        ["instrument", "instrument_type", "ionization", "sample_inlet"]
    )
    qq_mask = pl.any_horizontal(
        instrument_data_columns.str.contains(r"(?i)QQ").fill_null(False)
    )

    return cast(
        polarsFrame,
        data.filter(qq_mask.not_()).with_columns(
            pl.any_horizontal(
                instrument_data_columns.str.contains(r"(?i)LC|HPLC").fill_null(False)
            ).alias("is_LC"),
            pl.any_horizontal(
                instrument_data_columns.str.contains(
                    r"(?i)orbi(?:trap)?|HCD"
                ).fill_null(False)
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
            ).alias("is_orbitrap"),
            pl.any_horizontal(
                instrument_data_columns.str.contains(r"(?i)TOF").fill_null(False)
            ).alias("is_TOF"),
            pl.any_horizontal(
                instrument_data_columns.str.contains(r"(?i)ESI|LC").fill_null(False)
            ).alias("is_ESI"),
        ),
    )


def _extract_collision_energy_values(data: polarsFrame) -> polarsFrame:
    # If collision_energy_list is already present (e.g. from MGF), just ensure mean and flag
    cols = data.collect_schema().names()
    if "collision_energy_list" in cols:
        return cast(
            polarsFrame,
            data.with_columns(
                pl.col("collision_energy_list")
                .list.len()
                .ge(2)
                .fill_null(False)
                .alias("multiple_collision_energies"),
                pl.col("collision_energy_list")
                .list.mean()
                .alias("collision_energy_mean"),
                pl.lit(None).cast(pl.Float64).alias("collision_energy_NCE"),
                pl.lit(None).cast(pl.Float64).alias("collision_energy_ev"),
            ),
        )

    pat_nce = r"(?i)(?:NCE\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)|([0-9]+(?:\.[0-9]+)?)\s*(?:%|(?:\(?NCE\)?)))"
    pat_ev = r"(?i)([0-9]+(?:\.[0-9]+)?)\s*e?V"
    pat_num = r"([0-9]+(?:\.[0-9]+)?)"
    pat_list_content = r"\[(.*?)\]"

    return cast(
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
        )
        .with_columns(
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
        .with_columns(  # if NCE is missing, normalize by using ev and mass. other way roudn would be wrong, since NIST have different normalization masses seamingly in random, but 500 is supposedly the "correct" one. technically meanign less for non orbitrap, but at least this gives us something
            pl.when(
                pl.col("collision_energy_NCE").is_null()
                & pl.col("collision_energy_ev").is_not_null()
            ).then(
                pl.col("collision_energy_ev").mul(500).truediv(pl.col("precursor_mz"))
            )
        )
        .with_columns(
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
        )
        .drop("_bare_energy", "_list_mean", "_collision_energies_ev_raw"),
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
        lf if isinstance(NIST, pl.LazyFrame) else lf.collect(engine="streaming"),
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

    duplicate_pairs = pairs.filter(
        (pl.col("sim_forward") >= threshold) & (pl.col("sim_reverse") >= threshold)
    ).collect()

    if not duplicate_pairs.is_empty():
        # Log total duplicates
        total_dups = duplicate_pairs.select("_dedup_idx_right").n_unique()
        print(f"Deduplication: found {total_dups} duplicate spectra")

        # Log duplicates per file
        dups_per_file = duplicate_pairs.group_by("source_file_right").agg(
            pl.col("_dedup_idx_right").n_unique().alias("num_duplicates")
        )
        for row in dups_per_file.iter_rows(named=True):
            print(f"  {row['source_file_right']}: {row['num_duplicates']} duplicates")

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


def _enrich_with_pubchem(lf: pl.LazyFrame, pubchem_path: Path) -> pl.LazyFrame:
    """get smiles from pubchem based on matching base_inchikey"""
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

    lf = lf.join(pubchem_lf, on="base_inchikey", how="left")

    # Override identifiers with PubChem

    return lf.with_columns(
        pl.coalesce([pl.col("smiles_pubchem"), pl.col("smiles")]).alias("smiles"),
        pl.coalesce([pl.col("inchi_pubchem"), pl.col("inchi")]).alias("inchi"),
        pl.coalesce([pl.col("inchikey_pubchem"), pl.col("inchikey")]).alias("inchikey"),
    ).drop(cs.ends_with("_pubchem"), "cid")


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
