import re
from pathlib import Path
from typing import List, Optional, TypeVar, Union, cast

import polars as pl
from parallel_rdkit import (
    inchi_to_smiles_parallel,
    msready_inchi_inchikey_parallel,
    smiles_to_inchikey_parallel,
)

from ..formula_annotation.element_table import ADDUCT_MASSES
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
    dedup_threshold: float = 0.99,
) -> pl.DataFrame:
    """Unified entry point for processing multiple spectral library files with deduplication and enrichment."""
    lazyframes = []
    for f in files:
        lf = process_single_file(
            f,
            raw_fragment_tolerance_ppm=raw_fragment_tolerance_ppm,
            normalized_fragment_tolerance_ppm=normalized_fragment_tolerance_ppm,
            molecular_ion_tolerance_ppm=molecular_ion_tolerance_ppm,
            includes_MSn=includes_MSn,
            lazy=True,
        )
        lazyframes.append(lf)

    combined_lf = pl.concat(lazyframes, how="vertical_relaxed")

    # Filter rows with no molecular info early
    combined_lf = _filter_invalid_entries(combined_lf)

    # Optional PubChem enrichment
    if pubchem_path is not None:
        combined_lf = _enrich_with_pubchem(combined_lf, pubchem_path)

    # Collect for deduplication and MS-ready standardization
    df = combined_lf.collect(engine="streaming")

    # Pairwise deduplication
    df = _deduplicate_spectra(
        df, tolerance_ppm=normalized_fragment_tolerance_ppm, threshold=dedup_threshold
    )

    # MS-ready standardization
    df = _standardize_structures(df)

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
) -> pl.LazyFrame:
    """
    Core processing pipeline.
    including organizing of metadata, spectrum annotation
    """
    lf = _fill_missing_inchikeys(lf)
    lf = _add_base_inchikey(lf)
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
    cols = [
        "name",
        "nist_id",
        "db_id",
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
    ]
    # Add MSn columns if they exist
    existing_cols = lf.collect_schema().names()
    msn_cols = [c for c in existing_cols if c.startswith("msn_")]
    return lf.select(cols + msn_cols)


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
        inchi_list = inchi_only_df["inchi"].to_list()
        smiles_from_inchi = inchi_to_smiles_parallel(inchi_list)
        inchi_to_smiles_map = dict(zip(inchi_list, smiles_from_inchi))

        result_df = result_df.with_columns(
            pl.when(inchi_only_mask)
            .then(pl.col("inchi").map_dict(inchi_to_smiles_map, default=None))
            .otherwise(pl.col("smiles"))
            .alias("smiles")
        )

    # Then convert smiles to inchikey
    smiles_but_no_inchikey_mask = (
        pl.col("inchikey").is_null() & pl.col("smiles").is_not_null()
    )

    if result_df.filter(smiles_but_no_inchikey_mask).height > 0:
        smiles_df = result_df.filter(smiles_but_no_inchikey_mask)
        smiles_list = smiles_df["smiles"].to_list()
        inchikeys = smiles_to_inchikey_parallel(smiles_list)
        smiles_to_inchikey_map = dict(zip(smiles_list, inchikeys))

        result_df = result_df.with_columns(
            pl.when(smiles_but_no_inchikey_mask)
            .then(pl.col("smiles").map_dict(smiles_to_inchikey_map, default=None))
            .otherwise(pl.col("inchikey"))
            .alias("inchikey")
        )

    return result_df.lazy()


def _add_base_inchikey(lf: pl.LazyFrame) -> pl.LazyFrame:
    if "base_inchikey" in lf.collect_schema().names():
        return lf
    return lf.with_columns(
        pl.col("inchikey").str.extract(r"(.+?)-").alias("base_inchikey")
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


def _annotate_and_filter_metadata(data: polarsFrame) -> polarsFrame:
    instrument_data_columns = pl.selectors.by_name(
        ["instrument", "instrument_type", "ionization"]
    )
    qq_mask = pl.any_horizontal(
        instrument_data_columns.str.contains(r"(?i)QQ").fill_null(False)
    )

    return cast(
        polarsFrame,
        data.filter(qq_mask.not_()).with_columns(
            pl.any_horizontal(
                instrument_data_columns.str.contains(r"(?i)LC").fill_null(False)
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
            .alias("collision_energy_NCE"),
            pl.col("collision_energy_raw")
            .str.extract(pat_ev, group_index=1)
            .cast(pl.Float64, strict=False)
            .alias("collision_energy_ev"),
            pl.col("collision_energy_raw")
            .str.extract(pat_list_content, group_index=1)
            .str.extract_all(r"\d+(?:\.\d+)?")
            .list.eval(pl.element().cast(pl.Float64, strict=False))
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
        .drop("_bare_energy", "_list_mean"),
    )


def _annotate_spectra(
    data: polarsFrame,
    raw_fragment_tolerance_ppm: float,
    normalized_fragment_tolerance_ppm: float,
) -> polarsFrame:
    adduct_df = pl.DataFrame(
        {
            "precursor_type": list(ADDUCT_MASSES.keys()),
            "adduct_mass": list(ADDUCT_MASSES.values()),
        }
    )

    if isinstance(data, pl.LazyFrame):
        data_frame = data.join(adduct_df.lazy(), on="precursor_type", how="left")
    else:
        data_frame = data.join(adduct_df, on="precursor_type", how="left")

    return cast(
        polarsFrame,
        data_frame.with_columns(
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
    df: pl.DataFrame, tolerance_ppm: float, threshold: float
) -> pl.DataFrame:
    """Pairwise deduplication based on explained intensity."""
    df = df.with_row_index("_dedup_idx")

    # Join on base_inchikey to find potential duplicates
    pairs = (
        df.select(
            [
                "_dedup_idx",
                "base_inchikey",
                "cleaned_normalized_mz",
                "cleaned_normalized_intensity",
                "precursor_mz",
            ]
        )
        .join(
            df.select(
                [
                    "_dedup_idx",
                    "base_inchikey",
                    "cleaned_normalized_mz",
                    "cleaned_normalized_intensity",
                    "precursor_mz",
                ]
            ),
            on="base_inchikey",
            suffix="_right",
        )
        .filter(pl.col("_dedup_idx") < pl.col("_dedup_idx_right"))
    )

    if pairs.is_empty():
        return df.drop("_dedup_idx")

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
            ms2_tolerance_in_ppm=tolerance_ppm, permissive=True
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
            ms2_tolerance_in_ppm=tolerance_ppm, permissive=True
        )
        .alias("sim_reverse"),
    )

    to_remove = (
        pairs.filter(
            (pl.col("sim_forward") >= threshold) & (pl.col("sim_reverse") >= threshold)
        )
        .select("_dedup_idx_right")
        .unique()
    )

    return df.filter(~pl.col("_dedup_idx").is_in(to_remove["_dedup_idx_right"])).drop(
        "_dedup_idx"
    )


def _enrich_with_pubchem(lf: pl.LazyFrame, pubchem_path: Path) -> pl.LazyFrame:
    """get smiles from pubchem based on matching base_inchikey"""
    pubchem_lf = (
        pl.scan_parquet(str(pubchem_path), low_memory=True)
        .select(
            [
                pl.col("CID").alias("cid"),
                pl.col("InChIKey").alias("inchikey"),
                pl.col("SMILES").alias("smiles"),
                pl.col("InChI").alias("inchi"),
                pl.col("Formula").alias("formula"),
                pl.col("monoisotopic_mass"),
            ]
        )
        .with_columns(
            pl.col("inchikey").str.split("-").list.get(0).alias("base_inchikey")
        )
        # Deduplicate pubchem by base_inchikey
        .unique(subset="base_inchikey")
    )

    pubchem_lf = pubchem_lf

    lf = lf.join(pubchem_lf, on="base_inchikey", how="left", suffix="_pubchem")

    # Fill missing identifiers
    return lf.with_columns(
        pl.when(pl.col("smiles").is_null() | (pl.col("smiles") == ""))
        .then(pl.col("smiles_pubchem"))
        .otherwise(pl.col("smiles"))
        .alias("smiles"),
        pl.when(pl.col("inchi").is_null() | (pl.col("inchi") == ""))
        .then(pl.col("inchi_pubchem"))
        .otherwise(pl.col("inchi"))
        .alias("inchi"),
        pl.when(pl.col("inchikey").is_null() | (pl.col("inchikey") == ""))
        .then(pl.col("inchikey_pubchem"))
        .otherwise(pl.col("inchikey"))
        .alias("inchikey"),
    ).drop(
        [
            c
            for c in lf.collect_schema().names()
            if c.endswith("_pubchem") or c in ["cid", "formula", "monoisotopic_mass"]
        ]
    )


def _standardize_structures(df: pl.DataFrame) -> pl.DataFrame:

    # Get unique SMILES per base_inchikey
    mapping_df = df.group_by("base_inchikey").agg(pl.col("smiles").drop_nulls().first())

    if mapping_df["smiles"].null_count() > 0:
        missing = mapping_df.filter(pl.col("smiles").is_null())[
            "base_inchikey"
        ].to_list()
        raise ValueError(f"Missing SMILES for base_inchikeys: {missing}")

    smiles_list = mapping_df["smiles"].to_list()
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
    return df.with_columns(
        pl.col("msready_smiles").alias("smiles"),
        pl.col("msready_inchi").alias("inchi"),
        pl.col("msready_inchikey").alias("inchikey"),
        pl.col("msready_inchikey").str.extract(r"(.+?)-").alias("base_inchikey"),
    ).drop(["msready_smiles", "msready_inchi", "msready_inchikey"])
