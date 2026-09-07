"""MS-ready identity mapping and bounded exact spectral comparison."""
from __future__ import annotations

from dataclasses import dataclass
import re
import polars as pl
import hrms_utils.hrms_core  # noqa: F401 - registers the Polars expression namespace
from parallel_rdkit import msready_smiles_parallel, smiles_to_inchikey_parallel


@dataclass(frozen=True)
class SimilarityConfiguration:
    ms2_tolerance_in_ppm: float = 5.0
    require_same_ion_mode: bool = True
    require_ms2_when_available: bool = True
    clean_spectra_first: bool = False
    ignore_precursor: bool = True
    noise_threshold: float = 0.001
    maximum_exact_pairs_per_chunk: int = 100_000

_KEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def _empty_identity() -> pl.DataFrame:
    return pl.DataFrame(schema={"structure_smiles": pl.String, "structure_smiles_msready": pl.String, "structure_inchikey": pl.String, "structure_base_inchikey": pl.String, "structure_standardization_status": pl.String})


def build_structure_identity_lookup(smiles: pl.Series) -> pl.DataFrame:
    values = smiles.filter(smiles.is_not_null() & (smiles.str.strip_chars() != "")).unique().to_list()
    if not values:
        return _empty_identity()
    msready = msready_smiles_parallel(values, silent=True)
    if len(msready) != len(values):
        raise RuntimeError("parallel_rdkit returned the wrong number of MS-ready structures")
    statuses: list[str] = []
    keys: list[str | None] = []
    base_keys: list[str | None] = []
    for value in msready:
        if not value:
            statuses.append("msready_failed")
            keys.append(None)
            base_keys.append(None)
        elif value == "<INORGANIC>":
            statuses.append("inorganic")
            keys.append(None)
            base_keys.append(None)
        else:
            statuses.append("pending")
            keys.append(None)
            base_keys.append(None)
    successful_indices = [index for index, status in enumerate(statuses) if status == "pending"]
    successful_values = [msready[index] for index in successful_indices]
    unique_ready = list(dict.fromkeys(successful_values))
    generated = smiles_to_inchikey_parallel(unique_ready)
    if len(generated) != len(unique_ready):
        raise RuntimeError("parallel_rdkit returned the wrong number of InChIKeys")
    key_by_ready: dict[str, str | None] = {}
    for value, key in zip(unique_ready, generated):
        key_by_ready[value] = key if _KEY_PATTERN.fullmatch(key or "") else None
    for index in successful_indices:
        key = key_by_ready[msready[index]]
        if key is None:
            statuses[index] = "inchikey_failed"
        else:
            statuses[index] = "success"
            keys[index] = key
            base_keys[index] = key.split("-", 1)[0]
    return pl.DataFrame({"structure_smiles": values, "structure_smiles_msready": msready, "structure_inchikey": keys, "structure_base_inchikey": base_keys, "structure_standardization_status": statuses})


def attach_structure_identities(candidates: pl.DataFrame, library: pl.DataFrame | pl.LazyFrame, identity_lookup: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    candidate_smiles = candidates.select(pl.col("recovered_smiles_raw").alias("structure_smiles")).filter(pl.col("structure_smiles").is_not_null()).unique()
    candidate_identity = identity_lookup.filter(pl.col("structure_smiles").is_in(candidate_smiles.get_column("structure_smiles"))) if not candidate_smiles.is_empty() else _empty_identity()
    candidate_identity = candidate_identity.rename({"structure_smiles": "recovered_smiles_raw", "structure_smiles_msready": "recovered_smiles_msready", "structure_inchikey": "recovered_inchikey", "structure_base_inchikey": "recovered_base_inchikey", "structure_standardization_status": "standardization_status_identity"})
    replace_columns = [column for column in ["recovered_smiles_msready", "recovered_inchikey", "recovered_base_inchikey", "standardization_status"] if column in candidates.columns]
    enriched = candidates.drop(replace_columns).join(candidate_identity, on="recovered_smiles_raw", how="left").rename({"standardization_status_identity": "standardization_status"})
    enriched = enriched.with_columns(pl.when(pl.col("recovery_status") == "unique").then(pl.col("standardization_status")).otherwise(pl.lit("not_attempted")).alias("standardization_status"))
    library_frame = library.collect() if isinstance(library, pl.LazyFrame) else library
    library_smiles = library_frame.get_column("smiles") if "smiles" in library_frame.columns else pl.Series("smiles", [], dtype=pl.String)
    library_identity = identity_lookup.filter(pl.col("structure_smiles").is_in(library_smiles))
    library_enriched = library_frame.join(library_identity.select(pl.col("structure_smiles").alias("smiles"), pl.col("structure_smiles_msready").alias("library_smiles_msready"), pl.col("structure_inchikey").alias("library_inchikey_recomputed"), pl.col("structure_base_inchikey").alias("library_base_inchikey_recomputed"), pl.col("structure_standardization_status").alias("library_standardization_status")), on="smiles", how="left")
    return enriched, library_enriched.with_columns(
        pl.when(pl.col("smiles").is_null() | (pl.col("smiles").str.strip_chars() == ""))
        .then(pl.lit("missing_smiles"))
        .otherwise(pl.col("library_standardization_status"))
        .alias("library_standardization_status")
    )


def spectrum_status_expression(configuration: SimilarityConfiguration, input_has_mslevel: bool) -> pl.Expr:
    # Keep validation in one ordered expression so callers can inspect/filter lazily.
    mz = pl.col("cleaned_normalized_mz")
    intensity = pl.col("cleaned_normalized_intensity")
    null_mz = mz.list.eval(pl.element().is_null()).list.any().fill_null(False)
    null_intensity = intensity.list.eval(pl.element().is_null()).list.any().fill_null(False)
    finite_mz = mz.list.eval(pl.element().is_finite() & (pl.element() > 0)).list.all().fill_null(False)
    finite_intensity = intensity.list.eval(pl.element().is_finite() & (pl.element() >= 0)).list.all().fill_null(False)
    unsorted = mz.list.diff().list.eval(pl.element() < 0).list.any()
    missing_ion_mode = ((pl.col("ion_mode").is_null() | (pl.col("ion_mode").str.strip_chars() == "")).fill_null(False) if configuration.require_same_ion_mode else pl.lit(False))
    return (pl.when(mz.is_null() | intensity.is_null()).then(pl.lit("missing_spectrum"))
        .when((mz.list.len() == 0) | (intensity.list.len() == 0)).then(pl.lit("empty_spectrum"))
        .when(mz.list.len() != intensity.list.len()).then(pl.lit("length_mismatch"))
        .when(null_mz | null_intensity | ~finite_mz | ~finite_intensity).then(pl.lit("invalid_peak_value"))
        .when(intensity.list.sum() <= 0).then(pl.lit("zero_total_intensity"))
        .when(unsorted).then(pl.lit("unsorted_mz"))
        .when(pl.col("precursor_mz").is_null() | ~pl.col("precursor_mz").is_finite() | (pl.col("precursor_mz") <= 0)).then(pl.lit("invalid_precursor_mz"))
        .when(missing_ion_mode).then(pl.lit("missing_ion_mode"))
        .when(input_has_mslevel and configuration.require_ms2_when_available & (pl.col("mslevel") != 2).fill_null(True)).then(pl.lit("non_ms2"))
        .otherwise(pl.lit("usable")))


def _with_status(frame: pl.DataFrame, configuration: SimilarityConfiguration, input_has_mslevel: bool) -> pl.DataFrame:
    return frame.with_columns(spectrum_status_expression(configuration, input_has_mslevel).alias("source_spectrum_status"))


def _prepare_comparison_frames(candidates: pl.DataFrame, library_frame: pl.DataFrame, configuration: SimilarityConfiguration, input_has_mslevel: bool) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    all_library = library_frame if "source_spectrum_status" in library_frame.columns else _with_status(library_frame, configuration, input_has_mslevel)
    all_candidates = candidates if "source_spectrum_status" in candidates.columns else _with_status(candidates, configuration, input_has_mslevel)
    queries = all_candidates.filter((pl.col("recovery_status") == "unique") & pl.col("recovered_base_inchikey").is_not_null() & (pl.col("source_spectrum_status") == "usable"))
    references = all_library.filter(pl.col("library_base_inchikey_recomputed").is_not_null() & (pl.col("source_spectrum_status") == "usable"))
    mode_keys = ["library_base_inchikey_recomputed"] + (["ion_mode"] if configuration.require_same_ion_mode else [])
    query_keys = ["recovered_base_inchikey"] + (["ion_mode"] if configuration.require_same_ion_mode else [])
    counts = all_library.filter(pl.col("library_base_inchikey_recomputed").is_not_null()).group_by(mode_keys).agg(pl.len().alias("library_structure_match_count"))
    queries = queries.join(counts.rename({"library_base_inchikey_recomputed": "recovered_base_inchikey"}), on=query_keys, how="left").with_columns(pl.col("library_structure_match_count").fill_null(0).cast(pl.UInt64))
    references = references.join(queries.select(query_keys).unique(), left_on=mode_keys, right_on=query_keys, how="semi")
    return queries, references, counts


def prepare_comparison_groups(candidates: pl.DataFrame, library: pl.LazyFrame, configuration: SimilarityConfiguration, input_has_mslevel: bool) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # The LazyFrame is collected once and all validation is vectorized.
    return _prepare_comparison_frames(candidates, library.collect(), configuration, input_has_mslevel)


def plan_exact_pair_chunks(group_counts: pl.DataFrame, maximum_pairs: int) -> pl.DataFrame:
    if maximum_pairs <= 0:
        raise ValueError("maximum_exact_pairs_per_chunk must be positive")
    records: list[dict[str, object]] = []
    chunk_index = 0
    sort_columns = [column for column in ["recovered_base_inchikey", "ion_mode"] if column in group_counts.columns]
    ordered_groups = group_counts.sort(sort_columns) if sort_columns else group_counts
    for group in ordered_groups.to_dicts():
        q_count = int(group.get("query_count", 0))
        r_count = int(group.get("reference_count", 0))
        if not q_count or not r_count:
            continue
        ref_block = min(r_count, maximum_pairs)
        query_block = max(1, maximum_pairs // ref_block)
        for query_offset in range(0, q_count, query_block):
            for reference_offset in range(0, r_count, ref_block):
                q_len = min(query_block, q_count - query_offset)
                r_len = min(ref_block, r_count - reference_offset)
                records.append({**{key: group[key] for key in group if key not in {"query_count", "reference_count"}}, "group_product": q_count * r_count, "query_offset": query_offset, "query_length": q_len, "reference_offset": reference_offset, "reference_length": r_len, "estimated_pair_count": q_len * r_len, "chunk_index": chunk_index})
                chunk_index += 1
    return pl.DataFrame(records) if records else pl.DataFrame(schema={"query_offset": pl.UInt64, "query_length": pl.UInt64, "reference_offset": pl.UInt64, "reference_length": pl.UInt64, "group_product": pl.UInt64, "estimated_pair_count": pl.UInt64, "chunk_index": pl.UInt64})


def score_exact_pair_chunk(queries: pl.DataFrame, references: pl.DataFrame, configuration: SimilarityConfiguration) -> pl.DataFrame:
    if queries.is_empty() or references.is_empty():
        return pl.DataFrame(schema={"source_row_id": pl.UInt64, "library_row_id": pl.UInt64, "similarity": pl.Float64})
    left = queries.with_row_index("_query_index")
    right = references.with_row_index("_reference_index")
    pairs = left.join(right, how="cross", suffix="_right").with_columns(pl.struct(mz1=pl.col("cleaned_normalized_mz"), intensities1=pl.col("cleaned_normalized_intensity"), mz2=pl.col("cleaned_normalized_mz_right"), intensities2=pl.col("cleaned_normalized_intensity_right"), precursor_mz1=pl.col("precursor_mz"), precursor_mz2=pl.col("precursor_mz_right")).alias("spectra")).with_columns(pl.col("spectra").spectral_similarity.dotprod_similarity(ms2_tolerance_in_ppm=configuration.ms2_tolerance_in_ppm, clean_spectra_first=configuration.clean_spectra_first, ignore_precursor=configuration.ignore_precursor, noise_threshold=configuration.noise_threshold).alias("similarity"))
    return pairs.select("source_row_id", "library_row_id", "similarity")


def _empty_diagnostics() -> pl.DataFrame:
    return pl.DataFrame(schema={"chunk_index": pl.UInt64, "group_product": pl.UInt64, "estimated_pair_count": pl.UInt64, "actual_pair_count": pl.UInt64, "nonfinite_score_count": pl.UInt64})


def compare_same_compound_spectra(candidates: pl.DataFrame, library: pl.DataFrame | pl.LazyFrame, configuration: SimilarityConfiguration, input_has_mslevel: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    library_frame = library.collect() if isinstance(library, pl.LazyFrame) else library
    queries, references, structure_counts = _prepare_comparison_frames(candidates, library_frame, configuration, input_has_mslevel)
    base = candidates.select("source_row_id").with_columns(pl.lit(0, dtype=pl.UInt64).alias("library_spectrum_match_count"))
    candidate_keys = ["recovered_base_inchikey"] + (["ion_mode"] if configuration.require_same_ion_mode else [])
    candidate_structure_counts = candidates.select(["source_row_id"] + candidate_keys).join(structure_counts.rename({"library_base_inchikey_recomputed": "recovered_base_inchikey"}), on=candidate_keys, how="left").select("source_row_id", "library_structure_match_count")
    base = base.join(candidate_structure_counts, on="source_row_id", how="left").with_columns(
        pl.when(candidates.get_column("recovered_base_inchikey").is_not_null())
        .then(pl.col("library_structure_match_count").fill_null(0))
        .otherwise(pl.lit(None, dtype=pl.UInt64))
        .cast(pl.UInt64)
        .alias("library_structure_match_count")
    )
    if queries.is_empty() or references.is_empty():
        return base.with_columns(pl.lit(None, dtype=pl.Float64).alias("max_similarity"), pl.lit(None, dtype=pl.Float64).alias("best_similarity"), pl.lit(None, dtype=pl.UInt64).alias("best_library_row_id"), pl.lit(None, dtype=pl.String).alias("best_library_source_file"), pl.lit(None, dtype=pl.String).alias("best_library_name"), pl.lit(None, dtype=pl.String).alias("best_library_precursor_type")), _empty_diagnostics()
    keys = ["recovered_base_inchikey"] + (["ion_mode"] if configuration.require_same_ion_mode else [])
    reference_keys = ["library_base_inchikey_recomputed"] + (["ion_mode"] if configuration.require_same_ion_mode else [])
    q_counts = queries.group_by(keys).agg(pl.len().alias("query_count"))
    r_counts = references.group_by(reference_keys).agg(pl.len().alias("reference_count")).rename({"library_base_inchikey_recomputed": "recovered_base_inchikey"})
    groups = q_counts.join(r_counts, on=keys, how="inner")
    plans = plan_exact_pair_chunks(groups, configuration.maximum_exact_pairs_per_chunk)
    diagnostics: list[dict[str, object]] = []
    winners: list[pl.DataFrame] = []
    for plan in plans.to_dicts():
        group_filter = [pl.col(key) == plan[key] for key in keys]
        q_group = queries.filter(pl.all_horizontal(group_filter)).sort("source_row_id").slice(int(plan["query_offset"]), int(plan["query_length"]))
        r_group = references.filter(pl.all_horizontal([pl.col(reference_key) == plan[group_key] for reference_key, group_key in zip(reference_keys, keys)])).sort("library_row_id").slice(int(plan["reference_offset"]), int(plan["reference_length"]))
        scored = score_exact_pair_chunk(q_group, r_group, configuration)
        expected = int(plan["estimated_pair_count"])
        if len(scored) != expected or expected > configuration.maximum_exact_pairs_per_chunk:
            raise RuntimeError("exact pair chunk violated its declared pair budget")
        finite = scored.filter(pl.col("similarity").is_finite())
        winners.append(finite.sort(["source_row_id", "similarity", "library_row_id"], descending=[False, True, False]).unique("source_row_id", keep="first"))
        diagnostics.append({"chunk_index": plan["chunk_index"], "group_product": plan["group_product"], "estimated_pair_count": expected, "actual_pair_count": len(scored), "nonfinite_score_count": len(scored) - len(finite)})
        counts = scored.group_by("source_row_id").agg(pl.len().alias("submitted"))
        base = base.join(counts, on="source_row_id", how="left").with_columns(
            (pl.col("library_spectrum_match_count") + pl.col("submitted").fill_null(0)).alias("library_spectrum_match_count")
        ).drop("submitted")
    all_winners = pl.concat(winners).sort(["source_row_id", "similarity", "library_row_id"], descending=[False, True, False]).unique("source_row_id", keep="first") if winners else pl.DataFrame()
    if not all_winners.is_empty():
        metadata_columns = ["library_row_id"]
        for column in ["source_file", "name", "precursor_type"]:
            if column in references.columns:
                metadata_columns.append(column)
        metadata = references.select(metadata_columns)
        for column in ["source_file", "name", "precursor_type"]:
            if column not in metadata.columns:
                metadata = metadata.with_columns(pl.lit(None, dtype=pl.String).alias(column))
        aggregate = all_winners.join(metadata, on="library_row_id", how="left").select("source_row_id", pl.col("similarity").alias("max_similarity"), pl.col("similarity").alias("best_similarity"), "library_row_id", pl.col("source_file").alias("best_library_source_file"), pl.col("name").alias("best_library_name"), pl.col("precursor_type").alias("best_library_precursor_type"))
    else:
        aggregate = pl.DataFrame(schema={"source_row_id": pl.UInt64, "max_similarity": pl.Float64, "best_similarity": pl.Float64, "library_row_id": pl.UInt64, "best_library_source_file": pl.String, "best_library_name": pl.String, "best_library_precursor_type": pl.String})
    return base.join(aggregate, on="source_row_id", how="left").rename({"library_row_id": "best_library_row_id"}), pl.DataFrame(diagnostics) if diagnostics else _empty_diagnostics()
