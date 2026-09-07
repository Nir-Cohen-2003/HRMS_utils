"""Experiment CLI for recovering source-fragment structures from annotated libraries."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from rdkit import rdBase

from compare_library import SimilarityConfiguration, _with_status, attach_structure_identities, build_structure_identity_lookup, compare_same_compound_spectra, spectrum_status_expression
from recovery_chemistry import normalize_precursor_types, parse_source_loss_precursors, prepare_source_candidates, recover_candidate_structures

_RESERVED = {
    "library_row_id", "source_row_id", "normalized_precursor_type", "eligibility_status", "eligibility_reason", "precursor_parse_reason",
    "neutral_loss_formula", "neutral_loss_formula_array", "parent_formula_used_array", "parent_formula_source", "parent_graph_formula_array",
    "retained_target_formula_array", "retained_target_formula", "eligible_bond_count", "matching_bond_count", "matching_component_count",
    "matching_bond_indices", "fragmentation_failure_bond_count", "radical_rejected_bond_count", "recovery_status", "recovered_smiles_raw",
    "recovered_bond_index", "recovered_bond_atom_indices", "dropped_component_formula_array", "dropped_component_formula", "recovered_formal_charge",
    "recovered_smiles_msready", "recovered_inchikey", "recovered_base_inchikey", "standardization_status", "source_spectrum_status",
    "library_structure_match_count", "library_spectrum_match_count", "max_similarity", "best_similarity", "best_library_row_id",
    "best_library_source_file", "best_library_name", "best_library_precursor_type",
}
_ARRAY_COLUMNS = {"molecular_formula_array", "precursor_formula_array"}
_REQUIRED = {"smiles", "precursor_type", "cleaned_normalized_mz", "cleaned_normalized_intensity", "precursor_mz"}


@dataclass(frozen=True)
class ExperimentConfiguration:
    library_path: Path
    output_path: Path
    similarity: SimilarityConfiguration


def _schema_error(message: str) -> ValueError:
    return ValueError(f"Invalid annotated library schema: {message}")


def load_library(configuration: ExperimentConfiguration) -> pl.LazyFrame:
    path = configuration.library_path
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Library does not exist or is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Library is not readable: {path}")
    try:
        schema = pl.scan_parquet(path).collect_schema()
    except Exception as error:
        raise ValueError(f"Unable to read library parquet schema: {path}") from error
    errors: list[str] = []
    missing = sorted(_REQUIRED - set(schema.names()))
    if missing:
        errors.append(f"missing required columns: {', '.join(missing)}")
    if not ({"molecular_formula_array", "molecular_formula"} & set(schema.names())):
        errors.append("missing at least one molecular_formula_array or molecular_formula column")
    if configuration.similarity.require_same_ion_mode and "ion_mode" not in schema.names():
        errors.append("missing ion_mode (use --allow-cross-ion-mode to opt out)")
    for column in ["smiles", "precursor_type", "ion_mode", "molecular_formula", "name", "source_file", "inchi", "inchikey", "base_inchikey"]:
        if column in schema.names() and schema[column] not in {pl.String, pl.Utf8, pl.Null}:
            errors.append(f"{column} must be String, got {schema[column]}")
    for column in _ARRAY_COLUMNS & set(schema.names()):
        dtype = schema[column]
        if dtype == pl.Null:
            continue
        if isinstance(dtype, pl.Array):
            if dtype.width != 12 or dtype.inner not in {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
                errors.append(f"{column} must be an integer Array width 12, got {dtype}")
        else:
            errors.append(f"{column} must be an integer Array width 12, got {dtype}")
    for column in ["cleaned_normalized_mz", "cleaned_normalized_intensity"]:
        if column in schema.names() and schema[column] != pl.Null and not (isinstance(schema[column], pl.List) and schema[column].inner.is_numeric()):
            errors.append(f"{column} must be a numeric List, got {schema[column]}")
    for column in ["precursor_mz"]:
        if column in schema.names() and schema[column] != pl.Null and not schema[column].is_numeric():
            errors.append(f"{column} must be numeric, got {schema[column]}")
    reserved = sorted(_RESERVED & set(schema.names()))
    if reserved:
        errors.append(f"reserved experiment columns already exist: {', '.join(reserved)}")
    if errors:
        raise _schema_error("; ".join(errors))
    # Cast to the stable working contract. Nulls are deliberately retained for row-level statuses.
    columns = schema.names()
    expressions: list[pl.Expr] = [pl.int_range(pl.len(), dtype=pl.UInt64).alias("library_row_id")]
    for column in columns:
        expression = pl.col(column)
        if column in {"smiles", "precursor_type", "ion_mode", "molecular_formula", "name", "source_file", "inchi", "inchikey", "base_inchikey"}:
            expression = expression.cast(pl.String, strict=False)
        elif column in _ARRAY_COLUMNS:
            expression = expression.cast(pl.Array(pl.Int64, 12), strict=False)
        elif column in {"cleaned_normalized_mz", "cleaned_normalized_intensity"}:
            expression = expression.cast(pl.List(pl.Float64), strict=False)
        elif column == "precursor_mz":
            expression = expression.cast(pl.Float64, strict=False)
        elif column == "mslevel":
            expression = expression.cast(pl.Int64, strict=False)
        elif column == "clean_precursor":
            expression = expression.cast(pl.Boolean, strict=False)
        expressions.append(expression.alias(column))
    return pl.scan_parquet(path).select(expressions)


def _audit(precursors: pl.DataFrame) -> dict[str, Any]:
    return {str(row["precursor_selection_status"]): int(row["row_count"]) for row in precursors.to_dicts()}


def _empty_comparison_columns(results: pl.DataFrame) -> pl.DataFrame:
    return results.with_columns(
        pl.lit("missing_spectrum").alias("source_spectrum_status"),
        pl.when(pl.col("recovered_base_inchikey").is_not_null()).then(pl.lit(0, dtype=pl.UInt64)).otherwise(pl.lit(None, dtype=pl.UInt64)).alias("library_structure_match_count"),
        pl.lit(0, dtype=pl.UInt64).alias("library_spectrum_match_count"),
        pl.lit(None, dtype=pl.Float64).alias("max_similarity"), pl.lit(None, dtype=pl.Float64).alias("best_similarity"), pl.lit(None, dtype=pl.UInt64).alias("best_library_row_id"),
        pl.lit(None, dtype=pl.String).alias("best_library_source_file"), pl.lit(None, dtype=pl.String).alias("best_library_name"), pl.lit(None, dtype=pl.String).alias("best_library_precursor_type"),
    )


def validate_result_invariants(results: pl.DataFrame, expected_candidate_count: int, diagnostics: pl.DataFrame | None = None) -> None:
    if len(results) != expected_candidate_count:
        raise AssertionError(f"candidate preservation failed: expected {expected_candidate_count}, got {len(results)}")
    if results.get_column("source_row_id").n_unique() != len(results):
        raise AssertionError("source_row_id is not unique")
    allowed = {"unique", "no_answer", "several_answers"}
    invalid = results.filter(pl.col("recovery_status").is_not_null() & ~pl.col("recovery_status").is_in(allowed))
    if not invalid.is_empty():
        raise AssertionError("unexpected recovery status")
    non_products = results.filter(pl.col("recovery_status").is_in(["no_answer", "several_answers"]) | pl.col("recovery_status").is_null())
    for column in ["recovered_smiles_raw", "recovered_bond_index", "recovered_bond_atom_indices", "recovered_inchikey", "recovered_base_inchikey"]:
        if non_products.filter(pl.col(column).is_not_null()).height:
            raise AssertionError(f"{column} populated without a unique recovery")
    mismatch = results.filter(pl.col("eligibility_reason") == "parent_graph_formula_disagreement")
    if mismatch.filter(pl.col("retained_target_formula_array").is_not_null() | (pl.col("recovery_status").is_not_null())).height:
        raise AssertionError("formula-disagreement rows have recovery data")
    eligible = results.filter(pl.col("eligibility_status") == "eligible")
    if eligible.filter(pl.col("parent_formula_used_array") != pl.col("parent_graph_formula_array")).height:
        raise AssertionError("eligible rows do not have exact metadata/graph formula agreement")
    target_rows = eligible.filter(pl.col("retained_target_formula_array").is_not_null())
    expected_target = target_rows.with_columns((pl.col("parent_formula_used_array") - pl.col("neutral_loss_formula_array")).alias("_expected_target"))
    if expected_target.filter(pl.col("retained_target_formula_array") != pl.col("_expected_target")).height:
        raise AssertionError("retained target is not exactly P-L")
    unique_rows = results.filter(pl.col("recovery_status") == "unique")
    if unique_rows.filter((pl.col("matching_bond_count") != 1) | (pl.col("matching_component_count") != 1) | pl.col("recovered_smiles_raw").is_null()).height:
        raise AssertionError("unique recovery counts/products are inconsistent")
    no_answer = results.filter(pl.col("recovery_status") == "no_answer")
    if no_answer.filter((pl.col("matching_bond_count") != 0) | (pl.col("matching_component_count") != 0) | (pl.col("matching_bond_indices").list.len() != 0)).height:
        raise AssertionError("no_answer matching diagnostics are inconsistent")
    if results.filter(pl.col("recovery_status") == "unique").select("recovered_smiles_raw").n_unique() < 1 and unique_rows.height:
        raise AssertionError("unique recovery has no raw product")
    if results.filter(pl.col("recovered_inchikey").is_not_null() & ~pl.col("recovered_inchikey").str.contains(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")).height:
        raise AssertionError("recovered InChIKey has invalid shape")
    both_scores = results.filter(pl.col("max_similarity").is_not_null() & (pl.col("max_similarity") != pl.col("best_similarity")))
    if not both_scores.is_empty():
        raise AssertionError("best and maximum similarities differ")
    if results.filter(pl.col("library_spectrum_match_count") < 0).height:
        raise AssertionError("submitted spectrum count is negative")
    if diagnostics is not None and not diagnostics.is_empty():
        estimated = int(diagnostics.get_column("estimated_pair_count").sum())
        actual = int(diagnostics.get_column("actual_pair_count").sum())
        if estimated != actual:
            raise AssertionError("planned and materialized pair counts do not reconcile")
        if int(results.get_column("library_spectrum_match_count").sum()) != actual:
            raise AssertionError("submitted pair counts do not reconcile with chunk diagnostics")


def _score_statistics(results: pl.DataFrame) -> dict[str, Any]:
    values = np.asarray(results.filter(pl.col("max_similarity").is_not_null()).get_column("max_similarity").to_list(), dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"count": 0, "missing_count": int(len(results)), "min": None, "mean": None, "sample_std": None, "quantiles": {str(q): None for q in [0, .01, .05, .25, .5, .75, .95, .99, 1]}, "histogram": [0] * 10, "out_of_range_count": 0}
    histogram = [int(np.sum((finite >= low) & (finite < high))) for low, high in [(i / 10, (i + 1) / 10) for i in range(9)]]
    histogram.append(int(np.sum((finite >= .9) & (finite <= 1))))
    return {"count": int(len(finite)), "missing_count": int(len(results) - len(finite)), "min": float(finite.min()), "mean": float(finite.mean()), "sample_std": float(finite.std(ddof=1)) if len(finite) > 1 else None, "quantiles": {str(q): float(np.quantile(finite, q)) for q in [0, .01, .05, .25, .5, .75, .95, .99, 1]}, "histogram": histogram, "out_of_range_count": int(np.sum((finite < 0) | (finite > 1)))}


def _value_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {
        str(row[column]) if row[column] is not None else "null": int(row["count"])
        for row in frame.group_by(column, maintain_order=True).len(name="count").to_dicts()
    }


def _numeric_distribution(frame: pl.DataFrame, column: str) -> dict[str, object]:
    if column not in frame.columns:
        return {"count": 0, "min": None, "max": None, "mean": None}
    values = np.asarray(frame.get_column(column).drop_nulls().to_list(), dtype=float)
    if not values.size:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {"count": int(values.size), "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean())}


def build_distribution_report(results: pl.DataFrame, configuration: ExperimentConfiguration, total_input_rows: int, parser_audit: object, diagnostics: pl.DataFrame, library: pl.DataFrame | None = None) -> dict[str, object]:
    parser_rows = parser_audit if isinstance(parser_audit, list) else []
    ordinary_count = sum(int(row.get("row_count") or 0) for row in parser_rows if row.get("precursor_selection_status") == "not_source_loss")
    unsupported_count = sum(int(row.get("row_count") or 0) for row in parser_rows if row.get("precursor_selection_status") == "unsupported_precursor")
    comparable_rows = results.filter(pl.col("parent_graph_formula_array").is_not_null() & pl.col("parent_formula_used_array").is_not_null())
    agreement_rows = comparable_rows.filter(pl.col("parent_graph_formula_array") == pl.col("parent_formula_used_array"))
    disagreement_rows = comparable_rows.filter(pl.col("parent_graph_formula_array") != pl.col("parent_formula_used_array"))
    exact_agreement = agreement_rows.height
    diagnostics_rows = diagnostics.to_dicts()
    library_standardization: dict[str, object] = {}
    global_key_presence = {"distinct_recovered_keys": 0, "present_anywhere": 0, "absent_anywhere": 0}
    if library is not None:
        library_standardization = {"per_distinct_raw_smiles": _value_counts(library.unique("smiles"), "library_standardization_status"), "weighted_library_rows": _value_counts(library, "library_standardization_status")}
        library_keys = set(library.get_column("library_base_inchikey_recomputed").drop_nulls().to_list())
        recovered_keys = set(results.get_column("recovered_base_inchikey").drop_nulls().to_list())
        global_key_presence = {"distinct_recovered_keys": len(recovered_keys), "present_anywhere": len(recovered_keys & library_keys), "absent_anywhere": len(recovered_keys - library_keys)}
    estimated_pairs = int(sum(int(row.get("estimated_pair_count", 0)) for row in diagnostics_rows))
    actual_pairs = int(sum(int(row.get("actual_pair_count", 0)) for row in diagnostics_rows))
    maximum_chunk = max((int(row.get("actual_pair_count", 0)) for row in diagnostics_rows), default=0)
    return {
        "rdkit_version": rdBase.rdkitVersion,
        "parallel_rdkit_version": "local-build-version-unavailable",
        "configuration": {"library_path": str(configuration.library_path), "output_path": str(configuration.output_path), **asdict(configuration.similarity)},
        "formula_order": ["H", "C", "N", "O", "F", "Na", "P", "S", "Cl", "K", "Br", "I"],
        "total_input_rows": total_input_rows,
        "strict_candidate_count": len(results),
        "precursor_parser_audit": parser_rows,
        "ordinary_non_source_count": ordinary_count,
        "unsupported_precursor_count": unsupported_count,
        "existing_neutral_loss_prefix_count": sum(int(row.get("row_count") or 0) for row in parser_rows if row.get("precursor_parse_reason") == "existing_neutral_loss_prefix"),
        "eligibility_status": _value_counts(results, "eligibility_status"),
        "eligibility_reasons": _value_counts(results, "eligibility_reason"),
        "recovery_status": _value_counts(results, "recovery_status"),
        "standardization_status": _value_counts(results, "standardization_status"),
        "source_spectrum_status": _value_counts(results, "source_spectrum_status"),
        "parent_graph_formula_validation": {
            "candidate_rows_with_graph_formula": int(results.get_column("parent_graph_formula_array").is_not_null().sum()),
            "distinct_parent_structures_with_graph_formula": int(results.filter(pl.col("parent_graph_formula_array").is_not_null()).select("smiles").unique().height),
            "rows_with_valid_metadata_formula": int(results.get_column("parent_formula_used_array").is_not_null().sum()),
            "rows_with_exact_agreement": exact_agreement,
            "distinct_parents_with_metadata_graph_agreement": int(agreement_rows.select("smiles").unique().height) if agreement_rows.height else 0,
            "distinct_parents_with_metadata_graph_disagreement": int(disagreement_rows.select("smiles").unique().height) if disagreement_rows.height else 0,
        },
        "bond_distributions": {"eligible": _numeric_distribution(results, "eligible_bond_count"), "matching": _numeric_distribution(results, "matching_bond_count"), "matching_components": _numeric_distribution(results, "matching_component_count")},
        "cut_failures": {"fragmentation_total": int(results.get_column("fragmentation_failure_bond_count").drop_nulls().sum() or 0), "radical_total": int(results.get_column("radical_rejected_bond_count").drop_nulls().sum() or 0), "rows_with_fragmentation_failure": int(results.filter(pl.col("fragmentation_failure_bond_count") > 0).height), "rows_with_radical_rejection": int(results.filter(pl.col("radical_rejected_bond_count") > 0).height)},
        "recovered_standardization": {"per_source_row": _value_counts(results.filter(pl.col("recovery_status") == "unique"), "standardization_status"), "distinct_raw_products": int(results.filter(pl.col("recovered_smiles_raw").is_not_null()).select("recovered_smiles_raw").unique().height)},
        "library_standardization": library_standardization,
        "global_recovered_key_presence": global_key_presence,
        "structure_coverage": {"identified_recoveries": int(results.filter(pl.col("recovered_base_inchikey").is_not_null()).height), "identified_recoveries_with_references": int(results.filter(pl.col("recovered_base_inchikey").is_not_null() & (pl.col("library_structure_match_count") > 0)).height), "identified_recoveries_without_references": int(results.filter(pl.col("recovered_base_inchikey").is_not_null() & (pl.col("library_structure_match_count") == 0)).height), "structure_match_count_distribution": _numeric_distribution(results, "library_structure_match_count"), "submitted_reference_count_distribution": _numeric_distribution(results, "library_spectrum_match_count")},
        "reference_spectrum_status": _value_counts(library, "source_spectrum_status") if library is not None and "source_spectrum_status" in library.columns else {},
        "source_spectrum_status_distribution": _value_counts(results, "source_spectrum_status"),
        "pair_summary": {"estimated_pair_count": estimated_pairs, "actual_pair_count": actual_pairs, "chunk_count": len(diagnostics_rows), "maximum_actual_chunk_size": maximum_chunk, "largest_group_product": max((int(row.get("group_product", 0)) for row in diagnostics_rows), default=0), "nonfinite_score_count": int(sum(int(row.get("nonfinite_score_count", 0)) for row in diagnostics_rows))},
        "similarity": _score_statistics(results),
        "pair_diagnostics": diagnostics_rows,
        "candidate_rule": "strict source loss; one heavy-atom non-ring single-bond capped cut; exact metadata/graph P agreement and exact P-L component equality",
        "precursor_policy": {"permitted_source_loss_prefixes": ["M", "M+H", "M-H", "M+Na", "M+K", "M+NH4", "M+Cl", "M+FA-H", "M+CHO2"], "ordinary_bodies_include": ["M-H2O-H"], "dehydration_prefixes_rejected": True},
        "capped_component_hypothesis": "RDKit no-dummy capped components generally sum to P+H2; dropped component need not equal literal loss",
        "comparison_policy": "recomputed base InChIKey, same ion mode and MS2 by default; deterministic smallest library row ID tie",
        "viability_note": "Unmatched recoveries are only absent keys in the supplied library, not verified new compounds or proven mechanisms.",
    }


def write_outputs(results: pl.DataFrame, report: dict[str, object], configuration: ExperimentConfiguration) -> None:
    results.sort("source_row_id").write_parquet(configuration.output_path, compression="zstd")
    configuration.output_path.with_suffix(".summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run_experiment(configuration: ExperimentConfiguration) -> pl.DataFrame:
    if not math.isfinite(configuration.similarity.ms2_tolerance_in_ppm) or configuration.similarity.ms2_tolerance_in_ppm <= 0:
        raise ValueError("ms2 tolerance must be finite and positive")
    if configuration.similarity.maximum_exact_pairs_per_chunk <= 0:
        raise ValueError("maximum exact pair count must be positive")
    report_path = configuration.output_path.with_suffix(".summary.json")
    if configuration.output_path.resolve() == report_path.resolve():
        raise ValueError("output and summary paths must be distinct")
    if configuration.output_path.exists() or report_path.exists():
        raise FileExistsError("output or summary already exists")
    if configuration.output_path.resolve() == configuration.library_path.resolve() or report_path.resolve() == configuration.library_path.resolve():
        raise ValueError("output/report cannot overwrite the input library")
    if not configuration.output_path.parent.exists() or not configuration.output_path.parent.is_dir():
        raise FileNotFoundError("output parent directory does not exist")
    if not os.access(configuration.output_path.parent, os.W_OK):
        raise PermissionError(f"output parent directory is not writable: {configuration.output_path.parent}")
    library = load_library(configuration)
    normalized_frame = normalize_precursor_types(library).collect()
    library_frame = normalized_frame
    total_input = library_frame.height
    # Parse exactly one normalized annotation and join counts null-safely, so
    # alternate whitespace/dash spellings cannot duplicate audit totals.
    distinct_precursors = normalized_frame.select("normalized_precursor_type").unique()
    parsed = parse_source_loss_precursors(distinct_precursors.rename({"normalized_precursor_type": "precursor_type"}))
    precursor_counts = normalized_frame.group_by("normalized_precursor_type").agg(pl.len().alias("row_count"))
    audit = parsed.join(precursor_counts, on="normalized_precursor_type", how="left", nulls_equal=True).select("normalized_precursor_type", "precursor_selection_status", "precursor_parse_reason", "row_count")
    if audit.get_column("row_count").fill_null(0).sum() != total_input:
        raise RuntimeError("precursor parser audit does not reconcile to the input row count")
    candidates = prepare_source_candidates(normalized_frame, parsed)
    candidates = recover_candidate_structures(candidates)
    all_smiles = library_frame.get_column("smiles")
    recovered = candidates.filter(pl.col("recovered_smiles_raw").is_not_null()).get_column("recovered_smiles_raw")
    identity = build_structure_identity_lookup(pl.concat([all_smiles, recovered]))
    candidates, library_with_identity = attach_structure_identities(candidates, library_frame, identity)
    input_has_mslevel = "mslevel" in library_frame.columns
    library_with_status = library_with_identity.with_columns(spectrum_status_expression(configuration.similarity, input_has_mslevel).alias("source_spectrum_status"))
    candidates_with_status = _with_status(candidates, configuration.similarity, input_has_mslevel)
    if candidates.is_empty():
        results = _empty_comparison_columns(candidates)
        diagnostics = pl.DataFrame()
    else:
        comparison, diagnostics = compare_same_compound_spectra(candidates_with_status, library_with_status, configuration.similarity, input_has_mslevel)
        results = candidates.join(comparison, on="source_row_id", how="left")
        # Status was computed once for the complete candidate frame and is reused by comparison/reporting.
        if "source_spectrum_status" in results.columns:
            results = results.drop("source_spectrum_status")
        results = results.join(candidates_with_status.select("source_row_id", "source_spectrum_status"), on="source_row_id", how="left")
        for column, dtype in [("library_structure_match_count", pl.UInt64), ("library_spectrum_match_count", pl.UInt64)]:
            if column not in results.columns:
                results = results.with_columns(pl.lit(0, dtype=dtype).alias(column))
            elif column == "library_structure_match_count":
                results = results.with_columns(
                    pl.when(pl.col("recovered_base_inchikey").is_not_null())
                    .then(pl.col(column).fill_null(0))
                    .otherwise(pl.lit(None, dtype=dtype))
                    .cast(dtype)
                )
            else:
                results = results.with_columns(pl.col(column).fill_null(0).cast(dtype))
    results = results.sort("source_row_id")
    validate_result_invariants(results, len(candidates), diagnostics)
    report = build_distribution_report(results, configuration, total_input, audit.to_dicts(), diagnostics, library_with_status)
    write_outputs(results, report, configuration)
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ms2-tolerance-in-ppm", type=float, default=5.0)
    parser.add_argument("--maximum-exact-pairs-per-chunk", type=int, default=100_000)
    parser.add_argument("--allow-cross-ion-mode", action="store_true")
    parser.add_argument("--include-non-ms2", action="store_true")
    args = parser.parse_args()
    configuration = ExperimentConfiguration(args.library, args.output, SimilarityConfiguration(ms2_tolerance_in_ppm=args.ms2_tolerance_in_ppm, require_same_ion_mode=not args.allow_cross_ion_mode, require_ms2_when_available=not args.include_non_ms2, maximum_exact_pairs_per_chunk=args.maximum_exact_pairs_per_chunk))
    results = run_experiment(configuration)
    summary = configuration.output_path.with_suffix(".summary.json").read_text()
    print(summary, end="")
    print(f"Wrote {len(results)} candidate rows to {configuration.output_path}")


if __name__ == "__main__":
    _main()
