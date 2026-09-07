"""Executable regressions for the source-fragment recovery experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from compare_library import (  # noqa: E402
    SimilarityConfiguration,
    build_structure_identity_lookup,
    compare_same_compound_spectra,
    spectrum_status_expression,
)
from recovery_chemistry import (  # noqa: E402
    normalize_precursor_types,
    parse_source_loss_precursors,
    prepare_source_candidates,
    recover_candidate_structures,
)

FORMULA_WIDTH = 12


def _formula(*, hydrogen: int, carbon: int, nitrogen: int = 0, oxygen: int = 0) -> list[int]:
    """Return a formula vector in the project's fixed H,C,N,O,... order."""
    return [hydrogen, carbon, nitrogen, oxygen, 0, 0, 0, 0, 0, 0, 0, 0]


def _recover_rows(
    smiles: list[str],
    formulas: list[list[int] | None],
    formula_text: list[str | None],
    precursor_types: list[str],
) -> pl.DataFrame:
    """Run candidate preparation and graph recovery for small synthetic rows."""
    input_frame = pl.DataFrame(
        {
            "library_row_id": pl.Series("library_row_id", list(range(len(smiles))), dtype=pl.UInt64),
            "smiles": smiles,
            "precursor_type": precursor_types,
            "molecular_formula": formula_text,
            "molecular_formula_array": pl.Series(
                "molecular_formula_array", formulas, dtype=pl.Array(pl.Int64, FORMULA_WIDTH)
            ),
        }
    )
    normalized = normalize_precursor_types(input_frame.lazy()).collect()
    parsed = parse_source_loss_precursors(normalized.select("precursor_type").unique())
    candidates = prepare_source_candidates(normalized, parsed)
    return recover_candidate_structures(candidates)


def _comparison_frames(reference_modes: list[str], reference_ids: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build usable query/reference spectra with a deliberately oversized group."""
    spectrum_mz = [[100.0, 150.0], [100.0, 150.0]]
    spectrum_intensity = [[0.6, 0.4], [0.6, 0.4]]
    candidates = pl.DataFrame(
        {
            "source_row_id": pl.Series("source_row_id", [1, 2], dtype=pl.UInt64),
            "recovery_status": ["unique", "unique"],
            "recovered_base_inchikey": ["AAAAAAAAAAAAAA", "AAAAAAAAAAAAAA"],
            "ion_mode": ["P", "P"],
            "cleaned_normalized_mz": spectrum_mz,
            "cleaned_normalized_intensity": spectrum_intensity,
            "precursor_mz": [300.0, 300.0],
            "source_spectrum_status": ["usable", "usable"],
        }
    )
    library = pl.DataFrame(
        {
            "library_row_id": pl.Series("library_row_id", reference_ids, dtype=pl.UInt64),
            "library_base_inchikey_recomputed": ["AAAAAAAAAAAAAA"] * len(reference_ids),
            "ion_mode": reference_modes,
            "cleaned_normalized_mz": [spectrum_mz[0]] * len(reference_ids),
            "cleaned_normalized_intensity": [spectrum_intensity[0]] * len(reference_ids),
            "precursor_mz": [300.0] * len(reference_ids),
            "source_spectrum_status": ["usable"] * len(reference_ids),
            "source_file": [f"reference-{row_id}" for row_id in reference_ids],
            "name": ["same compound"] * len(reference_ids),
            "precursor_type": ["M"] * len(reference_ids),
        }
    )
    return candidates, library


def test_identical_halves_from_one_bond_are_one_hypothesis() -> None:
    results = _recover_rows(
        ["CCCC"],
        [_formula(hydrogen=10, carbon=4)],
        ["C4H10"],
        ["M-C2H4"],
    )
    row = results.row(0, named=True)
    assert row["recovery_status"] == "unique"
    assert row["matching_bond_count"] == 1
    assert row["matching_component_count"] == 1
    assert row["recovered_smiles_raw"] == "CC"


def test_charged_parent_formula_disagreement_is_not_recovered() -> None:
    results = _recover_rows(
        ["C[NH3+]"],
        [_formula(hydrogen=6, carbon=2, nitrogen=1)],
        ["C2H6N"],
        ["M-CH2"],
    )
    row = results.row(0, named=True)
    assert row["eligibility_status"] == "invalid_formula"
    assert row["eligibility_reason"] == "parent_graph_formula_disagreement"
    assert row["recovery_status"] is None


def test_one_cached_parent_preserves_valid_and_invalid_metadata_rows() -> None:
    results = _recover_rows(
        ["CCO", "CCO"],
        [_formula(hydrogen=6, carbon=2, oxygen=1), None],
        ["C2H6O", "not-a-formula"],
        ["M-CH2", "M-CH2"],
    ).sort("source_row_id")
    assert results.select("smiles").unique().height == 1
    assert results.get_column("parent_graph_formula_array").drop_nulls().len() == 2
    assert results[0, "eligibility_status"] == "eligible"
    assert results[1, "eligibility_reason"] == "invalid_parent_formula_text"
    assert results[1, "recovery_status"] is None
    assert results[1, "matching_bond_count"] is None
    assert results[1, "matching_component_count"] is None
    assert results[1, "matching_bond_indices"] is None


def test_invalid_and_mismatched_spectra_are_rejected() -> None:
    frame = pl.DataFrame(
        {
            "cleaned_normalized_mz": [[100.0, float("nan")], [100.0], [100.0, float("inf")]],
            "cleaned_normalized_intensity": [[1.0, 1.0], [1.0, 2.0], [1.0, 1.0]],
            "precursor_mz": [300.0, 300.0, 300.0],
            "ion_mode": ["P", "P", "P"],
        }
    )
    statuses = frame.with_columns(
        spectrum_status_expression(SimilarityConfiguration(), input_has_mslevel=False).alias("status")
    ).get_column("status").to_list()
    assert statuses == ["invalid_peak_value", "length_mismatch", "invalid_peak_value"]


def test_identity_lookup_filters_and_deduplicates_before_identity_work() -> None:
    identities = build_structure_identity_lookup(pl.Series("smiles", ["CC", None, "", "CC", "CO"]))
    assert identities.get_column("structure_smiles").n_unique() == 2
    assert identities.get_column("structure_smiles").is_in(["CC", "CO"]).all()


def test_opposite_mode_is_excluded_and_ties_choose_smallest_library_row() -> None:
    candidates, library = _comparison_frames(["N", "P", "P"], [30, 20, 10])
    configuration = SimilarityConfiguration(maximum_exact_pairs_per_chunk=2)
    results, diagnostics = compare_same_compound_spectra(candidates, library, configuration, input_has_mslevel=False)
    row = results.filter(pl.col("source_row_id") == 1).row(0, named=True)
    assert row["library_structure_match_count"] == 2
    assert row["library_spectrum_match_count"] == 2
    assert row["best_library_row_id"] == 10
    assert diagnostics.get_column("actual_pair_count").sum() == 4


def test_small_chunks_equal_one_large_chunk_with_oversized_reference_group() -> None:
    candidates, library = _comparison_frames(["P", "P", "P"], [30, 20, 10])
    small_results, small_diagnostics = compare_same_compound_spectra(
        candidates,
        library,
        SimilarityConfiguration(maximum_exact_pairs_per_chunk=2),
        input_has_mslevel=False,
    )
    large_results, large_diagnostics = compare_same_compound_spectra(
        candidates,
        library,
        SimilarityConfiguration(maximum_exact_pairs_per_chunk=100),
        input_has_mslevel=False,
    )
    columns = ["source_row_id", "max_similarity", "best_library_row_id", "library_spectrum_match_count"]
    assert small_results.select(columns).sort("source_row_id").equals(large_results.select(columns).sort("source_row_id"))
    assert small_diagnostics.get_column("actual_pair_count").sum() == 6
    assert large_diagnostics.get_column("actual_pair_count").sum() == 6
    assert small_diagnostics.get_column("group_product").max() == 6
    assert large_diagnostics.get_column("group_product").max() == 6
