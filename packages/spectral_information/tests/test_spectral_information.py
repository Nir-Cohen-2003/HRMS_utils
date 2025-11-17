import polars as pl
import numpy as np
import pytest

from spectral_information import tree_spectral_info_score, NUM_ELEMENTS

# Helper function to create a formula vector
def create_formula_vector(c=0, h=0, o=0, n=0, p=0, s=0):
    # Assuming order: C, H, O, N, P, S
    return [int(c), int(h), int(o), int(n), int(p), int(s)]

def test_tree_spectral_info_score_simple():
    # Define a simple precursor and fragments
    precursor1 = create_formula_vector(c=6, h=12, o=6) # Glucose
    fragment1_1 = create_formula_vector(c=3, h=6, o=3)
    fragment1_2 = create_formula_vector(c=2, h=4, o=2)
    fragment1_3 = create_formula_vector(c=1, h=2, o=1)

    precursor2 = create_formula_vector(c=5, h=10, o=5) # Ribose
    fragment2_1 = create_formula_vector(c=2, h=4, o=2)
    fragment2_2 = create_formula_vector(c=3, h=6, o=3)

    # Create Polars Series for precursors and fragments
    precursors_series = pl.Series(
        "precursors",
        [precursor1, precursor2],
        dtype=pl.List(pl.Int32)
    )

    fragments_series = pl.Series(
        "fragments",
        [
            [fragment1_1, fragment1_2, fragment1_3],
            [fragment2_1, fragment2_2],
        ],
        dtype=pl.List(pl.List(pl.Int32))
    )

    # Call the plugin function
    scores = tree_spectral_info_score(
        precursors=precursors_series,
        fragments=fragments_series,
        distance_metric="l2",
        ignore_hydrogens=True,
    )

    # Assertions
    assert isinstance(scores, pl.Series)
    assert scores.dtype == pl.Float64
    assert len(scores) == 2
    assert scores[0] >= 0.0 # Scores should be non-negative
    assert scores[1] >= 0.0

    # Test with different distance metric
    scores_l1 = tree_spectral_info_score(
        precursors=precursors_series,
        fragments=fragments_series,
        distance_metric="l1",
        ignore_hydrogens=True,
    )
    assert isinstance(scores_l1, pl.Series)
    assert scores_l1.dtype == pl.Float64

    # Test with ignore_hydrogens=False
    scores_full = tree_spectral_info_score(
        precursors=precursors_series,
        fragments=fragments_series,
        distance_metric="cosine",
        ignore_hydrogens=False,
    )
    assert isinstance(scores_full, pl.Series)
    assert scores_full.dtype == pl.Float64

    # Test with empty fragments
    precursors_empty_frag = pl.Series(
        "precursors",
        [precursor1],
        dtype=pl.List(pl.Int32)
    )
    fragments_empty_frag = pl.Series(
        "fragments",
        [[]],
        dtype=pl.List(pl.List(pl.Int32))
    )
    scores_empty_frag = tree_spectral_info_score(
        precursors=precursors_empty_frag,
        fragments=fragments_empty_frag,
        distance_metric="l2",
        ignore_hydrogens=True,
    )
    assert isinstance(scores_empty_frag, pl.Series)
    assert scores_empty_frag.dtype == pl.Float64
    assert scores_empty_frag[0] == 0.0 # Expect 0 score for no fragments

    # Test with empty precursor
    precursors_empty_prec = pl.Series(
        "precursors",
        [[]],
        dtype=pl.List(pl.Int32)
    )
    fragments_empty_prec = pl.Series(
        "fragments",
        [[fragment1_1]],
        dtype=pl.List(pl.List(pl.Int32))
    )
    scores_empty_prec = tree_spectral_info_score(
        precursors=precursors_empty_prec,
        fragments=fragments_empty_prec,
        distance_metric="l2",
        ignore_hydrogens=True,
    )
    assert isinstance(scores_empty_prec, pl.Series)
    assert scores_empty_prec.dtype == pl.Float64
    assert scores_empty_prec[0] == 0.0 # Expect 0 score for empty precursor
