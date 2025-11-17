import polars as pl
import numpy as np
import pytest

from spectral_information import NUM_ELEMENTS

# Helper function to create a formula vector
def create_formula_vector(c=0, h=0, o=0, n=0, p=0, s=0):
    # Assuming order: H, C, N, O ,F, Na , P , S ,Cl ,K ,Br ,I
    formula = [0] * NUM_ELEMENTS
    formula[0] = int(h)
    formula[1] = int(c)
    formula[2] = int(n)
    formula[3] = int(o)
    formula[4] = 0  # F
    formula[5] = 0  # Na
    formula[6] = int(p)
    formula[7] = int(s)
    formula[8] = 0  # Cl
    formula[9] = 0  # K
    formula[10] = 0 # Br
    formula[11] = 0 # I
    return formula

def test_spectral_info_score_simple():
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
        "precursor_formula",
        [precursor1, precursor2],
        dtype=pl.Array(pl.Int32, NUM_ELEMENTS)
    )

    fragments_series = pl.Series(
        "fragment_formulas",
        [
            [fragment1_1, fragment1_2, fragment1_3],
            [fragment2_1, fragment2_2],
        ],
        dtype=pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))
    )

    df = pl.DataFrame(
        {
            "precursor_formula": precursors_series,
            "fragment_formulas": fragments_series,
        }
    )

    # Call the plugin function
    scores = df.select(
        pl.struct(["precursor_formula", "fragment_formulas"])
        .spectral_info.spectral_info_score(distance_metric="l2", ignore_hydrogens=True)
    ).to_series()


    # Assertions
    assert isinstance(scores, pl.Series)
    assert scores.dtype == pl.Float64
    assert len(scores) == 2
    assert scores[0] >= 0.0 # Scores should be non-negative
    assert scores[1] >= 0.0

    # Test with different distance metric
    scores_l1 = df.select(
        pl.struct(["precursor_formula", "fragment_formulas"])
        .spectral_info.spectral_info_score(distance_metric="l1", ignore_hydrogens=True)
    ).to_series()
    assert isinstance(scores_l1, pl.Series)
    assert scores_l1.dtype == pl.Float64

    # Test with ignore_hydrogens=False
    scores_full = df.select(
        pl.struct(["precursor_formula", "fragment_formulas"])
        .spectral_info.spectral_info_score(distance_metric="cosine", ignore_hydrogens=False)
    ).to_series()
    assert isinstance(scores_full, pl.Series)
    assert scores_full.dtype == pl.Float64

    # Test with empty fragments
    precursors_empty_frag = pl.Series(
        "precursor_formula",
        [precursor1],
        dtype=pl.Array(pl.Int32, NUM_ELEMENTS)
    )
    fragments_empty_frag = pl.Series(
        "fragment_formulas",
        [[]],
        dtype=pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))
    )
    
    df_empty = pl.DataFrame(
        {
            "precursor_formula": precursors_empty_frag,
            "fragment_formulas": fragments_empty_frag,
        }
    )

    scores_empty_frag = df_empty.select(
        pl.struct(["precursor_formula", "fragment_formulas"])
        .spectral_info.spectral_info_score(distance_metric="l2", ignore_hydrogens=True)
    ).to_series()

    assert isinstance(scores_empty_frag, pl.Series)
    assert scores_empty_frag.dtype == pl.Float64
    assert scores_empty_frag[0] == 0.0 # Expect 0 score for no fragments
