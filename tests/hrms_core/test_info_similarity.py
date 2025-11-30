import polars as pl
from hrms_utils.hrms_core import NUM_ELEMENTS

def test_info_similarity():
    # C6H12O6
    precursor = [0.0] * NUM_ELEMENTS
    precursor[0] = 12.0
    precursor[1] = 6.0
    precursor[3] = 6.0

    # CH2O
    frag1_1 = [0.0] * NUM_ELEMENTS
    frag1_1[0] = 2.0
    frag1_1[1] = 1.0
    frag1_1[3] = 1.0

    # C2H4O2
    frag1_2 = [0.0] * NUM_ELEMENTS
    frag1_2[0] = 4.0
    frag1_2[1] = 2.0
    frag1_2[3] = 2.0

    # C3H6O3
    frag2_1 = [0.0] * NUM_ELEMENTS
    frag2_1[0] = 6.0
    frag2_1[1] = 3.0
    frag2_1[3] = 3.0

    df = pl.DataFrame(
        {
            "precursor_formula1": [precursor],
            "fragment_formulas1": [[frag1_1, frag1_2]],
            "precursor_formula2": [precursor],
            "fragment_formulas2": [[frag1_2, frag2_1]],
        }
    )

    result_df = df.with_columns(
        scores=pl.struct(
            [
                "precursor_formula1",
                "fragment_formulas1",
                "precursor_formula2",
                "fragment_formulas2",
            ]
        ).spectral_similarity.info_similarity()
    ).unnest("scores")

    assert "spec1_info" in result_df.columns
    assert "spec2_info" in result_df.columns
    assert "union_info" in result_df.columns
    assert "diff1_info" in result_df.columns
    assert "diff2_info" in result_df.columns

    result = result_df.row(0, named=True)

    # Check that scores are calculated and are floats
    assert isinstance(result["spec1_info"], float)
    assert isinstance(result["spec2_info"], float)
    assert isinstance(result["union_info"], float)
    assert isinstance(result["diff1_info"], float)
    assert isinstance(result["diff2_info"], float)
    
    # spec1_info and spec2_info should be different
    assert result["spec1_info"] != result["spec2_info"]
    
    # union_info should be greater than spec1_info and spec2_info
    assert result["union_info"] > result["spec1_info"]
    assert result["union_info"] > result["spec2_info"]

    # The info of a single fragment should be 0, as there are no parent nodes
    assert result["diff1_info"] == 0.0
    assert result["diff2_info"] == 0.0

    # test with different precursors, should result in all 0
    precursor2 = precursor.copy()
    precursor2[1] = 7.0 # C7...

    df_diff_precursor = pl.DataFrame(
        {
            "precursor_formula1": [precursor],
            "fragment_formulas1": [[frag1_1, frag1_2]],
            "precursor_formula2": [precursor2],
            "fragment_formulas2": [[frag1_2, frag2_1]],
        }
    )

    result_df_diff = df_diff_precursor.with_columns(
        scores=pl.struct(
            [
                "precursor_formula1",
                "fragment_formulas1",
                "precursor_formula2",
                "fragment_formulas2",
            ]
        ).spectral_similarity.info_similarity()
    ).unnest("scores")
    
    result_diff = result_df_diff.row(0, named=True)
    
    assert result_diff["spec1_info"] == 0.0
    assert result_diff["spec2_info"] == 0.0
    assert result_diff["union_info"] == 0.0
    assert result_diff["diff1_info"] == 0.0
    assert result_diff["diff2_info"] == 0.0

