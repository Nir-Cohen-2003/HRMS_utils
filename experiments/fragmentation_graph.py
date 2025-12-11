"""
Fragmentation graph construction from MSP data.

This module constructs directed fragmentation graphs (parent → child) from MSP file parsing results.
The graphs represent fragmentation cascades where edges indicate possible parent-child relationships
based on formula subtraction rules and MS level constraints.

Why MS level filtering matters: In tandem MS experiments, MS³ fragments can only arise from the 
precursor ion selected from MS². A fragment observed in MS² but not selected as the MS³ precursor 
cannot directly produce MS³ fragments, even if the formulas suggest a valid subtraction relationship.

Why merge fragments across MS levels: A fragment with formula CH₄ appearing in both MS² and MS³ is 
the same chemical entity. Its appearance in both levels provides information about the fragmentation 
cascade that helps deduce true parent-child relationships.
"""

from enum import unique
import polars as pl
import numpy as np
from numpy.typing import NDArray
import itertools
from typing import Literal, Dict
from hrms_utils.hrms_core import NUM_ELEMENTS 

def construct_fragmentation_graphs_from_msp_data(
    msp_frame: pl.LazyFrame,
    water_absorption: bool = False,
) :
    
    # Input validation
    required_columns = [
        "base_inchikey", 
        "ion_mode", 
        "precursor_formula_array",
        "mslevel",
        "cleaned_fragment_formulas",
    ]
    missing = set(required_columns) - set(msp_frame.columns)
    assert not missing, (
        f"Missing required columns for fragmentation graph construction: {missing}. "
        f"Expected columns: {required_columns}"
    )
    
    partialy_merged: pl.LazyFrame = msp_frame.group_by(["base_inchikey", "ion_mode", "precursor_formula_array"]).agg(
        pl.concat_list("cleaned_fragment_formulas").list.unique().alias("cleaned_fragment_formulas"),
    ).with_row_index(name="index_partially_merged")
    
    partialy_merged_exploded = partialy_merged.select("index_partially_merged", "cleaned_fragment_formulas").explode("cleaned_fragment_formulas")
    
    water_vector = np.zeros(NUM_ELEMENTS, dtype=np.int32)
    if water_absorption:
        water_vector[0] = 2 # H
        water_vector[3] = 1 # O
    strong_pairs= partialy_merged_exploded.join(
        partialy_merged_exploded,
        left_on=["index_partially_merged"],
        right_on=["index_partially_merged"],
        suffix="_child",
    ).filter(
        (pl.col("cleaned_fragment_formulas") - pl.col("cleaned_fragment_formulas_child")+ pl.lit(value=water_vector,dtype=pl.Array(inner=pl.Int32,shape=(NUM_ELEMENTS,)))).arr.min().ge(0)
    )
    fully_merged= msp_frame.group_by(["base_inchikey", "ion_mode"]).agg(
        pl.concat_list("cleaned_fragment_formulas").list.unique().alias("all_cleaned_fragment_formulas"),
    ).with_row_index(name="index_fully_merged")
    
    fully_merged_exploded = fully_merged.select("index_fully_merged", "all_cleaned_fragment_formulas").explode("all_cleaned_fragment_formulas")
    weak_pairs = fully_merged_exploded.join(
        fully_merged_exploded,
        left_on=["index_fully_merged"],
        right_on=["index_fully_merged"],
        suffix="_child",
    ).filter(
        (pl.col("all_cleaned_fragment_formulas") - pl.col("all_cleaned_fragment_formulas_child")+ pl.lit(value=water_vector,dtype=pl.Array(inner=pl.Int32,shape=(NUM_ELEMENTS,)))).arr.min().ge(0)
    )
    
    # now, if some child fragments has any strong parent, we remove all weak parents that are not strong parents for that child,
    # but if it lack any strogn parent, we keep all weak parents
    ##### not tested yet ####
    # Why: Identify children that have at least one strong parent (within same precursor group)
    children_with_strong_parents = strong_pairs.select(
        pl.col("cleaned_fragment_formulas_child")
    ).unique()
    
    # Why: For children with strong parents, keep only weak pairs that are also strong pairs
    weak_pairs_with_strong_children = weak_pairs.join(
        children_with_strong_parents,
        left_on="all_cleaned_fragment_formulas_child",
        right_on="cleaned_fragment_formulas_child",
        how="semi",  # Keep only weak pairs where child has a strong parent
    )
    
    weak_pairs_to_keep_from_strong_children = weak_pairs_with_strong_children.join(
        strong_pairs.select(
            pl.col("cleaned_fragment_formulas").alias("all_cleaned_fragment_formulas"),
            pl.col("cleaned_fragment_formulas_child").alias("all_cleaned_fragment_formulas_child"),
        ),
        on=["all_cleaned_fragment_formulas", "all_cleaned_fragment_formulas_child"],
        how="semi",  # Keep only if this weak pair is also a strong pair
    )
    
    # Why: For children without any strong parents, keep all weak parent relationships
    weak_pairs_to_keep_from_orphan_children = weak_pairs.join(
        children_with_strong_parents,
        left_on="all_cleaned_fragment_formulas_child",
        right_on="cleaned_fragment_formulas_child",
        how="anti",  # Keep only weak pairs where child has no strong parent
    )
    
    # Why: Combine filtered weak pairs - either strong-backed or from orphan children
    filtered_weak_pairs = pl.concat(
        [weak_pairs_to_keep_from_strong_children, weak_pairs_to_keep_from_orphan_children]
    )
