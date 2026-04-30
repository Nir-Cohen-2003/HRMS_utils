import pytest
import polars as pl
import numpy as np
from typing import List
import hrms_utils

# Element indices
IDX_C = 1
IDX_S = 7
IDX_Cl = 8
IDX_Br = 10

NUM_ELEMENTS = 12

# Isotopic mass differences (approximate, matching Rust implementation)
DIFF_C = 1.003355
DIFF_S = 1.995796
DIFF_Cl = 1.99705
DIFF_Br = 1.99795

# Isotopic probabilities (matching Rust implementation)
PROB_0_C = 0.9893
PROB_1_C = 0.0107

PROB_0_Cl = 0.7578
PROB_1_Cl = 0.2422

PROB_0_Br = 0.5069
PROB_1_Br = 0.4931

PROB_0_S = 0.9493
PROB_1_S = 0.0429


def test_deduce_isotopic_pattern_basic():
    """
    Test basic isotopic pattern deduction with a synthetic spectrum containing
    a precursor and a Chlorine isotope.
    """
    precursor_mz = 200.0
    precursor_intensity = 1.0e6
    
    # Simulate 1 Chlorine atom
    # Ratio = (1 * 0.2422) / (0.7578) ~= 0.3196
    expected_cl_intensity = precursor_intensity * (PROB_1_Cl * 1 / PROB_0_Cl)
    
    ms1_mzs = [precursor_mz, precursor_mz + DIFF_Cl]
    ms1_intensities = [precursor_intensity, expected_cl_intensity]
    
    df = pl.DataFrame({
        "precursor_mz": [precursor_mz],
        "ms1_mzs": [ms1_mzs],
        "ms1_intensities": [ms1_intensities]
    })
    
    result = df.with_columns(
        pl.col("precursor_mz").mass_decomposition.deduce_isotopic_pattern(
            ms1_mzs=pl.col("ms1_mzs"),
            ms1_intensities=pl.col("ms1_intensities"),
            ms1_mass_tolerance_ppm=5.0,
            isotopic_mass_tolerance_ppm=5.0,
            minimum_intensity=1.0e4
        ).alias("bounds")
    )
    
    bounds = result["bounds"][0].to_list()
    
    # Check Cl bounds (index 8 for min, 8+12=20 for max)
    cl_min = bounds[IDX_Cl]
    cl_max = bounds[IDX_Cl + NUM_ELEMENTS]
    
    # We expect strictly 1 Cl because we matched the intensity exactly
    assert cl_min == 1
    assert cl_max == 1

def test_deduce_isotopic_pattern_no_isotopes():
    """
    Test with only precursor peak. Should infer 0 for Cl, Br, S if intensity is high enough.
    """
    precursor_mz = 300.0
    precursor_intensity = 1.0e7 # High intensity
    
    ms1_mzs = [precursor_mz]
    ms1_intensities = [precursor_intensity]
    
    df = pl.DataFrame({
        "precursor_mz": [precursor_mz],
        "ms1_mzs": [ms1_mzs],
        "ms1_intensities": [ms1_intensities]
    })
    
    result = df.with_columns(
        pl.col("precursor_mz").mass_decomposition.deduce_isotopic_pattern(
            ms1_mzs=pl.col("ms1_mzs"),
            ms1_intensities=pl.col("ms1_intensities"),
            minimum_intensity=1.0e4
        ).alias("bounds")
    )
    
    bounds = result["bounds"][0].to_list()
    
    # Should be 0 for Cl, Br, S because we expect to see them if they were there
    assert bounds[IDX_Cl] == 0
    assert bounds[IDX_Cl + NUM_ELEMENTS] == 0
    assert bounds[IDX_Br] == 0
    assert bounds[IDX_Br + NUM_ELEMENTS] == 0
    assert bounds[IDX_S] == 0
    assert bounds[IDX_S + NUM_ELEMENTS] == 0
    
    # For Carbon, it's tricky. If we don't see C13, max C is limited.
    # Max C ~ (min_intensity * P0) / (P1 * precursor_int)
    # 1e4 * 0.99 / (0.01 * 1e7) ~ 1e4 / 1e5 ~ 0.1 -> Max C should be 0
    assert bounds[IDX_C + NUM_ELEMENTS] == 0

def test_deduce_isotopic_pattern_bromine():
    """
    Test Bromine detection.
    """
    precursor_mz = 400.0
    precursor_intensity = 5.0e6
    
    # Simulate 1 Bromine
    # Ratio ~ 1 * 0.49 / 0.51 ~ 0.96
    br_intensity = precursor_intensity * (PROB_1_Br / PROB_0_Br)
    
    ms1_mzs = [precursor_mz, precursor_mz + DIFF_Br]
    ms1_intensities = [precursor_intensity, br_intensity]
    
    df = pl.DataFrame({
        "precursor_mz": [precursor_mz],
        "ms1_mzs": [ms1_mzs],
        "ms1_intensities": [ms1_intensities]
    })
    
    result = df.with_columns(
        pl.col("precursor_mz").mass_decomposition.deduce_isotopic_pattern(
            ms1_mzs=pl.col("ms1_mzs"),
            ms1_intensities=pl.col("ms1_intensities"),
        ).alias("bounds")
    )
    
    bounds = result["bounds"][0].to_list()
    
    assert bounds[IDX_Br] == 1
    assert bounds[IDX_Br + NUM_ELEMENTS] == 1

def test_deduce_isotopic_pattern_bounds_passthrough():
    """
    Test that provided min/max bounds are passed through for non-isotopic elements.
    """
    precursor_mz = 150.0
    ms1_mzs = [150.0]
    ms1_intensities = [1e6]
    
    # Set specific bounds for N (idx 2) and O (idx 3)
    min_bounds_in = {"N": 2, "O": 1}
    max_bounds_in = {"N": 5, "O": 10}
    
    df = pl.DataFrame({
        "precursor_mz": [precursor_mz],
        "ms1_mzs": [ms1_mzs],
        "ms1_intensities": [ms1_intensities]
    })
    
    result = df.with_columns(
        pl.col("precursor_mz").mass_decomposition.deduce_isotopic_pattern(
            ms1_mzs=pl.col("ms1_mzs"),
            ms1_intensities=pl.col("ms1_intensities"),
            min_bounds=min_bounds_in,
            max_bounds=max_bounds_in
        ).alias("bounds")
    )
    
    bounds = result["bounds"][0].to_list()
    
    # Check N
    assert bounds[2] == 2
    assert bounds[2 + NUM_ELEMENTS] == 5
    
    # Check O
    assert bounds[3] == 1
    assert bounds[3 + NUM_ELEMENTS] == 10



