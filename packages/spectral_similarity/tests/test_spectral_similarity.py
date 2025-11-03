import polars as pl
import numpy as np

# This is the correct way to import the namespace
import spectral_similarity


def test_entropy_similarity():
    df = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 300.0],
                    "intensities1": [0.5, 0.8, 1.0],
                    "mz2": [100.0, 200.0, 300.0],
                    "intensities2": [0.5, 0.8, 1.0],
                }
            ]
        },
        schema={
            "spectra": pl.Struct(
                {
                    "mz1": pl.List(pl.Float64),
                    "intensities1": pl.List(pl.Float64),
                    "mz2": pl.List(pl.Float64),
                    "intensities2": pl.List(pl.Float64),
                }
            )
        },
    )

    result = df.with_columns(
        similarity=pl.col("spectra").spectral.entropy_similarity(
            ms2_tolerance_in_ppm=10.0,
        )
    )

    # Why: identical spectra should have similarity of 1.0 within numerical precision
    result_filtered = result.filter((pl.col("similarity") - 1.0).abs() < 1e-6)
    assert (
        len(result_filtered) == len(result)
    ), f"Expected all rows to have similarity ≈ 1.0, but got {result['similarity'].to_list()}"


def test_entropy_similarity_precursor_filtering():
    # Test precursor filtering: peaks above and near precursor should be removed
    df = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 299.5, 300.0],  # peak at 299.5 and 300.0 should be removed
                    "intensities1": [1.0, 1.0, 1.0, 1.0],
                    "mz2": [100.0, 200.0],
                    "intensities2": [1.0, 1.0],
                }
            ]
        },
        schema={
            "spectra": pl.Struct(
                {
                    "mz1": pl.List(pl.Float64),
                    "intensities1": pl.List(pl.Float64),
                    "mz2": pl.List(pl.Float64),
                    "intensities2": pl.List(pl.Float64),
                }
            )
        },
    )

    # Why: with precursor_mz=300, peaks >= 299 should be removed by default (1 Da tolerance)
    # After filtering, mz1 becomes [100.0, 200.0], matching mz2, so similarity should be ~1.0
    result = df.with_columns(
        similarity=pl.col("spectra").spectral.entropy_similarity(precursor_mz=300.0)
    )

    result_filtered = result.filter(pl.col("similarity") > 0.99)
    assert (
        len(result_filtered) == len(result)
    ), f"Expected similarity > 0.99 after precursor filtering, got {result['similarity'].to_list()}"

    # Test ignore_precursor: precursor peak should be removed
    df_ignore = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 300.001],  # precursor peak within tolerance
                    "intensities1": [1.0, 1.0, 1.0],
                    "mz2": [100.0, 200.0],
                    "intensities2": [1.0, 1.0],
                }
            ]
        },
        schema={
            "spectra": pl.Struct(
                {
                    "mz1": pl.List(pl.Float64),
                    "intensities1": pl.List(pl.Float64),
                    "mz2": pl.List(pl.Float64),
                    "intensities2": pl.List(pl.Float64),
                }
            )
        },
    )

    # Why: precursor peak at 300.001 is within 10ppm of 300.0, should be removed when ignore_precursor=True
    result_ignore = df_ignore.with_columns(
        similarity=pl.col("spectra").spectral.entropy_similarity(
            precursor_mz=300.0, ignore_precursor=True, ms2_tolerance_in_ppm=10.0
        )
    )

    result_filtered = result_ignore.filter(pl.col("similarity") > 0.99)
    assert (
        len(result_filtered) == len(result_ignore)
    ), f"Expected similarity > 0.99 after precursor removal, got {result_ignore['similarity'].to_list()}"


def test_cosine_similarities():
    df = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 300.0],
                    "intensities1": [0.5, 0.8, 1.0],
                    "mz2": [100.0, 200.0, 400.0],
                    "intensities2": [0.5, 0.8, 1.0],
                }
            ]
        },
        schema={
            "spectra": pl.Struct(
                {
                    "mz1": pl.List(pl.Float64),
                    "intensities1": pl.List(pl.Float64),
                    "mz2": pl.List(pl.Float64),
                    "intensities2": pl.List(pl.Float64),
                }
            )
        },
    )

    # Why: compute expected cosine similarity manually
    # matching peaks for dot product: [(100, 0.5, 0.5), (200, 0.8, 0.8)]
    # but denominator uses ALL intensities from both spectra
    all_intensities1 = np.array([0.5, 0.8, 1.0])
    all_intensities2 = np.array([0.5, 0.8, 1.0])
    matching_intensities1 = np.array([0.5, 0.8])
    matching_intensities2 = np.array([0.5, 0.8])
    
    # Why: classic cosine with intensity_power=1.0, mass_power=0.0
    # numerator: sum of products of matching peaks
    # denominator: norm of all peaks in each spectrum
    expected_cosine = np.dot(matching_intensities1, matching_intensities2) / (
        np.linalg.norm(all_intensities1) * np.linalg.norm(all_intensities2)
    )

    # General cosine
    result_general = df.with_columns(
        similarity=pl.col("spectra").spectral.general_cosine_similarity(
            intensity_power=1.0, mass_power=0.0
        )
    )

    result_filtered = result_general.filter(
        (pl.col("similarity") - expected_cosine).abs() < 1e-6
    )
    assert len(result_filtered) == len(
        result_general
    ), f"Expected cosine similarity ≈ {expected_cosine}, got {result_general['similarity'].to_list()}"

    # Why: mass weighted cosine (mass_power=1.0, intensity_power=0.5)
    # # ALL peaks contribute to denominator, not just matching ones
    # all_weighted_intensities1 = (all_intensities1**0.5) * np.array([100.0, 200.0, 300.0])
    # all_weighted_intensities2 = (all_intensities2**0.5) * np.array([100.0, 200.0, 400.0])
    # matching_weighted_intensities1 = (matching_intensities1**0.5) * np.array([100.0, 200.0])
    # matching_weighted_intensities2 = (matching_intensities2**0.5) * np.array([100.0, 200.0])
    
    # expected_mw_cosine = np.dot(.venv)matching_weighted_intensities1, matching_weighted_intensities2) / (
    #     np.linalg.norm(all_weighted_intensities1) * np.linalg.norm(all_weighted_intensities2)
    # )

    # result_mw = df.with_columns(
    #     similarity=pl.col("spectra").spectral.mass_weighted_cosine_similarity()
    # )

    # result_filtered = result_mw.filter(
    #     (pl.col("similarity") - expected_mw_cosine).abs() < 1e-6
    # )
    # assert len(result_filtered) == len(
    #     result_mw
    # ), f"Expected mass-weighted cosine ≈ {expected_mw_cosine}, got {result_mw['similarity'].to_list()}"

    # Why: explained intensity: sum of min intensities / sum of spectrum2 intensities
    # matching peaks: min(0.5, 0.5) + min(0.8, 0.8) = 1.3
    # total intensity in spectrum2: 0.5 + 0.8 + 1.0 = 2.3
    
    df = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0],
                    "intensities1": [0.5, 0.8],
                    "mz2": [100.0, 200.0, 400.0],
                    "intensities2": [0.5, 0.8, 1.0],
                }
            ]
        },
        schema={
            "spectra": pl.Struct(
                {
                    "mz1": pl.List(pl.Float64),
                    "intensities1": pl.List(pl.Float64),
                    "mz2": pl.List(pl.Float64),
                    "intensities2": pl.List(pl.Float64),
                }
            )
        },
    )
    expected_ei = 1.3 / 2.3 
    
    result_ei = df.with_columns(
        similarity=pl.col("spectra").spectral.explained_intensity()
    )

    result_filtered = result_ei.filter((pl.col("similarity") - expected_ei).abs() < 1e-6)
    assert len(result_filtered) == len(
        result_ei
    ), f"Expected explained intensity ≈ {expected_ei}, got {result_ei['similarity'].to_list()}"

    # subset error
    df_ei_not_subset = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 500.0],
                    "intensities1": [0.5, 0.8, 1.0],
                    "mz2": [100.0, 200.0, 400.0],
                    "intensities2": [0.5, 0.8, 1.0]
                }
            ]
        },
        schema={
            "spectra": pl.Struct(
                {
                    "mz1": pl.List(pl.Float64),
                    "intensities1": pl.List(pl.Float64),
                    "mz2": pl.List(pl.Float64),
                    "intensities2": pl.List(pl.Float64)
                }
            )
        }
    )

    result_ei_not_subset = df_ei_not_subset.with_columns(
        similarity=pl.col("spectra").spectral.explained_intensity(intensity_power=1.0, mass_power=0.0)
    )

    assert result_ei_not_subset["similarity"][0] == -1.0, f"Expected explained intensity to be -1.0 when A is not a subset of B, got {result_ei_not_subset['similarity'][0]}"
    # now tha same for mass weighted
    result_ei_not_subset_mw = df_ei_not_subset.with_columns(
        similarity=pl.col("spectra").spectral.explained_intensity(intensity_power=0.5, mass_power=1.0)
    )

    # Why: mass-weighted explained intensity should also return -1.0 when A is not a subset of B
    assert result_ei_not_subset_mw["similarity"][0] == -1.0, (
        f"Expected mass-weighted explained intensity to be -1.0 when A is not a subset of B, "
        f"got {result_ei_not_subset_mw['similarity'][0]}"
    )


def test_explained_intensity_options():
    # Test with permissive=True, where spec A is not a subset of spec B
    df_permissive = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 500.0], # 500.0 is not in mz2
                    "intensities1": [0.5, 0.8, 1.0],
                    "mz2": [100.0, 200.0, 400.0],
                    "intensities2": [0.5, 0.8, 1.0]
                }
            ]
        },
        schema={
            "spectra": pl.Struct({
                "mz1": pl.List(pl.Float64), "intensities1": pl.List(pl.Float64),
                "mz2": pl.List(pl.Float64), "intensities2": pl.List(pl.Float64)
            })
        }
    )

    # With permissive=True, it should ignore the non-matching peak (500.0) and calculate EI
    # Matching peaks: 100.0, 200.0. Sum of intensities in spec A: 0.5 + 0.8 = 1.3
    # Total intensity in spec B: 0.5 + 0.8 + 1.0 = 2.3
    expected_ei_permissive = 1.3 / 2.3
    result_permissive = df_permissive.with_columns(
        similarity=pl.col("spectra").spectral.explained_intensity(permissive=True)
    )
    assert abs(result_permissive["similarity"][0] - expected_ei_permissive) < 1e-6, \
        f"Expected permissive explained intensity ≈ {expected_ei_permissive}, got {result_permissive['similarity'][0]}"

    # With permissive=False (default), it should return -1.0
    result_strict = df_permissive.with_columns(
        similarity=pl.col("spectra").spectral.explained_intensity(permissive=False)
    )
    assert result_strict["similarity"][0] == -1.0, \
        f"Expected strict explained intensity to be -1.0, got {result_strict['similarity'][0]}"

    # Test with ms2_tolerance_in_da
    df_da_tolerance = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.05], # 200.05 is within 0.1 Da of 200.0
                    "intensities1": [0.5, 0.8],
                    "mz2": [100.0, 200.0, 400.0],
                    "intensities2": [0.5, 0.8, 1.0]
                }
            ]
        },
        schema={
            "spectra": pl.Struct({
                "mz1": pl.List(pl.Float64), "intensities1": pl.List(pl.Float64),
                "mz2": pl.List(pl.Float64), "intensities2": pl.List(pl.Float64)
            })
        }
    )

    # With default ppm tolerance, peaks might not match
    # 200.05 vs 200.0 -> diff is 0.05. ppm diff is (0.05 / 200.0) * 1e6 = 250 ppm. Default is 5 ppm.
    result_ppm = df_da_tolerance.with_columns(
        similarity=pl.col("spectra").spectral.explained_intensity()
    )
    assert result_ppm["similarity"][0] == -1.0, \
        f"Expected explained intensity with default ppm tolerance to be -1.0, got {result_ppm['similarity'][0]}"
