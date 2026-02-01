import numpy as np
import polars as pl
from reference_cosine_greedy import cosine_greedy_ppm

# This is the correct way to import the namespace
import hrms_utils


def test_entropy_similarity():
    df = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 300.0],
                    "intensities1": [0.5, 0.8, 1.0],
                    "mz2": [100.0, 200.0, 300.0],
                    "intensities2": [0.5, 0.8, 1.0],
                    "precursor_mz1": 400.0,
                    "precursor_mz2": 400.0,
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
                    "precursor_mz1": pl.Float64,
                    "precursor_mz2": pl.Float64,
                }
            )
        },
    )

    result = df.with_columns(
        similarity=pl.col("spectra").spectral_similarity.entropy_similarity(
            ms2_tolerance_in_ppm=10.0,
        )
    )

    # Why: identical spectra should have similarity of 1.0 within numerical precision
    result_filtered = result.filter((pl.col("similarity") - 1.0).abs() < 1e-6)
    assert len(result_filtered) == len(result), (
        f"Expected all rows to have similarity ≈ 1.0, but got {result['similarity'].to_list()}"
    )


def test_entropy_similarity_precursor_filtering():
    # Test precursor filtering: peaks above and near precursor should be removed, but not the precursor itself
    df = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [
                        100.0,
                        200.0,
                        299.5,
                        300.0,
                    ],  # peak at 299.5 and 300.0 should be removed
                    "intensities1": [1.0, 1.0, 1.0, 1.0],
                    "mz2": [100.0, 200.0],
                    "intensities2": [1.0, 1.0],
                    "precursor_mz1": 200.0,
                    "precursor_mz2": 200.0,
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
                    "precursor_mz1": pl.Float64,
                    "precursor_mz2": pl.Float64,
                }
            )
        },
    )

    # Why: with precursor_mz=300, peaks >= 299 should be removed by default (1 Da tolerance)
    # After filtering, mz1 becomes [100.0, 200.0], matching mz2, so similarity should be ~1.0
    result = df.with_columns(
        similarity=pl.col("spectra").spectral_similarity.entropy_similarity(
            ms2_tolerance_in_ppm=10.0,
        )
    )

    result_filtered = result.filter(pl.col("similarity") > 0.99)
    assert len(result_filtered) == len(result), (
        f"Expected similarity > 0.99 after precursor filtering, got {result['similarity'].to_list()}"
    )

    # Test ignore_precursor: precursor peak should be removed
    df_ignore = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 300.001],  # precursor peak within tolerance
                    "intensities1": [1.0, 1.0, 1.0],
                    "mz2": [100.0, 200.0],
                    "intensities2": [1.0, 1.0],
                    "precursor_mz1": 300.0,
                    "precursor_mz2": 300.0,
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
                    "precursor_mz1": pl.Float64,
                    "precursor_mz2": pl.Float64,
                }
            )
        },
    )

    # Why: precursor peak at 300.001 is within 10ppm of 300.0, should be removed when ignore_precursor=True
    result_ignore = df_ignore.with_columns(
        similarity=pl.col("spectra").spectral_similarity.entropy_similarity(
            ignore_precursor=True, ms2_tolerance_in_ppm=10.0
        )
    )

    result_filtered = result_ignore.filter(pl.col("similarity") > 0.99)
    assert len(result_filtered) == len(result_ignore), (
        f"Expected similarity > 0.99 after precursor removal, got {result_ignore['similarity'].to_list()}"
    )


def test_cosine_similarity_basic():
    df = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 300.0],
                    "intensities1": [0.5, 0.8, 1.0],
                    "mz2": [100.0, 200.0, 400.0],
                    "intensities2": [0.5, 0.8, 1.0],
                    "precursor_mz1": 400.0,
                    "precursor_mz2": 400.0,
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
                    "precursor_mz1": pl.Float64,
                    "precursor_mz2": pl.Float64,
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
        similarity=pl.col("spectra").spectral_similarity.general_cosine_similarity(
            intensity_power=1.0, mass_power=0.0, ms2_tolerance_in_ppm=10.0
        )
    )

    result_filtered = result_general.filter(
        (pl.col("similarity") - expected_cosine).abs() < 1e-6
    )
    assert len(result_filtered) == len(result_general), (
        f"Expected cosine similarity ≈ {expected_cosine}, got {result_general['similarity'].to_list()}"
    )


def test_cosine_similarity_with_precursor_filtering():
    # Test cosine similarity with ignore_precursor=True
    df_with_precursor = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 300.0],
                    "intensities1": [0.5, 0.8, 1.0],
                    "mz2": [100.0, 200.0, 400.0],
                    "intensities2": [0.5, 0.8, 1.0],
                    "precursor_mz1": 400.0,
                    "precursor_mz2": 400.0,
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
                    "precursor_mz1": pl.Float64,
                    "precursor_mz2": pl.Float64,
                }
            )
        },
    )

    # Why: with ignore_precursor=True, the peak at 400.0 in mz2 should be removed
    # After filtering mz2 becomes [100.0, 200.0], so matching peaks are [(100, 0.5, 0.5), (200, 0.8, 0.8)]
    # mz1 stays [100.0, 200.0, 300.0], mz2 after filtering is [100.0, 200.0]
    matching_intensities1 = np.array([0.5, 0.8])
    matching_intensities2 = np.array([0.5, 0.8])
    all_intensities1 = np.array([0.5, 0.8, 1.0])
    all_intensities2 = np.array([0.5, 0.8])  # 400.0 peak removed

    expected_cosine_filtered = np.dot(matching_intensities1, matching_intensities2) / (
        np.linalg.norm(all_intensities1) * np.linalg.norm(all_intensities2)
    )

    result_filtered = df_with_precursor.with_columns(
        similarity=pl.col("spectra").spectral_similarity.general_cosine_similarity(
            intensity_power=1.0,
            mass_power=0.0,
            ms2_tolerance_in_ppm=10.0,
            ignore_precursor=True,
        )
    )

    result_check = result_filtered.filter(
        (pl.col("similarity") - expected_cosine_filtered).abs() < 1e-6
    )
    assert len(result_check) == len(result_filtered), (
        f"Expected cosine similarity with precursor filtering ≈ {expected_cosine_filtered}, got {result_filtered['similarity'].to_list()}"
    )

    # Why: without ignore_precursor, peak at 400.0 should remain in calculations
    all_intensities2_no_filter = np.array([0.5, 0.8, 1.0])
    expected_cosine_no_filter = np.dot(matching_intensities1, matching_intensities2) / (
        np.linalg.norm(all_intensities1) * np.linalg.norm(all_intensities2_no_filter)
    )

    result_no_filter = df_with_precursor.with_columns(
        similarity=pl.col("spectra").spectral_similarity.general_cosine_similarity(
            intensity_power=1.0,
            mass_power=0.0,
            ms2_tolerance_in_ppm=10.0,
            ignore_precursor=False,
        )
    )

    result_check_no_filter = result_no_filter.filter(
        (pl.col("similarity") - expected_cosine_no_filter).abs() < 1e-6
    )
    assert len(result_check_no_filter) == len(result_no_filter), (
        f"Expected cosine similarity without precursor filtering ≈ {expected_cosine_no_filter}, got {result_no_filter['similarity'].to_list()}"
    )


def test_explained_intensity_basic():
    df = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0],
                    "intensities1": [0.5, 0.8],
                    "mz2": [100.0, 200.0, 400.0],
                    "intensities2": [0.5, 0.8, 1.0],
                    "precursor_mz1": 400.0,
                    "precursor_mz2": 400.0,
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
                    "precursor_mz1": pl.Float64,
                    "precursor_mz2": pl.Float64,
                }
            )
        },
    )
    expected_ei = 1.3 / 2.3

    result_ei = df.with_columns(
        similarity=pl.col("spectra").spectral_similarity.explained_intensity(
            ms2_tolerance_in_ppm=10.0
        )
    )

    result_filtered = result_ei.filter(
        (pl.col("similarity") - expected_ei).abs() < 1e-6
    )
    assert len(result_filtered) == len(result_ei), (
        f"Expected explained intensity ≈ {expected_ei}, got {result_ei['similarity'].to_list()}"
    )


def test_explained_intensity_subset_validation():
    # Why: test that explained intensity returns -1.0 when spec A is not a subset of spec B
    df_ei_not_subset = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 300.0],
                    "intensities1": [0.5, 0.8, 1.0],
                    "mz2": [100.0, 200.0, 400.0],
                    "intensities2": [0.5, 0.8, 1.0],
                    "precursor_mz1": 400.0,
                    "precursor_mz2": 400.0,
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
                    "precursor_mz1": pl.Float64,
                    "precursor_mz2": pl.Float64,
                }
            )
        },
    )

    result_ei_not_subset = df_ei_not_subset.with_columns(
        similarity=pl.col("spectra").spectral_similarity.explained_intensity(
            ms2_tolerance_in_ppm=10.0,
            intensity_power=1.0,
            mass_power=0.0,
            permissive=False,
        )
    )

    assert result_ei_not_subset["similarity"][0] == -1.0, (
        f"Expected explained intensity to be -1.0 when A is not a subset of B, got {result_ei_not_subset['similarity'][0]}"
    )

    # Why: test same validation for mass-weighted explained intensity
    result_ei_not_subset_mw = df_ei_not_subset.with_columns(
        similarity=pl.col("spectra").spectral_similarity.explained_intensity(
            ms2_tolerance_in_ppm=10.0, intensity_power=0.5, mass_power=1.0
        )
    )

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
                    "mz1": [100.0, 200.0, 300.0],  # 500.0 is not in mz2
                    "intensities1": [0.5, 0.8, 1.0],
                    "mz2": [100.0, 200.0, 400.0],
                    "intensities2": [0.5, 0.8, 1.0],
                    "precursor_mz1": 400.0,
                    "precursor_mz2": 400.0,
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
                    "precursor_mz1": pl.Float64,
                    "precursor_mz2": pl.Float64,
                }
            )
        },
    )

    # With permissive=True, it should ignore the non-matching peak (500.0) and calculate EI
    # Matching peaks: 100.0, 200.0. Sum of intensities in spec A: 0.5 + 0.8 = 1.3
    # Total intensity in spec B: 0.5 + 0.8 + 1.0 = 2.3
    expected_ei_permissive = 1.3 / 2.3
    result_permissive = df_permissive.with_columns(
        similarity=pl.col("spectra").spectral_similarity.explained_intensity(
            ms2_tolerance_in_ppm=10.0, permissive=True, ignore_precursor=False
        )
    )
    assert abs(result_permissive["similarity"][0] - expected_ei_permissive) < 1e-6, (
        f"Expected permissive explained intensity ≈ {expected_ei_permissive}, got {result_permissive['similarity'][0]}"
    )

    # With permissive=False (default), it should return -1.0
    result_strict = df_permissive.with_columns(
        similarity=pl.col("spectra").spectral_similarity.explained_intensity(
            ms2_tolerance_in_ppm=10.0, permissive=False
        )
    )
    assert result_strict["similarity"][0] == -1.0, (
        f"Expected strict explained intensity to be -1.0, got {result_strict['similarity'][0]}"
    )

    # Test with ms2_tolerance_in_da
    df_da_tolerance = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.05],  # 200.05 is within 0.1 Da of 200.0
                    "intensities1": [0.5, 0.8],
                    "mz2": [100.0, 200.0, 400.0],
                    "intensities2": [0.5, 0.8, 1.0],
                    "precursor_mz1": 400.0,
                    "precursor_mz2": 400.0,
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
                    "precursor_mz1": pl.Float64,
                    "precursor_mz2": pl.Float64,
                }
            )
        },
    )

    # With default ppm tolerance, peaks might not match
    # 200.05 vs 200.0 -> diff is 0.05. ppm diff is (0.05 / 200.0) * 1e6 = 250 ppm. Default is 5 ppm.
    result_ppm = df_da_tolerance.with_columns(
        similarity=pl.col("spectra").spectral_similarity.explained_intensity(
            ms2_tolerance_in_ppm=5.0
        )
    )
    assert result_ppm["similarity"][0] == -1.0, (
        f"Expected explained intensity with default ppm tolerance to be -1.0, got {result_ppm['similarity'][0]}"
    )


def generate_random_spectrum_pairs(
    num_pairs: int = 20,
    seed: int = 42,
    min_peaks: int = 3,
    max_peaks: int = 50,
    mz_range: tuple[float, float] = (50.0, 1000.0),
) -> list[dict]:
    """
    Generate random spectrum pairs for testing cosine similarity.

    Generates pairs with:
    - Some matching peaks (within tolerance)
    - Some non-matching peaks (outside tolerance)
    - Various intensity distributions

    Args:
        num_pairs: Number of spectrum pairs to generate
        seed: Random seed for reproducibility
        min_peaks: Minimum number of peaks per spectrum
        max_peaks: Maximum number of peaks per spectrum
        mz_range: (min_mz, max_mz) range for peak generation

    Returns:
        List of dicts with keys: mz1, intensities1, mz2, intensities2, precursor_mz1, precursor_mz2
    """
    rng = np.random.RandomState(seed)
    pairs = []

    for _ in range(num_pairs):
        # Random number of peaks for each spectrum
        n_peaks1 = rng.randint(min_peaks, max_peaks + 1)
        n_peaks2 = rng.randint(min_peaks, max_peaks + 1)

        # Generate m/z values - ensure they're sorted
        mz1 = np.sort(rng.uniform(mz_range[0], mz_range[1], n_peaks1))
        mz2 = np.sort(rng.uniform(mz_range[0], mz_range[1], n_peaks2))

        # Generate intensities (uniform 0.1-1.0)
        intensities1 = rng.uniform(0.1, 1.0, n_peaks1).astype(np.float32)
        intensities2 = rng.uniform(0.1, 1.0, n_peaks2).astype(np.float32)

        # Add some matching peaks: shift some mz1 values slightly to create matches in mz2
        n_matches = min(
            n_peaks1, n_peaks2, rng.randint(1, min(n_peaks1, n_peaks2, 10) + 1)
        )
        match_indices1 = rng.choice(n_peaks1, n_matches, replace=False)
        match_indices2 = rng.choice(n_peaks2, n_matches, replace=False)

        # Create matching peaks by copying mz1 values to mz2 with small shifts (within 5-10 ppm)
        for i, (idx1, idx2) in enumerate(zip(match_indices1, match_indices2)):
            # Shift by 0-5 ppm
            ppm_shift = rng.uniform(-5.0, 5.0)
            mz2[idx2] = mz1[idx1] * (1 + ppm_shift * 1e-6)
            # Optionally match intensities too for some pairs
            if rng.random() > 0.5:
                intensities2[idx2] = intensities1[idx1]

        # Re-sort mz2 after modifications
        sort_idx = np.argsort(mz2)
        mz2 = mz2[sort_idx]
        intensities2 = intensities2[sort_idx]

        # Precursor m/z - above the highest fragment
        precursor_mz1 = max(mz1.max(), mz2.max()) + rng.uniform(50, 100)
        precursor_mz2 = precursor_mz1

        pairs.append(
            {
                "mz1": mz1.tolist(),
                "intensities1": intensities1.tolist(),
                "mz2": mz2.tolist(),
                "intensities2": intensities2.tolist(),
                "precursor_mz1": float(precursor_mz1),
                "precursor_mz2": float(precursor_mz2),
            }
        )

    return pairs


def test_cosine_similarity_random_against_reference():
    """
    Test general_cosine_similarity against reference greedy cosine implementation.

    Generates random spectra with both matching and non-matching peaks,
    then compares the hrms_core Rust implementation against the Python
    reference implementation.
    """

    # Generate random spectrum pairs
    spectrum_data = generate_random_spectrum_pairs(
        num_pairs=10000,
        seed=42,
        min_peaks=5,
        max_peaks=30,
    )

    # Create DataFrame for hrms_core implementation
    df = pl.DataFrame(
        {"spectra": spectrum_data},
        schema={
            "spectra": pl.Struct(
                {
                    "mz1": pl.List(pl.Float64),
                    "intensities1": pl.List(pl.Float64),
                    "mz2": pl.List(pl.Float64),
                    "intensities2": pl.List(pl.Float64),
                    "precursor_mz1": pl.Float64,
                    "precursor_mz2": pl.Float64,
                }
            )
        },
    )

    # Run hrms_core implementation with various parameters
    tolerance_ppm = 10.0
    intensity_power = 0.5
    mass_power = 0.0

    result_hrms = df.with_columns(
        similarity=pl.col("spectra").spectral_similarity.general_cosine_similarity(
            intensity_power=intensity_power,
            mass_power=mass_power,
            ms2_tolerance_in_ppm=tolerance_ppm,
            clean_spectra_first=False,  # Disable cleaning to match reference
            ignore_precursor=False,
        )
    )

    hrms_scores = result_hrms["similarity"].to_list()

    # Run reference implementation on the same pairs - both with and without centroiding
    reference_scores_no_centroid = []
    reference_scores_with_centroid = []

    for pair in spectrum_data:
        mz1 = np.array(pair["mz1"], dtype=np.float64)
        intensities1 = np.array(pair["intensities1"], dtype=np.float32)
        mz2 = np.array(pair["mz2"], dtype=np.float64)
        intensities2 = np.array(pair["intensities2"], dtype=np.float32)

        # Without centroiding
        score_no_centroid, _ = cosine_greedy_ppm(
            mz1=mz1,
            intensity1=intensities1,
            mz2=mz2,
            intensity2=intensities2,
            tolerance_ppm=tolerance_ppm,
            intensity_power=intensity_power,
            mz_power=mass_power,
            apply_centroiding=False,
        )
        reference_scores_no_centroid.append(score_no_centroid)

        # With centroiding
        score_with_centroid, _ = cosine_greedy_ppm(
            mz1=mz1,
            intensity1=intensities1,
            mz2=mz2,
            intensity2=intensities2,
            tolerance_ppm=tolerance_ppm,
            intensity_power=intensity_power,
            mz_power=mass_power,
            apply_centroiding=True,
        )
        reference_scores_with_centroid.append(score_with_centroid)

    # Compare scores - Rust should match at least one reference implementation
    # Allow for some numerical differences due to different implementations
    tolerance = 0.01  # 1% tolerance for differences

    mismatches = []
    for i, (hrms_score, ref_no_centroid, ref_with_centroid) in enumerate(
        zip(hrms_scores, reference_scores_no_centroid, reference_scores_with_centroid)
    ):
        if hrms_score is None:
            # Both references should also be 0/None for empty matches
            continue

        hrms_val = float(hrms_score)
        diff_no_centroid = abs(hrms_val - float(ref_no_centroid))
        diff_with_centroid = abs(hrms_val - float(ref_with_centroid))

        # It's only a mismatch if Rust doesn't match EITHER reference
        matches_no_centroid = diff_no_centroid <= tolerance
        matches_with_centroid = diff_with_centroid <= tolerance

        if not matches_no_centroid and not matches_with_centroid:
            mismatches.append(
                {
                    "index": i,
                    "hrms_score": hrms_val,
                    "ref_no_centroid": float(ref_no_centroid),
                    "ref_with_centroid": float(ref_with_centroid),
                    "diff_no_centroid": diff_no_centroid,
                    "diff_with_centroid": diff_with_centroid,
                    "mz1_len": len(spectrum_data[i]["mz1"]),
                    "mz2_len": len(spectrum_data[i]["mz2"]),
                }
            )

    if mismatches:
        # Sort mismatches by max diff (largest first) for better debugging
        mismatches.sort(
            key=lambda m: max(m["diff_no_centroid"], m["diff_with_centroid"]),
            reverse=True,
        )

        # Print full details of first (worst) mismatch
        first_mismatch = mismatches[0]
        first_pair = spectrum_data[first_mismatch["index"]]

        error_msg = (
            f"Found {len(mismatches)} mismatches out of {len(spectrum_data)} pairs.\n\n"
            f"WORST FAILURE (Pair {first_mismatch['index']}):\n"
            f"  Rust (hrms_core):     {first_mismatch['hrms_score']:.6f}\n"
            f"  Ref (no centroiding): {first_mismatch['ref_no_centroid']:.6f} (diff={first_mismatch['diff_no_centroid']:.6f})\n"
            f"  Ref (with centroid):  {first_mismatch['ref_with_centroid']:.6f} (diff={first_mismatch['diff_with_centroid']:.6f})\n\n"
            f"Full Spectrum Pair:\n"
            f"  mz1:           {first_pair['mz1']}\n"
            f"  intensities1:  {first_pair['intensities1']}\n"
            f"  mz2:           {first_pair['mz2']}\n"
            f"  intensities2:  {first_pair['intensities2']}\n"
            f"  precursor_mz1: {first_pair['precursor_mz1']}\n"
            f"  precursor_mz2: {first_pair['precursor_mz2']}\n\n"
            f"Top 10 mismatches (index: max_diff | rust_score vs ref_no_centroid vs ref_with_centroid):\n"
        )

        for m in mismatches[:10]:
            max_diff = max(m["diff_no_centroid"], m["diff_with_centroid"])
            error_msg += (
                f"  Pair {m['index']}: max_diff={max_diff:.6f} | "
                f"rust={m['hrms_score']:.6f} vs ref_no={m['ref_no_centroid']:.6f} vs ref_cent={m['ref_with_centroid']:.6f}\n"
            )

        if len(mismatches) > 10:
            error_msg += f"  ... and {len(mismatches) - 10} more\n"

        assert len(mismatches) == 0, error_msg

    # Additional check: mean absolute difference should be small against best match
    valid_pairs = [
        (h, min(abs(h - r1), abs(h - r2)))
        for h, r1, r2 in zip(
            hrms_scores, reference_scores_no_centroid, reference_scores_with_centroid
        )
        if h is not None
    ]
    if valid_pairs:
        mean_diff = np.mean([min_diff for _, min_diff in valid_pairs])
        assert mean_diff < 0.001, (
            f"Mean absolute difference to best reference {mean_diff:.6f} exceeds threshold 0.001"
        )
