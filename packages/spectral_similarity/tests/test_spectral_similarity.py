import polars as pl
import sys
import pkgutil

try:
    from spectral_similarity import calculate_similarity
except ImportError as e:
    print(f"Failed to import spectral_similarity: {e}")
    print("\nAvailable non-stdlib packages:")
    
    # Why: List all installed packages to help diagnose import issues
    # Exclude stdlib modules by checking if they have a __file__ attribute in site-packages
    available_packages = set()
    for importer, modname, ispkg in pkgutil.iter_modules():
        # Why: Filter out stdlib by checking if module path contains 'site-packages' or 'dist-packages'
        if hasattr(importer, 'path') and importer.path:
            if 'site-packages' in str(importer.path) or 'dist-packages' in str(importer.path):
                available_packages.add(modname)
    
    for pkg in sorted(available_packages):
        print(f"  - {pkg}")
    
    sys.exit(1)

def test_calculate_similarity():
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
        schema={"spectra": pl.Struct({"mz1": pl.List(pl.Float32), "intensities1": pl.List(pl.Float32), "mz2": pl.List(pl.Float32), "intensities2": pl.List(pl.Float32)})}
    )

    result = df.with_columns(
        similarity=pl.col("spectra").spectral.entropy_similarity(
            ms2_tolerance_in_ppm=10.0,
        )
    )

    assert "similarity" in result.columns, "similarity column should be present in result dataframe"
    assert result["similarity"][0] is not None, "similarity value should not be None"
    assert abs(result["similarity"][0] - 1.0) < 1e-6, f"Expected similarity of 1.0 for identical spectra, got {result['similarity'][0]}"

    # Test with different tolerance
    df2 = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 300.0],
                    "intensities1": [1.0, 1.0, 1.0],
                    "mz2": [100.0003, 200.0, 300.0], # diff 0.0003 Da
                    "intensities2": [1.0, 1.0, 1.0],
                }
            ]
        },
        schema={"spectra": pl.Struct({"mz1": pl.List(pl.Float32), "intensities1": pl.List(pl.Float32), "mz2": pl.List(pl.Float32), "intensities2": pl.List(pl.Float32)})}
    )

    # default tolerance is 5ppm. At 100Da, it is 5 * 1e-6 * 200 = 0.001 Da. 0.0003 < 0.001, so it should match.
    result2_default = df2.with_columns(similarity=calculate_similarity(pl.col("spectra")))
    assert abs(result2_default["similarity"][0] - 1.0) < 1e-3

    # 1ppm tolerance. At 100Da, it is 1 * 1e-6 * 200 = 0.0002 Da. 0.0003 > 0.0002, so it should not match fully.
    result2_1ppm = df2.with_columns(similarity=calculate_similarity(pl.col("spectra"), ms2_tolerance_in_ppm=1.0))
    assert result2_1ppm["similarity"][0] < 1.0

    # Test with noise_threshold
    df3 = pl.DataFrame(
        {
            "spectra": [
                {
                    "mz1": [100.0, 200.0, 300.0, 400.0],
                    "intensities1": [1.0, 1.0, 1.0, 0.0005], # max intensity is 1.0
                    "mz2": [100.0, 200.0, 300.0],
                    "intensities2": [1.0, 1.0, 1.0],
                }
            ]
        },
        schema={"spectra": pl.Struct({"mz1": pl.List(pl.Float32), "intensities1": pl.List(pl.Float32), "mz2": pl.List(pl.Float32), "intensities2": pl.List(pl.Float32)})}
    )

    # with default noise_threshold=0.001, the peak at 400.0 (intensity 0.0005) should be removed.
    # noise_level = 0.001 * 1.0 = 0.001. 0.0005 < 0.001.
    # So spec1 becomes identical to spec2. Similarity should be 1.0
    result3_default_noise = df3.with_columns(similarity=calculate_similarity(pl.col("spectra")))
    assert abs(result3_default_noise["similarity"][0] - 1.0) < 1e-3

    # with noise_threshold=0.0001, the peak at 400.0 should NOT be removed.
    # noise_level = 0.0001 * 1.0 = 0.0001. 0.0005 > 0.0001.
    # So spec1 is different from spec2. Similarity should be less than 1.0
    result3_low_noise = df3.with_columns(similarity=calculate_similarity(pl.col("spectra"), noise_threshold=0.0001))
    assert result3_low_noise["similarity"][0] < 1.0

if __name__ == "__main__":
    test_calculate_similarity()
    print("Test passed!")