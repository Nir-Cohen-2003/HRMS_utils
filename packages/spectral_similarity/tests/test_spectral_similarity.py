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
        similarity=calculate_similarity(pl.col("spectra"))
    )

    assert "similarity" in result.columns, "similarity column should be present in result dataframe"
    assert result["similarity"][0] is not None, "similarity value should not be None"
    # Why: For identical spectra, cosine similarity should be 1.0
    assert abs(result["similarity"][0] - 1.0) < 1e-6, f"Expected similarity of 1.0 for identical spectra, got {result['similarity'][0]}"

if __name__ == "__main__":
    test_calculate_similarity()
    print("Test passed!")
