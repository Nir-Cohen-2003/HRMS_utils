import polars as pl
from spectral_similarity import calculate_similarity

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

    assert "similarity" in result.columns
    assert result["similarity"][0] is not None
    # for identical spectra, the similarity should be 1.0
    assert abs(result["similarity"][0] - 1.0) < 1e-6

if __name__ == "__main__":
    test_calculate_similarity()
    print("Test passed!")
