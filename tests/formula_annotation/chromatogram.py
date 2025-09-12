from hrms_utils.interfaces import get_chromatogram
from hrms_utils.formula_annotation import annotate_chromatogram_with_formulas
from hrms_utils.formula_annotation.mass_decomposition import NUM_ELEMENTS
import polars as pl
from pathlib import Path

if __name__ == "__main__":
        
    # Resolve the chromatogram file relative to this script to fail fast if missing.
    chromatogram_path = Path(__file__).parent / "250515_006.txt"
    assert chromatogram_path.exists(), f"Required chromatogram file not found: {chromatogram_path}"

    chromatogram_df = get_chromatogram(str(chromatogram_path)).filter(
        pl.col("Height") > 2e6,
        pl.col("ms1_isotopes_m/z").is_not_null(),
        pl.col("msms_m/z").is_not_null(),
    )

    annotated_chromatogram = annotate_chromatogram_with_formulas(
        chromatogram_df,
        max_bounds={
            "C": 50,
            "H": 100,
            "O": 10,
            "N": 10,
            "S": 2,
            "P": 2,
        },
        precursor_mass_accuracy_ppm=3.0,
        fragment_mass_accuracy_ppm=5.0,
        isotopic_mass_accuracy_ppm=2.0,
        isotopic_intensity_relative_tolerance=0.05,
        isotopic_intensity_absolute_tolerance=1e6,
    )

    # Basic sanity on schema for the new cleaner outputs (single-formula, normalized)
    schema = annotated_chromatogram.schema
    assert "cleaned_msms_mz" in schema, "Expected normalized masses column 'cleaned_msms_mz'"
    assert "cleaned_msms_intensity" in schema, "Expected 'cleaned_msms_intensity' column"
    assert "cleaned_spectrum_formulas" in schema, "Expected 'cleaned_spectrum_formulas' column"
    assert "cleaned_fragment_errors_ppm" in schema, "Expected 'cleaned_fragment_errors_ppm' column"

    # Type checks (reduced nesting: List[Array(Int32, NUM_ELEMENTS)])
    expected_formulas_dtype = pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))
    assert schema["cleaned_spectrum_formulas"] == expected_formulas_dtype, (
        f"cleaned_spectrum_formulas dtype mismatch: {schema['cleaned_spectrum_formulas']} "
        f"!= {expected_formulas_dtype}"
    )
    assert schema["cleaned_msms_mz"] == pl.List(pl.Float64), "cleaned_msms_mz must be List(Float64)"
    assert schema["cleaned_msms_intensity"] == pl.List(pl.Float64), "cleaned_msms_intensity must be List(Float64)"
    assert schema["cleaned_fragment_errors_ppm"] == pl.List(pl.Float64), "cleaned_fragment_errors_ppm must be List(Float64)"

    # Cardinality consistency for at least one annotated row (if any)
    non_null = annotated_chromatogram.filter(
        pl.col("cleaned_spectrum_formulas").is_not_null()
        & (pl.col("cleaned_spectrum_formulas").list.len() > 0)
    )
    if non_null.height > 0:
        lengths = non_null.select(
            pl.col("cleaned_msms_mz").list.len().alias("n_masses"),
            pl.col("cleaned_msms_intensity").list.len().alias("n_ints"),
            pl.col("cleaned_spectrum_formulas").list.len().alias("n_forms"),
            pl.col("cleaned_fragment_errors_ppm").list.len().alias("n_errs"),
        ).row(0)
        n_masses, n_ints, n_forms, n_errs = lengths
        assert n_masses == n_ints == n_forms == n_errs, (
            f"List lengths mismatch: masses={n_masses}, intensities={n_ints}, "
            f"formulas={n_forms}, errors={n_errs}"
        )

    print(f"Number of annotated formulas: {annotated_chromatogram.filter(
        pl.col('decomposed_formulas').is_not_null()
    ).height}")
    print(f"number of peaks with any annotation: {annotated_chromatogram.filter(
        pl.col('decomposed_formulas').is_not_null()
    ).unique(subset=pl.col('Peak ID')).height}")

    # Show top-level columns including normalized masses and single-formula assignments
    print(annotated_chromatogram.select([
        "Peak ID", "Precursor_mz_MSDIAL", "decomposed_formulas",
        "cleaned_msms_mz", "cleaned_msms_intensity", "cleaned_spectrum_formulas"
    ]))

    # Compact preview with errors
    print(annotated_chromatogram.head(1).select([
        "Peak ID", "Precursor_mz_MSDIAL", "decomposed_formulas",
        "cleaned_spectrum_formulas", "cleaned_fragment_errors_ppm", "cleaned_msms_mz"
    ]).to_init_repr())

    print(annotated_chromatogram.schema)