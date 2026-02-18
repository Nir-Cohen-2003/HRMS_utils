import timeit
from pathlib import Path

import polars as pl

from hrms_utils.formats import (
    NUM_ELEMENTS,
    annotate_chromatogram_with_formulas,
    get_chromatogram,
)

if __name__ == "__main__":
    # Resolve the chromatogram file relative to this script to fail fast if missing.

    # chromatogram_path = Path(__file__).parent.parent / "data" / "MSDIAL_output.txt"
    chromatogram_path = Path(
        "/home/analytit_admin/Data/raw_data/iibr_data/methamphetamine_in_plasma/250120_04amph.txt"
    )
    if not chromatogram_path.exists():
        raise FileNotFoundError(
            f"Required chromatogram file not found: {chromatogram_path}"
        )

    chromatogram_df = get_chromatogram(str(chromatogram_path)).filter(
        pl.col("ms1_isotopes_m/z").is_not_null(),
        pl.col("msms_m/z").is_not_null(),
    )
    num_total_peaks: int = chromatogram_df.height
    chromatogram_df = chromatogram_df.filter(
        pl.col("Height") > 1e6,
        pl.col("Isotope").eq(0),  # only monoisotopic peaks
    )
    num_monoisotopic_peaks = chromatogram_df.height
    # Benchmark annotation: run the annotation 10 times and keep the fastest result.
    # Why: clone the input before each run to avoid in-place mutations changing subsequent timings.
    n_runs = 10
    timings: list[float] = []
    best_time: float | None = None
    best_annotated: pl.DataFrame | None = None

    for run_idx in range(n_runs):
        start = timeit.default_timer()
        result = annotate_chromatogram_with_formulas(
            chromatogram_df.clone(),  # clone to ensure each run sees the same input state
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
            isotopic_intensity_absolute_tolerance=2e5,
        )
        elapsed = timeit.default_timer() - start
        timings.append(elapsed)
        if best_time is None or elapsed < best_time:
            best_time = elapsed
            best_annotated = result

    # Fail fast if annotation never produced a result (should not happen)
    assert best_annotated is not None, (
        "Annotation did not produce any result across runs"
    )
    annotated_chromatogram = best_annotated

    print(f"Annotation timings (s): {timings}")
    print(f"Fastest annotation time over {n_runs} runs: {best_time:.4f} s")

    # Basic sanity on schema for the new cleaner outputs (single-formula, normalized)
    schema = annotated_chromatogram.schema
    assert "cleaned_msms_mz" in schema, (
        "Expected normalized masses column 'cleaned_msms_mz'"
    )
    assert "cleaned_msms_intensity" in schema, (
        "Expected 'cleaned_msms_intensity' column"
    )
    assert "cleaned_spectrum_formulas" in schema, (
        "Expected 'cleaned_spectrum_formulas' column"
    )
    assert "cleaned_fragment_errors_ppm" in schema, (
        "Expected 'cleaned_fragment_errors_ppm' column"
    )

    # Type checks (reduced nesting: List[Array(Int32, NUM_ELEMENTS)])
    expected_formulas_dtype = pl.List(pl.Array(pl.Int32, NUM_ELEMENTS))
    assert schema["cleaned_spectrum_formulas"] == expected_formulas_dtype, (
        f"cleaned_spectrum_formulas dtype mismatch: {schema['cleaned_spectrum_formulas']} "
        f"!= {expected_formulas_dtype}"
    )
    assert schema["cleaned_msms_mz"] == pl.List(pl.Float64), (
        "cleaned_msms_mz must be List(Float64)"
    )
    assert schema["cleaned_msms_intensity"] == pl.List(pl.Float64), (
        "cleaned_msms_intensity must be List(Float64)"
    )
    assert schema["cleaned_fragment_errors_ppm"] == pl.List(pl.Float64), (
        "cleaned_fragment_errors_ppm must be List(Float64)"
    )

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

    print(f"Initial number of peaks (with MS2): {num_total_peaks}")
    print(
        f"Number of monoisotopic peaks (Isotope==0) with height over the threshold: {num_monoisotopic_peaks}"
    )
    print(
        f"Number of annotated formulas: {
            annotated_chromatogram.filter(
                pl.col('precursor_formula').is_not_null()
            ).height
        }"
    )
    print(
        f"number of peaks with any annotation: {
            annotated_chromatogram.filter(pl.col('precursor_formula').is_not_null())
            .unique(subset=pl.col('Peak ID'))
            .height
        }"
    )

    show_columns = [
        "Peak ID",
        "Precursor_mz_MSDIAL",
        "Height",
        "precursor_formula",
        "precursor_formula_str",
        "cleaned_spectrum_formulas",
        "cleaned_spectrum_formulas_str",
        "cleaned_fragment_errors_ppm",
        "cleaned_msms_mz",
    ]
    # Show top-level columns including normalized masses and single-formula assignments
    print(annotated_chromatogram.select(show_columns))

    # Compact preview with errors
    print(
        annotated_chromatogram.slice(offset=1, length=5)
        .select(show_columns)
        .to_init_repr()
    )

    print(annotated_chromatogram.schema)
