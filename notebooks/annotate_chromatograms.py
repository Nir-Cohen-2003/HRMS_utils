import marimo

__generated_with = "0.23.4"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    from hrms_utils.formats import get_chromatogram, annotate_chromatogram_with_formulas

    return Path, annotate_chromatogram_with_formulas, get_chromatogram, pl


@app.cell
def _(Path):
    # =============================================================================
    # USER-EDITABLE PARAMETERS
    # =============================================================================

    # Input directories containing MSDIAL .mdpeak chromatogram exports
    DATA_DIR = Path("/home/analytit_admin/Data/raw_data/iibr_data/251224_spiked_plasma")
    POSITIVE_DIR = DATA_DIR / "positive"
    NEGATIVE_DIR = DATA_DIR / "negative"

    # Output directory — one Excel file per input .mdpeak will be written here
    OUTPUT_DIR = DATA_DIR / "annotated_output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Mass tolerances
    MS1_TOLERANCE_PPM = 3.0  # ppm for precursor mass decomposition and isotope matching
    MS2_TOLERANCE_PPM = 5.0  # ppm for raw fragment mass matching during spectrum cleaning
    NORMALIZED_MS2_TOLERANCE_PPM = 4.0  # ppm for normalized fragment mass checks

    # Isotopic pattern parameters
    ISOTOPIC_TOLERANCE_PPM = 2.0  # ppm when matching isotopic peaks to expected positions
    ISOTOPIC_MIN_INTENSITY = 1e4  # minimum absolute intensity to consider an isotope peak

    # Optional manual element upper bounds.
    # These cap the search space for mass decomposition and dramatically reduce
    # the number of candidate formulas. Set to None to rely solely on
    # isotopic-pattern deduction (may produce millions of candidates).
    MAX_BOUNDS = {
        "C": 0,
        "H": 100,
        "O": 20,
        "N": 10,
        "S": 0,
        "P": 2,
        "Cl": 0,
        "Br": 0,
        "F": 6,
        "Na": 0,
        "K": 0,
        "I": 0,
    }

    # Post-annotation quality filters (applied after formula candidates are generated)
    MAX_PRECURSOR_ERROR_PPM = 3.0  # keep candidates within this mass error
    MIN_EXPLAINED_FRAGMENTS = 1  # minimum cleaned fragments to keep a candidate
    MIN_EXPLAINED_INTENSITY = 0.9  # minimum explained intensity (0–1) to keep a candidate
    MAX_CANDIDATES_PER_PEAK = 100  # keep only the top-N candidates per Peak ID 

    # Pre-filters applied before annotation
    MIN_HEIGHT = 50_000.0  # minimum peak height; set to 0.0 to disable
    return (
        ISOTOPIC_MIN_INTENSITY,
        ISOTOPIC_TOLERANCE_PPM,
        MAX_BOUNDS,
        MAX_CANDIDATES_PER_PEAK,
        MAX_PRECURSOR_ERROR_PPM,
        MIN_EXPLAINED_FRAGMENTS,
        MIN_EXPLAINED_INTENSITY,
        MIN_HEIGHT,
        MS1_TOLERANCE_PPM,
        MS2_TOLERANCE_PPM,
        NEGATIVE_DIR,
        NORMALIZED_MS2_TOLERANCE_PPM,
        OUTPUT_DIR,
        POSITIVE_DIR,
    )


@app.cell
def _(NEGATIVE_DIR, POSITIVE_DIR, Path, pl):
    # =============================================================================
    # DISCOVER INPUT FILES
    # =============================================================================

    def discover_files(positive_dir: Path, negative_dir: Path) -> pl.DataFrame:
        """Return a DataFrame of all .mdpeak files with their mode and path."""
        rows = []
        for mode, directory in [("positive", positive_dir), ("negative", negative_dir)]:
            if directory.exists():
                for path in sorted(directory.glob("*.mdpeak")):
                    rows.append({
                        "sample_id": path.stem,
                        "mode": mode,
                        "path": str(path),
                    })
        return pl.DataFrame(rows)

    files_df = discover_files(POSITIVE_DIR, NEGATIVE_DIR)
    files_df
    return (files_df,)


@app.cell
def _(
    ISOTOPIC_MIN_INTENSITY,
    ISOTOPIC_TOLERANCE_PPM,
    MAX_BOUNDS,
    MAX_CANDIDATES_PER_PEAK,
    MAX_PRECURSOR_ERROR_PPM,
    MIN_EXPLAINED_FRAGMENTS,
    MIN_EXPLAINED_INTENSITY,
    MIN_HEIGHT,
    MS1_TOLERANCE_PPM,
    MS2_TOLERANCE_PPM,
    NORMALIZED_MS2_TOLERANCE_PPM,
    Path,
    annotate_chromatogram_with_formulas,
    get_chromatogram,
    pl,
):
    # =============================================================================
    # PROCESSING FUNCTIONS
    # =============================================================================

    def process_mdpeak(path: str, mode: str) -> pl.DataFrame:
        """Read a single .mdpeak, annotate formulas, and add computed metrics."""
        chrom = get_chromatogram(path)
        chrom = chrom.filter(
            pl.col("Precursor_mz_MSDIAL").le(900.0),
            pl.col("RT (min)").ge(1.0),
            pl.col("Height").ge(MIN_HEIGHT),
            pl.col("ms1_isotopes_m/z").is_not_null(),
            pl.col("msms_m/z").is_not_null()
        )

        # Add metadata
        sample_id = Path(path).stem
        chrom = chrom.with_columns(
            pl.lit(sample_id).alias("sample_id"),
            pl.lit(mode).alias("mode"),
        )

        # Annotate with candidate formulas using isotopic pattern + mass decomposition
        annotated = annotate_chromatogram_with_formulas(
            chrom,
            max_bounds=MAX_BOUNDS,
            precursor_mass_accuracy_ppm=MS1_TOLERANCE_PPM,
            fragment_mass_accuracy_ppm=MS2_TOLERANCE_PPM,
            normalized_fragment_mass_accuracy_ppm=NORMALIZED_MS2_TOLERANCE_PPM,
            isotopic_mass_accuracy_ppm=ISOTOPIC_TOLERANCE_PPM,
            isotopic_minimum_intensity=ISOTOPIC_MIN_INTENSITY,
        )

        # Add computed metrics
        annotated = annotated.with_columns(
            pl.col("msms_m/z").list.len().alias("num_msms_fragments"),
            pl.col("cleaned_msms_mz").list.len().alias("num_explained_fragments"),
            (
                pl.col("cleaned_msms_intensity").list.sum()
                .truediv(pl.col("msms_intensity").list.sum())
            ).alias("explained_intensity"),
        )

        # Post-annotation quality filters
        annotated = annotated.filter(
            pl.col("precursor_errors_ppm").abs().le(MAX_PRECURSOR_ERROR_PPM),
            pl.col("num_explained_fragments").ge(MIN_EXPLAINED_FRAGMENTS),
            pl.col("explained_intensity").ge(MIN_EXPLAINED_INTENSITY),
        )

        # Rank candidates per peak and keep top N to stay within Excel row limits
        if MAX_CANDIDATES_PER_PEAK is not None and MAX_CANDIDATES_PER_PEAK > 0:
            annotated = (
                annotated.with_columns(
                    (
                        pl.col("num_explained_fragments") * 10
                        - pl.col("precursor_errors_ppm").abs()
                    ).alias("_candidate_score")
                )
                .sort(["Peak ID", "_candidate_score"], descending=[False, True])
                .group_by("Peak ID", maintain_order=True)
                .head(MAX_CANDIDATES_PER_PEAK)
                .drop("_candidate_score")
            )

        # Add simple integer index over the final annotated frame
        annotated = annotated.with_columns(
            pl.int_range(1, pl.len() + 1).alias("annotation_id")
        )

        # Reorder so annotation_id, sample_id, mode come first
        cols = list(annotated.columns)
        new_order = ["annotation_id", "sample_id", "mode"] + [
            c for c in cols if c not in ("annotation_id", "sample_id", "mode")
        ]
        return annotated.select(new_order)

    def prepare_for_excel(df: pl.DataFrame) -> pl.DataFrame:
        """Convert nested Polars types to comma-delimited strings for Excel export."""
        # Drop raw element-count arrays in favour of human-readable strings
        df = df.drop("precursor_formula", "cleaned_spectrum_formulas", strict=False)

        conversions = [
            pl.col("msms_m/z").cast(pl.List(pl.Utf8)).list.join(",").alias("msms_m/z"),
            pl.col("msms_intensity").cast(pl.List(pl.Utf8)).list.join(",").alias("msms_intensity"),
            pl.col("msms_m/z_cleaned").cast(pl.List(pl.Utf8)).list.join(",").alias("msms_m/z_cleaned"),
            pl.col("msms_intensity_cleaned").cast(pl.List(pl.Utf8)).list.join(",").alias("msms_intensity_cleaned"),
            pl.col("ms1_isotopes_m/z").cast(pl.List(pl.Utf8)).list.join(",").alias("ms1_isotopes_m/z"),
            pl.col("ms1_isotopes_intensity").cast(pl.List(pl.Utf8)).list.join(",").alias("ms1_isotopes_intensity"),
            pl.col("cleaned_msms_mz").cast(pl.List(pl.Utf8)).list.join(",").alias("cleaned_msms_mz"),
            pl.col("cleaned_msms_intensity").cast(pl.List(pl.Utf8)).list.join(",").alias("cleaned_msms_intensity"),
            pl.col("cleaned_spectrum_formulas_str").cast(pl.List(pl.Utf8)).list.join(",").alias("cleaned_spectrum_formulas_str"),
            pl.col("cleaned_fragment_errors_ppm").cast(pl.List(pl.Utf8)).list.join(",").alias("cleaned_fragment_errors_ppm"),
            pl.col("isobars").cast(pl.List(pl.Utf8)).list.join(",").alias("isobars"),
            pl.col("min_bounds").cast(pl.List(pl.Int32)).cast(pl.List(pl.Utf8)).list.join(",").alias("min_bounds"),
            pl.col("max_bounds").cast(pl.List(pl.Int32)).cast(pl.List(pl.Utf8)).list.join(",").alias("max_bounds"),
        ]
        return df.with_columns(conversions)

    return prepare_for_excel, process_mdpeak


@app.cell
def _(OUTPUT_DIR, files_df, pl, prepare_for_excel, process_mdpeak):
    # =============================================================================
    # BATCH PROCESSING — one Excel file per .mdpeak input
    # =============================================================================

    results = []
    for row in files_df.iter_rows(named=True):
        path = row["path"]
        mode = row["mode"]
        sample_id = row["sample_id"]

        print(f"Processing {sample_id} ({mode}) ...")
        annotated = process_mdpeak(path, mode)

        output_path = OUTPUT_DIR / f"{sample_id}_annotated.xlsx"
        excel_ready = prepare_for_excel(annotated)
        excel_ready.select(
            "annotation_id",
            "Peak ID",
            "RT (min)",	
            "Precursor_mz_MSDIAL",	
            "Height",	
            "precursor_formula_str",
            "Precursor_type_MSDIAL",
            "min_bounds",
            "max_bounds",
            "explained_intensity",
            "num_msms_fragments",
            "num_explained_fragments",
            "ms1_isotopes_m/z",
            "ms1_isotopes_intensity",
            "msms_m/z",
            "msms_intensity",
            "cleaned_spectrum_formulas_str",
            "cleaned_msms_mz",
            "cleaned_msms_intensity",
        	"cleaned_fragment_errors_ppm"			
        ).write_excel(output_path)

        results.append({
            "sample_id": sample_id,
            "mode": mode,
            "input_peaks": annotated.select(pl.col("Peak ID").n_unique()).item(),
            "output_candidates": annotated.height,
            "output_path": str(output_path),
        })
        print(f"  -> Wrote {annotated.height} candidates to {output_path}")

    summary_df = pl.DataFrame(results)
    summary_df
    return (summary_df,)


@app.cell
def _(summary_df):
    # =============================================================================
    # SUMMARY
    # =============================================================================
    summary_df
    return


if __name__ == "__main__":
    app.run()
