# HRMS_utils/experiments/spectral_information/recompute_and_save_library_info_scores.py
"""
Standalone script to recompute spectral information scores from original library data
and save a new full library parquet.

Why: Allows recomputing information scores (e.g. when the existing values are suspected
to be wrong) without rerunning expensive similarity or tanimoto calculations. The output
is a drop-in replacement for the original library file, preserving all columns and
overwriting only the score column(s).

Usage:
    pixi run -e experiments recompute_info_scores

Or directly:
    python experiments/spectral_information/recompute_and_save_library_info_scores.py
"""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

import polars as pl

# Add the src directory to sys.path so hrms_utils can be imported
sys.path.append(str(Path(__file__).parents[2] / "src"))

# Import hrms_core to register the spectral_info plugin
import hrms_utils.hrms_core  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def recompute_and_save_library(
    original_library_path: Path,
    output_library_path: Path,
) -> None:
    """
    Recompute spectral information scores from an original library parquet and write
    a full library parquet that is a drop-in replacement for the original.

    Both score variants are computed when the required formula columns are present:
      - spectral_information_score          (ignore_hydrogens=True)
      - spectral_information_score_with_hydrogens (ignore_hydrogens=False)

    Args:
        original_library_path: Path to the original library parquet with full spectral data.
            Must contain columns: precursor_formula_array, cleaned_fragment_formulas.
        output_library_path: Path where the new full library parquet will be written.
    """
    logger.info("Loading original library from %s", original_library_path)
    lf = pl.scan_parquet(str(original_library_path))

    schema_names = lf.collect_schema().names()

    required_cols = ["precursor_formula_array", "cleaned_fragment_formulas"]
    missing = [c for c in required_cols if c not in schema_names]
    if missing:
        raise AssertionError(
            f"Original library is missing required columns: {missing}. "
            f"Available columns: {schema_names}"
        )

    # Ensure idx and mol_idx exist for downstream compatibility.
    # If they already exist (e.g. from a prior snapshot), keep them.
    if "idx" not in schema_names:
        lf = lf.with_row_index("idx")
    if "mol_idx" not in schema_names:
        if "base_inchikey" in schema_names and "ion_mode" in schema_names:
            lf = lf.with_columns(
                mol_idx=pl.col("idx").min().over(["base_inchikey", "ion_mode"])
            )
        else:
            lf = lf.with_columns(mol_idx=pl.col("idx"))

    # Drop any existing score columns so we can recompute them cleanly
    schema_names = lf.collect_schema().names()
    cols_to_drop = [
        c
        for c in [
            "spectral_information_score",
            "spectral_information_score_with_hydrogens",
        ]
        if c in schema_names
    ]
    if cols_to_drop:
        lf = lf.drop(cols_to_drop)

    logger.info("Recomputing spectral_information_score (ignore_hydrogens=True)")
    t_start = perf_counter()

    lf = lf.with_columns(
        pl.struct(
            [
                pl.col("precursor_formula_array").alias("precursor_formula"),
                pl.col("cleaned_fragment_formulas").alias("fragment_formulas"),
            ]
        )
        .spectral_info.spectral_info_score(distance_metric="l2", ignore_hydrogens=True)
        .alias("spectral_information_score"),
        pl.struct(
            [
                pl.col("precursor_formula_array").alias("precursor_formula"),
                pl.col("cleaned_fragment_formulas").alias("fragment_formulas"),
            ]
        )
        .spectral_info.spectral_info_score(distance_metric="l2", ignore_hydrogens=False)
        .alias("spectral_information_score_with_hydrogens"),
    )

    logger.info("Collecting full library with streaming engine...")
    df_library = lf.collect(engine="streaming")

    logger.info("Writing full library to %s", output_library_path)
    output_library_path.parent.mkdir(parents=True, exist_ok=True)
    df_library.write_parquet(output_library_path)

    t_end = perf_counter()
    logger.info(
        "Library written: %d spectra in %.3fs", len(df_library), t_end - t_start
    )

    # Log score statistics for quick verification
    for col in ["spectral_information_score", "spectral_information_score_with_hydrogens"]:
        if col in df_library.columns:
            score_stats = df_library[col].describe()
            logger.info("Recomputed %s statistics:\n%s", col, score_stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute spectral information scores in a library parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/home/analytit_admin/Data/spectral_libs/info_score/combined_library.parquet"
        ),
        help="Path to the original library parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_recomputed_info.parquet"
        ),
        help="Path for the output library parquet.",
    )
    args = parser.parse_args()

    recompute_and_save_library(
        original_library_path=args.input,
        output_library_path=args.output,
    )


if __name__ == "__main__":
    main()
