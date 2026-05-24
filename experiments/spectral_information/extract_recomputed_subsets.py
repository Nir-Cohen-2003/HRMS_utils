# HRMS_utils/experiments/spectral_information/extract_recomputed_subsets.py
"""
Extract NIST and FragHub subsets from a recombined library parquet and overwrite
the original per-source files with recomputed information scores.

Why: The original NIST and FragHub parquets do not contain cleaned_fragment_formulas,
so spectral_information_score cannot be recomputed from them directly. This script
splits an already-recomputed combined library back into per-source files and
overwrites the originals, ensuring all derived files contain consistent, up-to-date
score columns.

Usage (from this directory):
    pixi run extract_recomputed_subsets

Or directly:
    python extract_recomputed_subsets.py
"""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def extract_and_save_subsets(
    combined_library_path: Path,
    nist_output_path: Path,
    fraghub_output_path: Path,
) -> None:
    """
    Read the combined recomputed library, split into NIST and FragHub subsets,
    and write each subset to its own parquet file.

    NIST records are identified by non-null nist_id.
    FragHub records are identified by null nist_id.

    Args:
        combined_library_path: Path to the combined recomputed library parquet.
        nist_output_path: Path where the NIST subset parquet will be written.
        fraghub_output_path: Path where the FragHub subset parquet will be written.
    """
    logger.info("Loading combined library from %s", combined_library_path)
    lf = pl.scan_parquet(str(combined_library_path))

    schema_names = lf.collect_schema().names()
    assert "nist_id" in schema_names, (
        f"Combined library must contain 'nist_id' column to distinguish NIST from FragHub. "
        f"Available columns: {schema_names}"
    )

    t_start = perf_counter()

    # NIST: nist_id is not null
    logger.info("Extracting NIST subset (nist_id is not null)...")
    nist_lf = lf.filter(pl.col("nist_id").is_not_null())
    nist_df = nist_lf.collect(engine="streaming")
    logger.info("NIST subset: %d spectra", len(nist_df))

    nist_output_path.parent.mkdir(parents=True, exist_ok=True)
    nist_df.write_parquet(nist_output_path)
    logger.info("Wrote NIST subset to %s", nist_output_path)

    # FragHub: nist_id is null
    logger.info("Extracting FragHub subset (nist_id is null)...")
    fraghub_lf = lf.filter(pl.col("nist_id").is_null())
    fraghub_df = fraghub_lf.collect(engine="streaming")
    logger.info("FragHub subset: %d spectra", len(fraghub_df))

    fraghub_output_path.parent.mkdir(parents=True, exist_ok=True)
    fraghub_df.write_parquet(fraghub_output_path)
    logger.info("Wrote FragHub subset to %s", fraghub_output_path)

    t_end = perf_counter()
    logger.info("Extraction completed in %.3fs", t_end - t_start)

    # Verify score statistics for quick sanity check
    for label, df in [("NIST", nist_df), ("FragHub", fraghub_df)]:
        for col in ["spectral_information_score", "spectral_information_score_with_hydrogens"]:
            if col in df.columns:
                mean_score = df[col].mean()
                logger.info("%s %s mean: %.4f", label, col, mean_score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract NIST and FragHub subsets from a recombined recomputed library parquet."
    )
    parser.add_argument(
        "--combined",
        type=Path,
        default=Path(
            "/home/analytit_admin/Data/spectral_libs/info_score/combined_library_recomputed_info.parquet"
        ),
        help="Path to the combined recomputed library parquet.",
    )
    parser.add_argument(
        "--output-nist",
        type=Path,
        default=Path(
            "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/NIST.parquet"
        ),
        help="Path for the output NIST subset parquet (default: overwrites original).",
    )
    parser.add_argument(
        "--output-fraghub",
        type=Path,
        default=Path(
            "/home/analytit_admin/Data/spectral_libs/msp_for_Yonathan/fraghub.parquet"
        ),
        help="Path for the output FragHub subset parquet (default: overwrites original).",
    )
    args = parser.parse_args()

    extract_and_save_subsets(
        combined_library_path=args.combined,
        nist_output_path=args.output_nist,
        fraghub_output_path=args.output_fraghub,
    )


if __name__ == "__main__":
    main()
