# HRMS_utils/experiments/spectral_information/run_tanimoto.py
"""
Standalone script to compute Tanimoto similarities for a pairs parquet.

Why: Separates the heavy Tanimoto computation from similarity and plotting so
that each step can be run independently via pixi tasks.

Usage:
    pixi run -e experiments run_tanimoto
"""

import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter

sys.path.append(str(Path(__file__).parents[2] / "src"))

from utils import compute_and_save_tanimoto_scores  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Tanimoto similarities for a pairs parquet."
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="Path to pairs parquet (with idx, idx_right, mol_idx, mol_idx_right, dotprod_similarity).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for output pairs parquet with tanimoto_similarity added.",
    )
    parser.add_argument(
        "--left-snapshot",
        type=Path,
        required=True,
        help="Path to left library snapshot parquet (with idx and smiles).",
    )
    parser.add_argument(
        "--right-snapshot",
        type=Path,
        default=None,
        help="Optional path to right library snapshot parquet (with idx and smiles). Defaults to left.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Batch size for Tanimoto computation.",
    )
    args = parser.parse_args()

    assert args.pairs.exists(), f"Pairs parquet not found: {args.pairs}"
    assert args.left_snapshot.exists(), f"Left snapshot not found: {args.left_snapshot}"

    logger.info("Starting Tanimoto computation...")
    t_start = perf_counter()

    compute_and_save_tanimoto_scores(
        input_parquet_path=args.pairs,
        output_path=args.output,
        left_library_parquet_path=args.left_snapshot,
        right_library_parquet_path=args.right_snapshot,
        batch_size=args.batch_size,
    )

    t_end = perf_counter()
    logger.info("Tanimoto computation complete in %.3fs", t_end - t_start)


if __name__ == "__main__":
    main()
