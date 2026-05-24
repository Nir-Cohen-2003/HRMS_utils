#!/usr/bin/env python3
"""
Reconstruct the full pairs file with molecule indices and Tanimoto similarity.

This script reconstructs the expected schema for plot_similarity_vs_info.py from:
1. Pairs file (with spectrum indices and dotprod_similarity)
2. Tanimoto file (with molecule indices and tanimoto scores)
3. Library file (to map spectrum indices to molecule indices)

Uses Polars streaming engine throughout - no manual chunking needed.

Output: A parquet file with columns:
    - idx_left, idx_right (spectrum indices, from pairs file)
    - mol_idx_left, mol_idx_right (computed from library)
    - dotprod_similarity (from pairs file)
    - tanimoto_similarity (joined from tanimoto file)
"""

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def reconstruct_pairs_with_tanimoto(
    pairs_path: Path,
    tanimoto_path: Path,
    library_path: Path,
    output_path: Path,
) -> None:
    """
    Reconstruct the full pairs file with all columns needed for plotting.
    
    Uses Polars streaming engine for memory-efficient processing.
    Steps:
    1. Build spectrum→molecule mapping from library (minimal columns)
    2. Join mapping to pairs to add molecule indices
    3. Join Tanimoto scores on molecule pairs (bidirectional)
    4. Sink to parquet using streaming
    """
    logger.info("Starting reconstruction with streaming engine...")
    logger.info(f"  Pairs: {pairs_path}")
    logger.info(f"  Tanimoto: {tanimoto_path}")
    logger.info(f"  Library: {library_path}")
    logger.info(f"  Output: {output_path}")
    
    # Step 1: Build spectrum→molecule mapping (lazy, minimal columns)
    logger.info("Building spectrum-to-molecule mapping...")
    library_lf = pl.scan_parquet(str(library_path))
    
    # Compute mol_idx as dense rank of base_inchikey, matching similarity_vs_info.py
    spec_to_mol_lf = (
        library_lf
        .select(["msp_index", "base_inchikey"])
        .with_columns(
            pl.col("base_inchikey")
            .rank(method="dense")
            .cast(pl.Int64)
            .alias("mol_idx")
        )
        .select(["msp_index", "mol_idx"])
    )
    
    # Step 2: Load tanimoto data and prepare bidirectional lookup (minimal columns)
    logger.info("Preparing Tanimoto lookup...")
    tanimoto_lf = pl.scan_parquet(str(tanimoto_path))
    
    # Validate minimal required columns exist
    tanimoto_schema = tanimoto_lf.collect_schema()
    required_tanimoto_cols = ["mol1_idx", "mol2_idx", "tanimoto"]
    for col in required_tanimoto_cols:
        if col not in tanimoto_schema.names():
            raise ValueError(f"Tanimoto file missing required column '{col}'. Has: {tanimoto_schema.names()}")
    
    # Create bidirectional tanimoto lookup
    tanimoto_forward = tanimoto_lf.select([
        pl.col("mol1_idx").cast(pl.Int64).alias("mol_idx_left"),
        pl.col("mol2_idx").cast(pl.Int64).alias("mol_idx_right"),
        pl.col("tanimoto").cast(pl.Float32).alias("tanimoto_similarity"),
    ])
    
    tanimoto_reverse = tanimoto_lf.select([
        pl.col("mol2_idx").cast(pl.Int64).alias("mol_idx_left"),
        pl.col("mol1_idx").cast(pl.Int64).alias("mol_idx_right"),
        pl.col("tanimoto").cast(pl.Float32).alias("tanimoto_similarity"),
    ])
    
    tanimoto_bidirectional = pl.concat([tanimoto_forward, tanimoto_reverse])
    
    # Step 3: Process pairs with streaming joins
    logger.info("Processing pairs with streaming joins...")
    pairs_lf = pl.scan_parquet(str(pairs_path))
    
    # Validate pairs columns
    pairs_schema = pairs_lf.collect_schema()
    required_pairs_cols = ["idx_left", "idx_right", "dotprod_similarity"]
    for col in required_pairs_cols:
        if col not in pairs_schema.names():
            raise ValueError(f"Pairs file missing required column '{col}'. Has: {pairs_schema.names()}")
    
    # Join pairs with spectrum→mol mapping for left side
    pairs_with_mol = (
        pairs_lf
        .select(["idx_left", "idx_right", "dotprod_similarity"])
        .join(
            spec_to_mol_lf.select([
                pl.col("msp_index").alias("idx_left"),
                pl.col("mol_idx").alias("mol_idx_left"),
            ]),
            on="idx_left",
            how="left"
        )
        # Join for right side
        .join(
            spec_to_mol_lf.select([
                pl.col("msp_index").alias("idx_right"),
                pl.col("mol_idx").alias("mol_idx_right"),
            ]),
            on="idx_right",
            how="left"
        )
    )
    
    # Join tanimoto scores on molecule pairs
    result_lf = (
        pairs_with_mol
        .join(
            tanimoto_bidirectional,
            on=["mol_idx_left", "mol_idx_right"],
            how="left"
        )
        .select([
            "idx_left",
            "idx_right",
            "mol_idx_left",
            "mol_idx_right",
            "dotprod_similarity",
            "tanimoto_similarity",
        ])
    )
    
    # Step 4: Sink to parquet using streaming engine
    logger.info("Streaming results to output file...")
    result_lf.sink_parquet(output_path, maintain_order=False)
    
    # Log result stats
    result_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(f"Reconstruction complete: {result_count:,} rows written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct pairs file with molecule indices and Tanimoto similarity using streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reconstruct_pairs_with_tanimoto.py \\
      --pairs data_pairs_260311.parquet \\
      --tanimoto data_pairs_260311.tanimoto.parquet \\
      --library data.parquet \\
      --output data_pairs_260311_with_tanimoto.parquet
        """,
    )
    
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="Path to pairs parquet file (with idx_left, idx_right, dotprod_similarity)"
    )
    parser.add_argument(
        "--tanimoto",
        type=Path,
        required=True,
        help="Path to Tanimoto parquet file (with mol1_idx, mol2_idx, tanimoto)"
    )
    parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Path to library parquet file (with msp_index, base_inchikey)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write reconstructed parquet file"
    )
    
    args = parser.parse_args()
    
    # Validate inputs exist
    for path, name in [(args.pairs, "pairs"), (args.tanimoto, "tanimoto"), (args.library, "library")]:
        if not path.exists():
            logger.error(f"{name} file not found: {path}")
            sys.exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Run reconstruction
    reconstruct_pairs_with_tanimoto(
        pairs_path=args.pairs,
        tanimoto_path=args.tanimoto,
        library_path=args.library,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
