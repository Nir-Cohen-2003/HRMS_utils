"""
Integration script: read spectral library parquet → build MS2 + full fragmentation trees
→ save both to NPZ files → print key alignment summary.

Usage:
    pixi run -e experiments python experiments/frag_trees/build_and_store_trees.py \
        --input cladribine.parquet \
        --output-dir experiments/frag_trees/out/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from fragmentation_tree import (
    build_fragmentation_trees,
    FragmentationTree,
    FragmentationTreeConfig,
)
from ms2_tree_builder import build_ms2_trees
from tree_storage import (
    align_keys,
    save_trees_npz,
    TreeKey,
    TreeStorageConfig,
)

logger = logging.getLogger("build_and_store_trees")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_key_list(
    trees: dict[tuple[str, str, str], FragmentationTree],
) -> list[TreeKey]:
    """Build a sorted list of TreeKey from a tree dict keyed by (base_inchikey, ion_mode, precursor_type)."""
    return sorted(
        [
            TreeKey(base_inchikey=k[0], ion_mode=k[1], precursor_type=k[2])
            for k in trees
        ],
        key=lambda tk: (tk.base_inchikey, tk.ion_mode, tk.precursor_type),
    )


def _print_per_tree_stats(
    label: str,
    trees: dict[tuple[str, str, str], FragmentationTree],
) -> None:
    """Log per-tree statistics at DEBUG level."""
    for key, tree in sorted(trees.items()):
        n_edges = int((tree.edge_weights > 0).sum())
        logger.debug(
            "  %s  %s | %s | %s: %d nodes, %d edges, %d spectra",
            label,
            key[0],
            key[1],
            key[2],
            tree.n_fragments,
            n_edges,
            len(tree.spectrum_fragments),
        )


def _print_key_mismatches(
    label: str,
    keys: set[TreeKey],
) -> None:
    """Print mismatched keys (verbose only)."""
    if not keys:
        return
    print(f"\n{label} (missing in other file):")
    for k in sorted(keys, key=lambda x: (x.base_inchikey, x.ion_mode, x.precursor_type)):
        print(f"  {k.base_inchikey} | {k.ion_mode} | {k.precursor_type}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and store fragmentation trees from a spectral library. "
            "Reads a parquet file, builds both MS2-only and full MSn "
            "fragmentation trees, saves them as compressed NPZ files, "
            "and prints a key alignment summary."
        ),
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to spectral library parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output NPZ files (ms2_trees.npz, full_trees.npz).",
    )
    parser.add_argument(
        "--tolerance-ppm",
        type=float,
        default=5.0,
        help="Mass tolerance in ppm for merging and annotation (default: 5.0).",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable NPZ compression (use np.savez instead of np.savez_compressed).",
    )
    parser.add_argument(
        "--ion-mode",
        type=str,
        default=None,
        choices=["P", "N"],
        help="Optional ion mode filter (P or N).",
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging and per-tree statistics.",
    )
    verbosity.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output (warning level only).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ms2_npz_path = output_dir / "ms2_trees.npz"
    full_npz_path = output_dir / "full_trees.npz"

    # ------------------------------------------------------------------
    # Read spectral library
    # ------------------------------------------------------------------
    logger.info("Reading spectral library from: %s", input_path)
    df = pl.read_parquet(input_path)
    logger.info("Loaded %d rows, %d columns.", df.shape[0], df.shape[1])

    if args.ion_mode is not None:
        df = df.filter(pl.col("ion_mode") == args.ion_mode)
        logger.info(
            "Filtered to ion_mode='%s': %d rows remaining.",
            args.ion_mode,
            df.shape[0],
        )

    if df.is_empty():
        logger.error("Spectral library is empty after filtering. Nothing to process.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    tree_config = FragmentationTreeConfig(
        merge_tolerance_ppm=args.tolerance_ppm,
        annotation_tolerance_ppm=args.tolerance_ppm,
    )

    # ------------------------------------------------------------------
    # Build full MSn trees
    # ------------------------------------------------------------------
    logger.info("Building full MSn fragmentation trees...")
    full_trees = build_fragmentation_trees(df, tree_config)
    logger.info("Built %d full tree(s).", len(full_trees))

    if args.verbose and full_trees:
        _print_per_tree_stats("FULL", full_trees)

    # ------------------------------------------------------------------
    # Build MS2-only trees
    # ------------------------------------------------------------------
    logger.info("Building MS2-only fragmentation trees...")
    ms2_trees = build_ms2_trees(df, tree_config)
    logger.info("Built %d MS2 tree(s).", len(ms2_trees))

    if args.verbose and ms2_trees:
        _print_per_tree_stats("MS2 ", ms2_trees)

    # ------------------------------------------------------------------
    # Save to NPZ
    # ------------------------------------------------------------------
    compress = not args.no_compress

    logger.info("Saving MS2 trees to: %s", ms2_npz_path)
    ms2_storage_config = TreeStorageConfig(
        tree_type="ms2",
        compress=compress,
        include_spectrum_metadata=False,
    )
    save_trees_npz(ms2_trees, ms2_npz_path, ms2_storage_config)

    logger.info("Saving full trees to: %s", full_npz_path)
    full_storage_config = TreeStorageConfig(
        tree_type="full",
        compress=compress,
        include_spectrum_metadata=True,
    )
    save_trees_npz(full_trees, full_npz_path, full_storage_config)

    # ------------------------------------------------------------------
    # Key alignment summary
    # ------------------------------------------------------------------
    ms2_keys = _build_key_list(ms2_trees)
    full_keys = _build_key_list(full_trees)

    pairs = align_keys(ms2_keys, full_keys, fail_on_mismatch=False)

    ms2_set = set(ms2_keys)
    full_set = set(full_keys)
    only_ms2: set[TreeKey] = ms2_set - full_set
    only_full: set[TreeKey] = full_set - ms2_set

    # Print summary to stdout
    print("=" * 60)
    print("  Key Alignment Summary")
    print("=" * 60)
    print(f"  MS2 trees:           {len(ms2_trees)}")
    print(f"  Full trees:          {len(full_trees)}")
    print(f"  Matched keys:        {len(pairs)}")
    print(f"  MS2-only keys:       {len(only_ms2)}")
    print(f"  Full-only keys:      {len(only_full)}")
    print(f"  MS2 NPZ:             {ms2_npz_path}")
    print(f"  Full NPZ:            {full_npz_path}")
    print("=" * 60)

    if args.verbose:
        _print_key_mismatches("MS2-only keys", only_ms2)
        _print_key_mismatches("Full-only keys", only_full)

    logger.info("Done.")


if __name__ == "__main__":
    main()
