"""
Build a fragmentation tree for a single molecule from a spectral library file.

Usage:
    pixi run -e experiments python experiments/frag_trees/build_tree_for_molecule.py \\
        <library_path> <base_inchikey> [--ion-mode P|N] [--output <image.png>]

Errors:
    - Raises SystemExit(2) if the base_inchikey is absent from the file.
    - Raises SystemExit(3) if the molecule has only MS2 data (no MSn).
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure the workspace root is on path.
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import polars as pl

from experiments.frag_trees.fragmentation_tree import (
    build_fragmentation_trees,
    load_and_build_tree,
    visualize_tree,
)


def _check_compound_present(df: pl.DataFrame, base_inchikey: str) -> None:
    """Raise SystemExit(2) if the base_inchikey is absent from the DataFrame."""
    present = df.filter(pl.col("base_inchikey") == base_inchikey)
    if present.is_empty():
        print(
            f"ERROR: base_inchikey '{base_inchikey}' not found in the input file.",
            file=sys.stderr,
        )
        sys.exit(2)


def _check_has_msn(df: pl.DataFrame, base_inchikey: str) -> None:
    """Raise SystemExit(3) if all spectra for the molecule are MS2 only."""
    has_msn = df.filter(
        (pl.col("base_inchikey") == base_inchikey) & (pl.col("mslevel") > 2)
    )
    if has_msn.is_empty():
        mslevels = (
            df.filter(pl.col("base_inchikey") == base_inchikey)
            .select("mslevel")
            .to_series()
            .unique()
            .to_list()
        )
        print(
            f"ERROR: base_inchikey '{base_inchikey}' has no MSn data "
            f"(only mslevels: {sorted(mslevels)}). "
            f"Fragmentation tree construction requires MSn data.",
            file=sys.stderr,
        )
        sys.exit(3)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a fragmentation tree for a single molecule."
    )
    parser.add_argument(
        "library",
        type=str,
        help="Path to spectral library file (parquet, msp, mgf).",
    )
    parser.add_argument("inchikey", type=str, help="Base InChIKey to select.")
    parser.add_argument(
        "--ion-mode",
        type=str,
        default=None,
        choices=["P", "N"],
        help="Ion mode filter.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If provided, save the tree visualization to this PNG path.",
    )
    args = parser.parse_args()

    library_path = Path(args.library)
    if library_path.suffix.lower() in [".msp", ".mspec", ".mgf"]:
        from hrms_utils.formats.spectral_library import process_single_file

        df = process_single_file(library_path, includes_MSn=True)
    else:
        df = pl.read_parquet(library_path)

    # Validate.
    _check_compound_present(df, args.inchikey)
    _check_has_msn(df, args.inchikey)

    # Build tree.
    trees = build_fragmentation_trees(df)
    matching = [
        (key, tree)
        for key, tree in trees.items()
        if key[0] == args.inchikey
        and (args.ion_mode is None or key[1] == args.ion_mode)
    ]
    if not matching:
        print(
            f"ERROR: base_inchikey '{args.inchikey}' is present in the file "
            f"but no fragmentation tree could be built. The molecule may lack "
            f"MS2 spectra or have unannotated fragments.",
            file=sys.stderr,
        )
        sys.exit(4)

    # Print results.
    for key, tree in matching:
        print(f"Tree: {key}")
        print(f"  Fragments: {tree.n_fragments}")
        print(f"  Formulas: {tree.fragment_formulas_str}")
        print(f"  MS levels: {tree.spectrum_mslevels.tolist()}")
        print(f"  Edges:")
        n = tree.n_fragments
        for i in range(n):
            for j in range(n):
                w = tree.edge_weights[i, j]
                if w > 0:
                    print(
                        f"    {tree.fragment_formulas_str[i]} -> "
                        f"{tree.fragment_formulas_str[j]} (weight={w:.2f})"
                    )
        if args.output:
            out_path = Path(args.output)
            if len(matching) > 1:
                stem = out_path.stem
                suffix = out_path.suffix
                key_str = f"{key[0]}_{key[1]}_{key[2]}".replace("/", "_")
                out_path = out_path.with_name(f"{stem}_{key_str}{suffix}")
            visualize_tree(tree, output_path=str(out_path))
            print(f"  Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()
