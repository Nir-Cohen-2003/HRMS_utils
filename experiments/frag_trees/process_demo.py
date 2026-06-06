"""
Process the synthetic demo MGF and build/visualize the fragmentation tree.
"""

from pathlib import Path
import sys

# Ensure the workspace root is on path so hrms_utils imports work
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import polars as pl
from hrms_utils.formats.spectral_library import process_single_file
from fragmentation_tree import (
    build_fragmentation_trees,
    visualize_tree,
)

# ---------------------------------------------------------------------------
# 1. Process MGF file
# ---------------------------------------------------------------------------
mgf_path = Path("/home/ser/dev/HRMS_utils/demo_clean.mgf")
output_dir = Path("/home/ser/dev/HRMS_utils/frag_tree_outputs")
output_dir.mkdir(exist_ok=True)

print(f"Processing {mgf_path.name} ...")
df = process_single_file(mgf_path, includes_MSn=True)
print(f"Processed dataframe: {df.height} rows, {len(df.columns)} columns")
print(f"Columns: {df.columns}")

# Show the data
print("\n--- DataFrame contents ---")
print(df.select(["base_inchikey", "mslevel", "precursor_mz", "precursor_type", "cleaned_fragment_formulas_str"]))

# Save processed dataframe
df.write_parquet(output_dir / "demo_clean_processed.parquet")

# ---------------------------------------------------------------------------
# 2. Build fragmentation trees (mass-based, 5 ppm)
# ---------------------------------------------------------------------------
print("\n--- Building MASS-BASED tree (5 ppm tolerance) ---")
trees_mass = build_fragmentation_trees(df, mass_tolerance_ppm=5.0)
print(f"Built {len(trees_mass)} fragmentation trees")

for key, tree in trees_mass.items():
    print(f"\nTree for {key}:")
    print(f"  Fragments: {tree.n_fragments}")
    print(f"  Fragment formulas: {tree.fragment_formulas_str}")
    print(f"  Precursor index: {tree.precursor_idx}")
    print(f"  Precursor formula: {tree.fragment_formulas_str[tree.precursor_idx]}")
    print(f"  MS levels: {tree.spectrum_mslevels.tolist()}")
    print(f"  Edge weights (non-zero):")
    n = tree.n_fragments
    for i in range(n):
        for j in range(n):
            w = tree.edge_weights[i, j]
            if w > 0:
                print(f"    {tree.fragment_formulas_str[i]} -> {tree.fragment_formulas_str[j]} (weight={w:.2f})")

    # Save plot
    out_path = output_dir / "demo_clean_tree_mass_based.png"
    visualize_tree(tree, output_path=str(out_path))
    print(f"  Saved plot to: {out_path}")

# ---------------------------------------------------------------------------
# 3. Build fragmentation trees (formula-based, i.e. 0 ppm / exact match)
# ---------------------------------------------------------------------------
print("\n--- Building FORMULA-BASED tree (0 ppm tolerance) ---")
trees_formula = build_fragmentation_trees(df, mass_tolerance_ppm=0.0)
print(f"Built {len(trees_formula)} fragmentation trees")

for key, tree in trees_formula.items():
    print(f"\nTree for {key}:")
    print(f"  Fragments: {tree.n_fragments}")
    print(f"  Fragment formulas: {tree.fragment_formulas_str}")
    print(f"  Precursor index: {tree.precursor_idx}")
    print(f"  Precursor formula: {tree.fragment_formulas_str[tree.precursor_idx]}")
    print(f"  MS levels: {tree.spectrum_mslevels.tolist()}")
    print(f"  Edge weights (non-zero):")
    n = tree.n_fragments
    for i in range(n):
        for j in range(n):
            w = tree.edge_weights[i, j]
            if w > 0:
                print(f"    {tree.fragment_formulas_str[i]} -> {tree.fragment_formulas_str[j]} (weight={w:.2f})")

    # Save plot
    out_path = output_dir / "demo_clean_tree_formula_based.png"
    visualize_tree(tree, output_path=str(out_path))
    print(f"  Saved plot to: {out_path}")

print("\nDone!")
