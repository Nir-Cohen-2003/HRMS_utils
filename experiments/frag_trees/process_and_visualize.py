"""
Process enamol_neg_msn.mgf through spectral library pipeline,
build fragmentation trees, and visualize 5 compounds.
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
mgf_path = Path("/home/ser/dev/HRMS_utils/enamol_neg_msn.mgf")
output_dir = Path("/home/ser/dev/HRMS_utils/frag_tree_outputs")
output_dir.mkdir(exist_ok=True)

print(f"Processing {mgf_path.name} ...")
df = process_single_file(mgf_path, includes_MSn=True)
print(f"Processed dataframe: {df.height} rows, {len(df.columns)} columns")

# Save processed dataframe for inspection
df.write_parquet(output_dir / "processed_library.parquet")
print(f"Saved processed library to {output_dir / 'processed_library.parquet'}")

# ---------------------------------------------------------------------------
# 2. Build fragmentation trees
# ---------------------------------------------------------------------------
print("\nBuilding fragmentation trees ...")
trees = build_fragmentation_trees(df, mass_tolerance_ppm=5.0)
print(f"Built {len(trees)} fragmentation trees")

# ---------------------------------------------------------------------------
# 3. Select 5 compounds with the most fragments (interesting trees)
# ---------------------------------------------------------------------------
# Group by base_inchikey and pick those with MSn data and many fragments
tree_stats = []
for key, tree in trees.items():
    base_inchikey, ion_mode, precursor_type = key
    # Count unique mslevels
    unique_mslevels = set(tree.spectrum_mslevels.tolist())
    tree_stats.append({
        "base_inchikey": base_inchikey,
        "ion_mode": ion_mode,
        "precursor_type": precursor_type,
        "n_fragments": tree.n_fragments,
        "n_spectra": len(tree.spectrum_fragments),
        "mslevels": sorted(unique_mslevels),
        "key": key,
        "tree": tree,
    })

# Sort by number of fragments descending, prefer those with MS3+
tree_stats_sorted = sorted(
    tree_stats,
    key=lambda x: (max(x["mslevels"]) > 2, x["n_fragments"]),
    reverse=True,
)

print("\nTop 10 trees by fragment count (with MSn preference):")
for stat in tree_stats_sorted[:10]:
    print(f"  {stat['base_inchikey']} | {stat['ion_mode']} | {stat['precursor_type']} | "
          f"fragments={stat['n_fragments']} | spectra={stat['n_spectra']} | mslevels={stat['mslevels']}")

# Select top 5
selected = tree_stats_sorted[:5]
print(f"\nSelected {len(selected)} compounds for visualization")

# ---------------------------------------------------------------------------
# 4. Visualize and save
# ---------------------------------------------------------------------------
for i, stat in enumerate(selected, 1):
    tree = stat["tree"]
    key_str = f"{stat['base_inchikey']}_{stat['ion_mode']}_{stat['precursor_type']}"
    # Sanitize filename
    key_str = key_str.replace("/", "_").replace("\\", "_")
    out_path = output_dir / f"tree_{i:02d}_{key_str}.png"

    print(f"\n[{i}/5] Saving tree for {stat['base_inchikey']} ...")
    print(f"     Fragments: {tree.n_fragments}, Spectra: {stat['n_spectra']}, MSlevels: {stat['mslevels']}")
    print(f"     Formulas: {tree.fragment_formulas_str}")

    visualize_tree(tree, output_path=str(out_path))
    print(f"     Saved to: {out_path}")

print(f"\nAll done! Images saved to {output_dir}")
