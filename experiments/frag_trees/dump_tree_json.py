"""
Dump fragmentation tree for cladribine as JSON with fragments, children, and edge weights.

Usage:
    pixi run -e experiments python experiments/frag_trees/dump_tree_json.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure workspace root is on path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import polars as pl
import numpy as np

from experiments.frag_trees.fragmentation_tree import build_fragmentation_trees


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main() -> None:
    library_path = Path("cladribine.parquet")
    output_path = Path("cladribine_frag_tree.json")

    df = pl.read_parquet(library_path)
    trees = build_fragmentation_trees(df)

    print(f"Built {len(trees)} tree(s)")

    all_trees_data = {}

    for (base_inchikey, ion_mode, precursor_type), tree in trees.items():
        key = f"{base_inchikey}_{ion_mode}_{precursor_type.replace('/', '_').replace('[', '').replace(']', '')}"
        print(f"\n{'='*60}")
        print(f"Tree: {key}")
        print(f"  Fragments: {tree.n_fragments}")
        print(f"  Precursor: {tree.fragment_formulas_str[tree.precursor_idx]}")
        print(f"  Formulas: {tree.fragment_formulas_str}")
        print(f"  Precursor array (H,C,N,O,F,Na,P,S,Cl,K,Br,I): {tree.precursor_formula.tolist()}")

        # Build fragment list with children
        fragments = []
        n = tree.n_fragments
        for i in range(n):
            formula = tree.fragment_formulas_str[i]
            formula_array = tree.fragment_formulas[i].tolist()
            total_atoms = int(np.sum(tree.fragment_formulas[i]))
            mslevel = int(tree.spectrum_mslevels[i]) if i < len(tree.spectrum_mslevels) else -1

            children = []
            for j in range(n):
                w = float(tree.edge_weights[i, j])
                if w > 0:
                    children.append({
                        "index": j,
                        "formula": tree.fragment_formulas_str[j],
                        "formula_array": tree.fragment_formulas[j].tolist(),
                        "weight": w,
                    })

            fragments.append({
                "index": i,
                "formula": formula,
                "formula_array": formula_array,
                "total_atoms": total_atoms,
                "is_precursor": i == tree.precursor_idx,
                "children": children,
            })

        tree_data = {
            "base_inchikey": tree.base_inchikey,
            "ion_mode": tree.ion_mode,
            "precursor_type": precursor_type,
            "precursor_formula": tree.fragment_formulas_str[tree.precursor_idx],
            "precursor_formula_array": tree.precursor_formula.tolist(),
            "n_fragments": tree.n_fragments,
            "fragments": fragments,
            "edge_matrix": tree.edge_weights.tolist(),
        }
        all_trees_data[key] = tree_data

    with open(output_path, "w") as f:
        json.dump(all_trees_data, f, indent=2, cls=NumpyEncoder)

    print(f"\nSaved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
