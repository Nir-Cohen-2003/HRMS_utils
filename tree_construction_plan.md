# Fragmentation Tree Construction Implementation Plan

This document outlines the implementation plan for constructing fragmentation trees from MS^n spectra data using Polars.

---

## Overview

Construct directed fragmentation graphs (parent → child) from MSP data. The graph represents fragmentation cascades where edges indicate possible parent-child relationships based on formula subtraction rules and MS level constraints.

### Key Design Decisions

- **One tree per molecule**: Trees are constructed per `(base_inchikey, ion_mode)` group.
- **MS level hierarchy**: Higher MS levels (MS3, MS4, MS5...) provide stronger evidence for parent-child relationships than lower levels (MS2) or weak/inferred edges.
- **Formula identity**: Fragments are identified by formula only — the same formula in different MS levels is the same node.
- **Polars-native**: All operations use Polars expressions — no Python loops.

---

## File Location

```
src/hrms_utils/fragmentation/tree_construction.py
```

---

## Function Signature

```python
def construct_fragmentation_graphs_from_msp_data(
    msp_frame: pl.LazyFrame,
    water_absorption: bool = False,
) -> pl.LazyFrame:
```

---

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `index_molecule` | `UInt32` | Sorted molecule index (sorted, set_sorted) |
| `parent_formula` | `Array(Int32, 12)` | Parent node formula |
| `child_formula` | `Array(Int32, 12)` | Child node formula |
| `edge_ms_level` | `Int64` | MS level evidence (0 = weak/inferred, 2+ = from that MS level group) |

---

## Implementation Steps

### Step 0: Create Sorted Molecule Index

**Goal**: Replace `(base_inchikey, ion_mode)` with a sorted integer molecule index to enable Polars streaming engine.

- [ ] Create molecule index by selecting unique `(base_inchikey, ion_mode)` pairs from `msp_frame`
- [ ] Sort by `(base_inchikey, ion_mode)` to ensure deterministic ordering
- [ ] Add row index column `index_molecule` (UInt32)
- [ ] Join this index back to `msp_frame` on `(base_inchikey, ion_mode)`
- [ ] All subsequent operations use `index_molecule` as the molecule key

---

### Step 1: Partial Merge (Per Precursor Group)

**Goal**: Group fragments by precursor to create MS-level-specific edge sources.

- [ ] Group `msp_frame` by `(index_molecule, precursor_formula_array)`
- [ ] Aggregate:
  - Merge `cleaned_fragment_formulas` lists with `concat_list().list.unique()`
  - Include `precursor_formula_array` as a node in the fragment list (add the precursor as a potential parent)
  - Take `max(mslevel)` as `edge_ms_level` for this group
- [ ] Add row index `index_precursor_group` for the self-join
- [ ] Sort by `index_molecule` and call `.set_sorted("index_molecule")`

---

### Step 2: Compute Level-Specific Pairs (Within Precursor Groups)

**Goal**: Find parent-child edges within each precursor group (MS-level-specific evidence).

- [ ] Explode the fragment list from Step 1
- [ ] Self-join on `index_precursor_group` to get all potential parent-child pairs
- [ ] Apply formula subtraction filter:
  - `(parent - child + water_vector).arr.min() >= 0` (valid subtraction)
  - `(parent - child + water_vector).arr.max() > 0` (exclude self-edges)
  - Note: `water_vector` is zeros unless `water_absorption=True`, then `[2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]` (H₂O)
- [ ] Join back to get `index_molecule` and `edge_ms_level`
- [ ] Select: `(index_molecule, parent_formula, child_formula, edge_ms_level)`
- [ ] Ensure `parent_formula` and `child_formula` column names match the schema

---

### Step 3: Full Merge (Per Molecule) — Weak Pairs

**Goal**: Find all potential parent-child edges across the entire molecule (inferred/weak evidence).

- [ ] Group by `index_molecule` (not by precursor)
- [ ] Aggregate:
  - Merge all `cleaned_fragment_formulas` lists across all precursor groups
  - Include all `precursor_formula_array` values as potential nodes
  - Take unique of all formulas
- [ ] Sort by `index_molecule` and call `.set_sorted("index_molecule")`
- [ ] Explode the fragment list
- [ ] Self-join on `index_molecule`
- [ ] Apply the same formula subtraction filter as Step 2
- [ ] Assign `edge_ms_level = 0` (lowest priority, weak/inferred)
- [ ] Select: `(index_molecule, parent_formula, child_formula, edge_ms_level)`

---

### Step 4: Combine and Filter by MS Level Priority

**Goal**: For each child, keep only edges at the highest available MS level.

- [ ] Concatenate level-specific pairs (Step 2) and weak pairs (Step 3)
- [ ] Add column `max_edge_ms_level` using window function:
  ```python
  pl.col("edge_ms_level").max().over(["index_molecule", "child_formula"])
  ```
- [ ] Filter to keep only edges where `edge_ms_level == max_edge_ms_level`
- [ ] Drop the `max_edge_ms_level` temporary column

**Rationale**: If a child has a parent at MS level 4, all edges from levels 0, 2, 3 are discarded. This enforces the "strong evidence overrides weak evidence" rule across the multi-level MS hierarchy.

---

### Step 5: Final Deduplication and Cleanup

**Goal**: Remove duplicate edges and prepare final output.

- [ ] Deduplicate on `(index_molecule, parent_formula, child_formula)` using `.unique()`
  - If the same edge appears at multiple MS levels with the same max level, this removes duplicates
- [ ] Sort by `index_molecule` and call `.set_sorted("index_molecule")`
- [ ] Verify output schema matches:
  - `index_molecule: UInt32`
  - `parent_formula: Array(Int32, 12)`
  - `child_formula: Array(Int32, 12)`
  - `edge_ms_level: Int64`

---

## Bugs Fixed (vs. Original Code)

| Bug | How Fixed |
|-----|-----------|
| Self-edges allowed (`A - A >= 0`) | Added `arr.max() > 0` filter to require strict decrease in at least one element |
| `children_with_strong_parents` not scoped per molecule | All operations keyed on `index_molecule` instead of global formula matching |
| Strong/weak filtering doesn't handle multi-level MS | Replaced with explicit `edge_ms_level` and window-based max filtering |
| Precursor not included as a node | Precursor formula explicitly added to fragment lists before exploding |
| No return type / output schema | Function returns `pl.LazyFrame` with documented schema |

---

## Performance Considerations

- **Polars streaming**: `index_molecule` is sorted with `.set_sorted()` to enable Polars 1.38+ streaming engine
- **Lazy evaluation**: All operations stay lazy until caller invokes `.collect()`
- **Join strategy**: Self-joins are within-group (keyed on group index), not full cross-joins
- **Window functions**: `max().over()` for MS level filtering is efficient for this data volume
- **Complexity**: O(n²) per molecule for self-joins, where n = unique fragments. Typical HRMS has 10-100 fragments per molecule.

---

## Testing Checklist (Post-Implementation)

- [ ] Verify no self-edges in output (`parent != child` for all rows)
- [ ] Verify all edges satisfy formula subtraction constraint
- [ ] Verify `edge_ms_level` 0 edges are only present when no higher-level edge exists for that child
- [ ] Test with `water_absorption=True` and verify H₂O is correctly added
- [ ] Test with molecules having MS2, MS3, MS4, MS5 data and verify level hierarchy is respected
- [ ] Performance test: run on sample MSP file with 1000+ molecules

---

## Deferred Features

These are intentionally **not** included in this implementation:

- **Transitive reduction**: If A→C and A→B→C both survive filtering, both are kept as valid edges
- **Intensity weights**: Fragment intensities are not carried into the tree (could be added later as edge weights)
- **Cycle detection**: The formula subtraction constraint guarantees a DAG; no detection needed
- **Root assignment**: The precursor formula is a node but not explicitly marked as "root"

---

## Dependencies

- `polars >= 1.38.0`
- `numpy` (for `water_vector`)
- `hrms_utils.formula_annotation.element_table.NUM_ELEMENTS`

---

## Notes

- The `water_absorption` flag allows parent formulas to effectively gain H₂O before fragmenting. This models water addition reactions in fragmentation.
- MS levels are extracted from the MSP data and should be integers (2, 3, 4, 5...). Higher integers = stronger evidence.
- The output is a LazyFrame of edges. Callers can materialize it with `.collect()` or join it with other molecule metadata as needed.
