# FragTree NPZ v1 — Format Specification

**Version:** 1
**Status:** Stable
**Date:** 2026-07-08

## 1. Overview

The FragTree NPZ format stores collections of fragmentation trees (chemical graphs representing
mass-spectrometry fragmentation cascades) as a single compressed `.npz` file per collection.
It is designed to be disk-efficient, NumPy-native, and trivially convertible to PyTorch Geometric
`Data` objects.

Two tree types are supported:

| Tree type | `tree_type` field | Description |
|-----------|-------------------|-------------|
| MS2 tree (`ms2`) | `"ms2"` | Pooled MS2 peaks connected by subformula (superset/subset) relationships. One synthetic spectrum per tree. |
| Full MSn tree (`full`) | `"full"` | Complete fragmentation tree from the MSn pipeline: MS2+MSn peaks, "lowest provable parent" edge rules, per-spectrum metadata. |

**Key design principle:** The two tree types are stored in **separate files** with **matching keys**
so they can be paired for supervised graph learning (MS2 = input, full = target).

---

## 2. File Layout

- **One `.npz` file per tree collection.** Convention: `<prefix>_ms2_trees.npz` and `<prefix>_full_trees.npz`.
- Written with `np.savez_compressed` (compression overhead negligible at expected scale).
- Each file is a zip of named `.npy` arrays with a `format_version` scalar for forward compatibility.

### 2.1 Internal structure

The file uses CSR-offset collation, analogous to PyTorch Geometric's `InMemoryDataset` collation:

```
┌────────────────────────────────────────┐
│  format_version  (scalar, int32)       │
│  tree_type        (scalar, <U4)        │
│                                        │
│  base_inchikey    (n_trees,) <U14     │
│  ion_mode         (n_trees,) <U1      │
│  precursor_type   (n_trees,) <U16     │
│  precursor_formulas (n_trees, 12) int32│
│  precursor_indices  (n_trees,) int32   │
│                                        │
│  node_offsets     (n_trees+1,) int64   │  ← CSR: graph i nodes
│  edge_offsets     (n_trees+1,) int64   │  ← CSR: graph i edges
│                                        │
│  node_features    (total_nodes, 12) int32
│  fragment_errors_ppm (total_nodes,) float64
│  edge_index       (2, total_edges) int32     ← COO, local indices
│  edge_weights     (total_edges,) float64
│                                        │
│  [fragment_formulas_str] (total_nodes,) <U64  ← optional
│                                        │
│  --- FULL-TREE ONLY ---                │
│  spectrum_offsets            (n_trees+1,) int64
│  spectrum_fragments_offsets  (total_spectra+1,) int64
│  spectrum_fragments_flat     (total_fragment_refs,) int32
│  spectrum_mslevels           (total_spectra,) int32
│  spectrum_msn_precursors     (total_spectra,) int32
└────────────────────────────────────────┘
```

---

## 3. Field Reference

### 3.1 Header / metadata

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `format_version` | `()` | int32 | Format version (currently `1`). |
| `tree_type` | `()` | `<U4` | `"ms2"` or `"full"`. |

### 3.2 Keys (per-tree identifiers)

Stored explicitly so pairing across files never depends on row order.

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `base_inchikey` | `(n_trees,)` | `<U14` | First 14 characters of InChIKey. |
| `ion_mode` | `(n_trees,)` | `<U1` | `"P"` (positive) or `"N"` (negative). |
| `precursor_type` | `(n_trees,)` | `<U16` | Adduct string, e.g. `"[M+H]+"`, `"[M-H]-"`. |

### 3.3 Per-tree scalars

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `precursor_formulas` | `(n_trees, 12)` | int32 | Molecular precursor element counts per tree. |
| `precursor_indices` | `(n_trees,)` | int32 | Local node index of the precursor within each graph. |

### 3.4 CSR offsets

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `node_offsets` | `(n_trees + 1,)` | int64 | Cumulative node counts. Graph `i` nodes = `node_features[node_offsets[i]:node_offsets[i+1]]`. `node_offsets[0] == 0`. |
| `edge_offsets` | `(n_trees + 1,)` | int64 | Cumulative edge counts. Graph `i` edges = `edge_index[:, edge_offsets[i]:edge_offsets[i+1]]`. `edge_offsets[0] == 0`. |

### 3.5 Concatenated graph data (CSR-indexed)

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `node_features` | `(total_nodes, 12)` | int32 | Concatenated formula arrays. Element order: **H, C, N, O, F, Na, P, S, Cl, K, Br, I** (indices 0–11). Maps to PyG `x`. |
| `fragment_errors_ppm` | `(total_nodes,)` | float64 | Per-node mass annotation error. |
| `edge_index` | `(2, total_edges)` | int32 | COO edges with **local** node indices (0-based within each graph). Row 0 = source, row 1 = target. See §4 for direction convention. Maps to PyG `edge_index` (cast to `torch.long`). |
| `edge_weights` | `(total_edges,)` | float64 | One weight per edge. Incoming weights sum to 1.0 per child node. Maps to PyG `edge_attr` (reshaped to `(n_edges, 1)`). |

### 3.6 Optional metadata

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `fragment_formulas_str` | `(total_nodes,)` | `<U64` | String formula per node, e.g. `"C10H11ClN5O3"`. Derivable from `node_features` + element table, stored for debug/viz convenience. May be absent. |

### 3.7 Full-tree-only spectrum metadata (present only when `tree_type == "full"`)

These encode the per-spectrum MSn metadata using a **two-level CSR** scheme:

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `spectrum_offsets` | `(n_trees + 1,)` | int64 | Cumulative spectrum counts per tree. Tree `i` spectra = `[spectrum_offsets[i], spectrum_offsets[i+1])`. |
| `spectrum_fragments_offsets` | `(total_spectra + 1,)` | int64 | Cumulative fragment-reference counts per spectrum. |
| `spectrum_fragments_flat` | `(total_fragment_refs,)` | int32 | Flattened per-spectrum fragment indices (local to each graph). |
| `spectrum_mslevels` | `(total_spectra,)` | int32 | MS level per spectrum (2, 3, …). |
| `spectrum_msn_precursors` | `(total_spectra,)` | int32 | MSn precursor node index per spectrum, `-1` if none. |

### 3.8 MS2 file spectrum metadata

MS2 NPZ files **omit** the `spectrum_*` fields from §3.7. When loading an MS2 file,
the loader synthesizes trivial spectrum metadata:

- **Exactly one spectrum per tree** (all fragment indices, `mslevel=2`, `msn_precursor=-1`).

This synthesis relies on the invariant that `build_ms2_trees` produces exactly one spectrum per
tree. The loader asserts this invariant: `assert (spectrum_offsets[1:] - spectrum_offsets[:-1] == 1).all()`.

---

## 4. CSR Offset Convention

### 4.1 Per-graph slicing

To extract graph `i` from a `TreeArrays` object:

```python
def slice_graph(arrays, i):
    n_start = arrays.node_offsets[i]
    n_end = arrays.node_offsets[i + 1]
    e_start = arrays.edge_offsets[i]
    e_end = arrays.edge_offsets[i + 1]

    return {
        "node_features": arrays.node_features[n_start:n_end],
        "fragment_errors_ppm": arrays.fragment_errors_ppm[n_start:n_end],
        "edge_index": arrays.edge_index[:, e_start:e_end],
        "edge_weights": arrays.edge_weights[e_start:e_end],
        "precursor_idx": arrays.precursor_indices[i],
    }
```

**Key properties:**
- `edge_index` values are **local** to each graph (0-based within the graph's node range).
- `node_offsets[i+1] - node_offsets[i]` gives the node count of graph `i`.
- `edge_offsets[i+1] - edge_offsets[i]` gives the edge count of graph `i`.
- For MS2 files: spectrum `j` of tree `i` is accessed via `spectrum_offsets[i] + j`.

### 4.2 Single-node trees (precursor only, zero edges)

A tree with `n=1` node and `e=0` edges is a valid state (all fragments dropped by orphan removal,
only the precursor survives). The format handles this naturally:

```
node_offsets[i+1] - node_offsets[i] == 1     # one node
edge_offsets[i+1] - edge_offsets[i] == 0     # zero edges
node_features slice:     shape (1, 12)
edge_index slice:        shape (2, 0)   ← empty, valid
edge_weights slice:      shape (0,)     ← empty, valid
```

`np.nonzero` on a `(1, 1)` all-zero dense matrix correctly yields empty index arrays,
so COO extraction requires no special handling for this case.

---

## 5. Edge Direction Convention (NORMATIVE)

### 5.1 Parent = source, child = target

The stored `edge_index` has **source = parent (strict superset fragment), target = child (strict subset fragment)**.

This follows from the construction:

1. `_build_superset_matrix(formulas)` returns `M` where `M[i, j] = True` iff `formulas[i]` is a **strict superset** of `formulas[j]`. Therefore `i` is the **parent** (heavier / larger fragment) and `j` is the **child** (lighter / smaller fragment).

2. `_compute_edge_weights(superset_matrix)` produces `edge_weights[i, j]` = weight of edge from fragment `i` to fragment `j`. For each child `j`, the sum of incoming weights equals 1.0.

3. Storage-time COO extraction: `rows, cols = np.nonzero(edge_weights)` gives `rows` = parents, `cols` = children.

4. `edge_index = np.stack([rows, cols])` → **row 0 = source = parent, row 1 = target = child**.

### 5.2 Implication for PyG

PyTorch Geometric's default message passing direction is `flow="source_to_target"`, which propagates
messages **parent → child** (heavier → lighter). This matches the chemical intuition of fragmentation
cascading from precursor downward.

### 5.3 If your model needs child → parent flow

Flip the edge index:

```python
# In PyG Data object:
data.edge_index = data.edge_index[[1, 0]]  # swap source ↔ target
# Or equivalently:
data.edge_index = data.edge_index.flip(0)
```

The on-disk format is **always parent=source, child=target**. No `edge_direction` flag is stored.

---

## 6. Key Matching

### 6.1 TreeKey

A tree is uniquely identified by `(base_inchikey, ion_mode, precursor_type)`:

```python
@dataclass(frozen=True)
class TreeKey:
    base_inchikey: str   # 14-char InChIKey prefix
    ion_mode: str        # "P" or "N"
    precursor_type: str  # e.g. "[M+H]+", "[M-H]-"
```

### 6.2 Canonical ordering

When writing, trees are sorted by `(base_inchikey, ion_mode, precursor_type)` so that
both MS2 and full files built from the same spectral library naturally end up in the
same order, making positional pairing a valid fast path.

**However**, key-based pairing (via `align_keys`) should always be used in production.
Do not rely on positional equality alone.

### 6.3 Pairing semantics

```
MS2 file  ──►  dict[TreeKey, FragmentationTree] (MS2)
                       │
                       ├── align_keys() → list[(ms2_idx, full_idx)] pairs
                       │
Full file ──►  dict[TreeKey, FragmentationTree] (full)
```

Keys present in only one file are **mismatches**. By default, `align_keys` raises with a
clear message listing the mismatched keys (fail-fast). A config flag can downgrade to
warning + drop.

### 6.4 Critical distinction

The MS2 tree and full tree for the **same key** hold **different graphs**:
- MS2 tree: nodes from MS2 peaks only, edges from subformula relationships.
- Full tree: nodes from MS2+MSn peaks, edges from "lowest provable parent" rules.

Pairing is by key, NOT by node position within the graph.

---

## 7. PyTorch Geometric Mapping

### 7.1 Field mapping table

| NPZ field (sliced per-graph) | PyG `Data` field | Transform |
|------------------------------|------------------|-----------|
| `node_features` | `x` | `torch.from_numpy(...).to(dtype)` (default: `float32`) |
| `edge_index` | `edge_index` | `torch.from_numpy(...).to(torch.long)` — **mandatory** |
| `edge_weights` | `edge_attr` | `torch.from_numpy(...).reshape(-1, 1).to(torch.float32)` |
| `fragment_errors_ppm` | `node_error` (custom) | `torch.from_numpy(...).to(torch.float32)` (optional) |
| `precursor_indices[i]` | `precursor_idx` (custom) | `torch.tensor(int, dtype=torch.long)` — 0-D tensor |

### 7.2 Conversion example

```python
# Lazy import (torch not required to load NPZ files):
import torch
from torch_geometric.data import Data
from tree_storage import load_tree_arrays_npz

arrays = load_tree_arrays_npz("full_trees.npz")
data_list = []

for i in range(len(arrays.base_inchikey)):
    n_start = arrays.node_offsets[i]
    n_end = arrays.node_offsets[i + 1]
    e_start = arrays.edge_offsets[i]
    e_end = arrays.edge_offsets[i + 1]

    data = Data(
        x=torch.from_numpy(arrays.node_features[n_start:n_end].copy()).float(),
        edge_index=torch.from_numpy(arrays.edge_index[:, e_start:e_end].copy()).long(),
        edge_attr=torch.from_numpy(arrays.edge_weights[e_start:e_end].copy()).float().view(-1, 1),
    )
    data.precursor_idx = torch.tensor(int(arrays.precursor_indices[i]), dtype=torch.long)

    # If your model needs child→parent flow:
    # data.edge_index = data.edge_index[[1, 0]]

    data_list.append(data)

# For mini-batching:
from torch_geometric.data import Batch
batch = Batch.from_data_list(data_list)
# batch.precursor_idx shape: (batch_size,)
```

### 7.3 `precursor_idx` batching behavior

`precursor_idx` is stored as a **0-D tensor** (`torch.tensor(int, dtype=torch.long)`, shape `()`).
When `Batch.from_data_list` collates a list of `Data` objects:

- **Single `Data`:** `data.precursor_idx.shape == ()`
- **Batched `Batch`:** `batch.precursor_idx.shape == (batch_size,)`

Downstream code must handle both cases. This rank change is standard PyG behavior.

### 7.4 Edge direction reminder

The stored direction is **parent → child**. If your GNN message-passing model expects the
opposite (child → parent, common in molecule-to-fragment prediction), flip `edge_index`:
`data.edge_index = data.edge_index[[1, 0]]`.

---

## 8. Round-Trip Guarantees

The format guarantees **lossless** round-trip for all fields:

| Field | Round-trip |
|-------|-----------|
| `fragment_formulas` (node_features) | Exact int32 equality |
| `edge_weights` (via COO → dense reconstruction) | Exact float64 equality |
| `fragment_errors_ppm` | Exact float64 equality |
| `fragment_formulas_str` | Exact string equality (when stored) |
| `precursor_formula` | Exact int32 equality |
| `spectrum_*` fields (full trees) | Exact reconstruction |
| Trivial spectrum metadata (MS2 trees) | Synthesized: one spectrum, mslevel=2, msn_precursor=-1, all fragment indices |

### 8.1 MS2 trivial-metadata reconstruction

When loading an MS2 file (which omits `spectrum_*` fields), the loader synthesizes:
- `spectrum_fragments` = `[np.arange(n_fragments, dtype=np.int32)]` (one spectrum covering all nodes)
- `spectrum_mslevels` = `np.array([2], dtype=np.int32)`
- `spectrum_msn_precursors` = `[-1]`

This is valid because `build_ms2_trees` always produces exactly one spectrum per MS2 tree.

---

## 9. Versioning & Extension Policy

### 9.1 Current version

`format_version = 1`

### 9.2 Adding fields

To add optional fields in a future version:
1. Bump `format_version`.
2. Additive-only: new fields may be **appended**; do not remove or rename existing fields.
3. Readers should check `format_version` and gracefully degrade for unknown fields.

### 9.3 Breaking changes

If a breaking schema change is unavoidable (field removal, dtype change, CSR restructuring):
- Bump `format_version` to a new major number.
- Provide a v1→v2 migration script.

### 9.4 Replan triggers

The following situations require format changes (bump `format_version`):
- MS2 trees need > 1 spectrum per tree → store `spectrum_*` fields for MS2 files.
- A formula string exceeds 64 characters → bump `fragment_formulas_str` dtype width.
- A new per-graph field is needed (e.g., graph-level target `y`) → add as optional field.

---

## 10. Disk Efficiency

### 10.1 Compression

`np.savez_compressed` provides reasonable compression with minimal CPU overhead at this scale.

### 10.2 Sparse edges

The dense `(n, n)` edge weight matrix is stored as COO (sparse) — a `(n_nodes,)` tree with
`~O(n_nodes)` edges uses `O(n_edges)` storage instead of `O(n_nodes²)`.

### 10.3 `fragment_formulas_str`

Optional; when omitted, string formulas can be derived from `node_features` + element table.
Keeping them adds `<U64` bytes per node (~64 bytes) for debug/viz convenience.

### 10.4 Scale

Expected sizes (for reference):
- 1,000 trees × 200 nodes avg × 12 int32 formula elements ≈ 9.6 MB
- 100,000 trees × 200 nodes ≈ 960 MB
- Both fit comfortably in RAM on a modern workstation.

---

## 11. Element Table Reference

Formula arrays use **12 elements** in fixed order:

| Index | Symbol | Mass |
|-------|--------|------|
| 0 | H | 1.007825 |
| 1 | C | 12.000000 |
| 2 | N | 14.003074 |
| 3 | O | 15.994915 |
| 4 | F | 18.998403 |
| 5 | Na | 22.989770 |
| 6 | P | 30.973762 |
| 7 | S | 31.972071 |
| 8 | Cl | 34.968853 |
| 9 | K | 38.963707 |
| 10 | Br | 78.918338 |
| 11 | I | 126.904468 |

---

## 12. API Reference

### 12.1 Save

```python
from tree_storage import save_trees_npz, TreeStorageConfig
from ms2_tree_builder import build_ms2_trees
from fragmentation_tree import build_fragmentation_trees, FragmentationTreeConfig

config = FragmentationTreeConfig(merge_tolerance_ppm=5.0)

ms2_trees = build_ms2_trees(df, config)
save_trees_npz(ms2_trees, "ms2_trees.npz", TreeStorageConfig(tree_type="ms2"))

full_trees = build_fragmentation_trees(df, config)
save_trees_npz(full_trees, "full_trees.npz", TreeStorageConfig(tree_type="full"))
```

### 12.2 Load

```python
from tree_storage import load_trees_npz, load_tree_arrays_npz, align_keys

# Full reconstruction:
trees = load_trees_npz("full_trees.npz")

# Raw arrays (for PyG conversion):
arrays = load_tree_arrays_npz("full_trees.npz")

# Key matching:
ms2_keys = [TreeKey(b, i, p) for b, i, p in zip(
    ms2_arrays.base_inchikey, ms2_arrays.ion_mode, ms2_arrays.precursor_type
)]
full_keys = [TreeKey(b, i, p) for b, i, p in zip(
    full_arrays.base_inchikey, full_arrays.ion_mode, full_arrays.precursor_type
)]
pairs = align_keys(ms2_keys, full_keys)
```

### 12.3 End-to-end

```python
# read → build → save
from build_and_store_trees import main
main("cladribine.parquet", "output/")
# Produces: output/ms2_trees.npz, output/full_trees.npz
```
