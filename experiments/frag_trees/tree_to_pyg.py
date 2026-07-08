"""
PyTorch Geometric converter for fragmentation tree NPZ files.

Converts the raw collated arrays (TreeArrays) into a list of
torch_geometric.data.Data objects suitable for GNN training.

torch and torch_geometric are imported LAZILY inside load_trees_as_pyg()
so that tree_storage.py and the rest of the storage pipeline remain
PyTorch-free. This module cannot be executed until torch + torch_geometric
are added to the project dependencies; an ImportError from the lazy imports
is the expected failure mode.

Edge direction:
    edge_index is passed through unchanged from the NPZ file, where
    source = parent (heavier fragment, strict superset) and target = child
    (lighter fragment). PyG message passing with default
    flow="source_to_target" therefore propagates parent -> child, matching
    the chemical intuition of fragmentation cascading from precursor
    downward.
    Callers needing child -> parent flow must flip edge_index:
        data.edge_index = data.edge_index[[1, 0]]

precursor_idx batching behavior:
    Each Data carries precursor_idx as a 0-D tensor (shape ()). When
    torch_geometric.data.Batch.from_data_list collates a list of these
    Data objects, it stacks the 0-D tensors into a 1-D tensor of shape
    (batch_size,) — one precursor index per graph in the batch, in graph
    order. Downstream code must therefore expect:
        - single (unbatched) Data:  data.precursor_idx.shape == ()
        - batched Batch:            batch.precursor_idx.shape == (batch_size,)
    This rank change is standard PyG behavior but is documented here
    because the per-graph and batched objects present different tensor
    ranks for the same logical field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiments.frag_trees.tree_storage import TreeArrays, load_tree_arrays_npz


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _slice_graph(arrays: TreeArrays, graph_index: int) -> dict[str, np.ndarray]:
    """Extract per-graph arrays via CSR offsets (pure NumPy).

    Uses ``node_offsets`` and ``edge_offsets`` to slice the flat collated
    arrays into per-graph views. All returned arrays are **copies**
    (``np.load`` returns read-only memory-mapped views; we copy to get
    writable, contiguous NumPy arrays for safe torch conversion).

    Args:
        arrays: Loaded TreeArrays instance.
        graph_index: 0-based index of the tree to extract.

    Returns:
        Dict with keys:
            - ``node_features``:   shape ``(n_nodes, 12)``, int32
            - ``edge_index``:      shape ``(2, n_edges)``, int32
            - ``edge_weights``:    shape ``(n_edges,)``, float64
            - ``fragment_errors_ppm``: shape ``(n_nodes,)``, float64
            - ``precursor_idx``:   scalar int32
    """
    node_start = int(arrays.node_offsets[graph_index])
    node_end = int(arrays.node_offsets[graph_index + 1])
    edge_start = int(arrays.edge_offsets[graph_index])
    edge_end = int(arrays.edge_offsets[graph_index + 1])

    return {
        "node_features": arrays.node_features[node_start:node_end].copy(),
        "edge_index": arrays.edge_index[:, edge_start:edge_end].copy(),
        "edge_weights": arrays.edge_weights[edge_start:edge_end].copy(),
        "fragment_errors_ppm": arrays.fragment_errors_ppm[node_start:node_end].copy(),
        "precursor_idx": arrays.precursor_indices[graph_index].copy(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_trees_as_pyg(
    npz_path: str | Path,
    x_dtype: str = "float32",
    carry_node_error: bool = False,
    carry_precursor_idx: bool = True,
) -> list[Any]:
    """Load an NPZ tree file and return a list of ``torch_geometric.data.Data``.

    ``torch`` and ``torch_geometric`` are imported **lazily** inside this
    function so that the module can be imported without those dependencies.
    The function will raise ``ImportError`` if torch or torch_geometric are
    not installed.

    Edge direction:
        ``edge_index`` is passed through unchanged (parent=source,
        child=target).  See module-level docstring for details and the
        flip instruction.

    ``precursor_idx`` batching:
        Each ``Data`` carries ``precursor_idx`` as a 0-D tensor
        (``shape == ()``).  After ``Batch.from_data_list`` this becomes a
        1-D tensor of shape ``(batch_size,)``.  See module-level docstring.

    Args:
        npz_path: Path to the ``.npz`` file written by
            :func:`tree_storage.save_trees_npz`.
        x_dtype: Torch dtype string for node features, e.g. ``"float32"``,
            ``"float64"``, or ``"int32"``.  Passed to
            ``getattr(torch, x_dtype)``.
        carry_node_error: If ``True``, attach per-node annotation errors
            as ``Data.node_error`` (float32 tensor).  Default ``False``.
        carry_precursor_idx: If ``True`` (default), attach the precursor
            node index as ``Data.precursor_idx`` (0-D long tensor).

    Returns:
        List of ``torch_geometric.data.Data`` objects, one per tree in the
        NPZ file, preserving the file's key ordering.

    Raises:
        ImportError: If ``torch`` or ``torch_geometric`` are not installed.
    """
    # Lazy imports — torch/torch_geometric are NOT project dependencies yet.
    import torch
    from torch_geometric.data import Data

    arrays: TreeArrays = load_tree_arrays_npz(npz_path)
    n_trees: int = len(arrays.base_inchikey)

    torch_dtype = getattr(torch, x_dtype)

    out: list[Data] = []
    for i in range(n_trees):
        g: dict[str, np.ndarray] = _slice_graph(arrays, i)

        data = Data(
            x=torch.from_numpy(g["node_features"]).to(torch_dtype),
            edge_index=torch.from_numpy(g["edge_index"]).to(torch.long),
            edge_attr=torch.from_numpy(g["edge_weights"]).reshape(-1, 1).to(torch.float32),
        )

        if carry_node_error:
            data.node_error = torch.from_numpy(g["fragment_errors_ppm"]).to(torch.float32)

        if carry_precursor_idx:
            # 0-D tensor; becomes (batch_size,) 1-D after Batch.from_data_list
            data.precursor_idx = torch.tensor(int(g["precursor_idx"]), dtype=torch.long)

        out.append(data)

    return out
