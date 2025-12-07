"""
Fragmentation graph construction from MSP data.

This module constructs directed fragmentation graphs (parent → child) from MSP file parsing results.
The graphs represent fragmentation cascades where edges indicate possible parent-child relationships
based on formula subtraction rules and MS level constraints.

Why MS level filtering matters: In tandem MS experiments, MS³ fragments can only arise from the 
precursor ion selected from MS². A fragment observed in MS² but not selected as the MS³ precursor 
cannot directly produce MS³ fragments, even if the formulas suggest a valid subtraction relationship.

Why merge fragments across MS levels: A fragment with formula CH₄ appearing in both MS² and MS³ is 
the same chemical entity. Its appearance in both levels provides information about the fragmentation 
cascade that helps deduce true parent-child relationships.
"""

import polars as pl
import numpy as np
from numpy.typing import NDArray
from typing import Literal


def construct_fragmentation_graphs_from_msp_data(
    dataframe: pl.DataFrame,
    water_absorption: bool = False,
    element_array_width: int | None = None,
) -> list[tuple[NDArray[np.float32], NDArray[np.int64]]]:
    """
    Construct fragmentation graphs from MSP parsing results.
    
    This function takes the output of read_MSPEC_file() and constructs directed fragmentation graphs
    for each compound-polarity-precursor combination. Each graph consists of:
    - node features: merged and deduplicated fragments as a 2D array (num_nodes, 2) with [mz, intensity]
    - edge_index: directed edges as a 2D array (2, num_edges) with [source_idx, target_idx]
    
    The function groups spectra by compound and ion mode, merges fragments across MS levels,
    and constructs parent-child relationships based on formula subtraction rules.
    
    Args:
        dataframe: Output from read_MSPEC_file() containing spectrum data
        water_absorption: If True, allow fragment relationships with H₂O loss
        element_array_width: Number of elements in formula arrays (auto-detected if None)
    
    Returns:
        List of tuples, each containing:
            - fragments: NDArray[np.float32] of shape (num_nodes, 2) with [mz, intensity] for each fragment
            - edge_index: NDArray[np.int64] of shape (2, num_edges) with [source_node_idx, target_node_idx]
    
    Raises:
        AssertionError: If required columns are missing or formula arrays have inconsistent shapes
    
    Notes:
        - Fragments are identified solely by their formula (element counts)
        - The same fragment appearing in multiple MS levels is treated as a single node
        - Parent-child edges are filtered based on MS level constraints
        - The precursor is included as the root node of each graph
    """
    # Input validation
    required_columns = [
        "base_inchikey", 
        "ion_mode", 
        "precursor_formula_array",
        "mslevel",
        "cleaned_fragment_formulas",
        "cleaned_fragment_formulas_str",
        "cleaned_normalized_mz",
        "cleaned_normalized_intensity",
    ]
    missing = set(required_columns) - set(dataframe.columns)
    assert not missing, (
        f"Missing required columns for fragmentation graph construction: {missing}. "
        f"Expected columns: {required_columns}"
    )
    
    # Auto-detect element array width if not provided
    if element_array_width is None:
        first_precursor = dataframe.select("precursor_formula_array").head(1).item()
        element_array_width = len(first_precursor)
    
    # Group by compound and ion mode
    grouped = dataframe.group_by(["base_inchikey", "ion_mode"], maintain_order=True)
    
    graphs = []
    
    for (base_inchikey, ion_mode), group_df in grouped:
        # Further group by precursor formula to handle multiple precursors per compound
        precursor_groups = group_df.group_by("precursor_formula_array", maintain_order=True)
        
        for precursor_formula_tuple, spectra_df in precursor_groups:
            # Convert precursor formula tuple to numpy array
            precursor_formula = np.array(precursor_formula_tuple, dtype=np.int32)
            
            # Extract and merge all fragments across MS levels
            fragments_array, edge_index = _construct_single_graph(
                spectra_df=spectra_df,
                precursor_formula=precursor_formula,
                water_absorption=water_absorption,
                element_array_width=element_array_width,
            )
            
            if fragments_array is not None and edge_index is not None:
                graphs.append((fragments_array, edge_index))
    
    return graphs


def _construct_single_graph(
    spectra_df: pl.DataFrame,
    precursor_formula: NDArray[np.int32],
    water_absorption: bool,
    element_array_width: int,
) -> tuple[NDArray[np.float32] | None, NDArray[np.int64] | None]:
    """
    Construct a single fragmentation graph for one precursor.
    
    Args:
        spectra_df: DataFrame containing all spectra for a single precursor
        precursor_formula: Formula array for the precursor
        water_absorption: If True, allow H₂O loss relationships
        element_array_width: Width of formula arrays
    
    Returns:
        Tuple of (fragments_array, edge_index) or (None, None) if no valid graph
    """
    # Collect all fragments with their MS levels and spectral data
    fragment_data = _collect_and_deduplicate_fragments(
        spectra_df=spectra_df,
        precursor_formula=precursor_formula,
        element_array_width=element_array_width,
    )
    
    if not fragment_data:
        return None, None
    
    # fragment_data is a dict: formula_tuple -> {mslevels: set, mz: float, intensity: float}
    
    # Create node features (fragments array) - shape (num_nodes, 2) with [mz, intensity]
    fragment_formulas = []
    fragment_mslevels = []
    mz_values = []
    intensity_values = []
    
    for formula_tuple, data in fragment_data.items():
        fragment_formulas.append(np.array(formula_tuple, dtype=np.int32))
        fragment_mslevels.append(data['mslevels'])
        mz_values.append(data['mz'])
        intensity_values.append(data['intensity'])
    
    fragments_array = np.column_stack([
        np.array(mz_values, dtype=np.float32),
        np.array(intensity_values, dtype=np.float32)
    ])
    
    # Construct edges based on formula subtraction rules
    edge_list = _generate_parent_child_edges(
        fragment_formulas=fragment_formulas,
        fragment_mslevels=fragment_mslevels,
        water_absorption=water_absorption,
        element_array_width=element_array_width,
    )
    
    if not edge_list:
        # No edges - return just the nodes with empty edge_index
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(edge_list, dtype=np.int64).T  # Shape (2, num_edges)
    
    return fragments_array, edge_index


def _collect_and_deduplicate_fragments(
    spectra_df: pl.DataFrame,
    precursor_formula: NDArray[np.int32],
    element_array_width: int,
) -> dict[tuple, dict]:
    """
    Collect all fragments across MS levels and deduplicate by formula.
    
    Fragments are identified solely by their formula (element counts).
    The same fragment in multiple MS levels is treated as a single node.
    The precursor is added as the root node.
    
    Args:
        spectra_df: DataFrame containing spectra for a single precursor
        precursor_formula: Formula array for the precursor
        element_array_width: Width of formula arrays
    
    Returns:
        Dictionary mapping formula_tuple -> {mslevels: set, mz: float, intensity: float}
    """
    fragment_data = {}
    
    # Add precursor as root node (MS level 1)
    precursor_tuple = tuple(precursor_formula)
    fragment_data[precursor_tuple] = {
        'mslevels': {1},  # Precursor is at MS level 1
        'mz': 0.0,  # Placeholder, will be updated if precursor appears in fragments
        'intensity': 1.0,  # Placeholder
    }
    
    # Iterate through all spectra
    for row in spectra_df.iter_rows(named=True):
        mslevel = row['mslevel']
        fragment_formulas = row['cleaned_fragment_formulas']
        fragment_mz = row['cleaned_normalized_mz']
        fragment_intensity = row['cleaned_normalized_intensity']
        
        if fragment_formulas is None or len(fragment_formulas) == 0:
            continue
        
        # Process each fragment in this spectrum
        for i, formula_array in enumerate(fragment_formulas):
            formula_tuple = tuple(formula_array)
            
            # Get corresponding mz and intensity
            mz = fragment_mz[i] if i < len(fragment_mz) else 0.0
            intensity = fragment_intensity[i] if i < len(fragment_intensity) else 0.0
            
            if formula_tuple in fragment_data:
                # Fragment already seen - add this MS level and update mz/intensity if better
                fragment_data[formula_tuple]['mslevels'].add(mslevel)
                # Use max intensity if fragment appears multiple times
                if intensity > fragment_data[formula_tuple]['intensity']:
                    fragment_data[formula_tuple]['mz'] = mz
                    fragment_data[formula_tuple]['intensity'] = intensity
            else:
                # New fragment
                fragment_data[formula_tuple] = {
                    'mslevels': {mslevel},
                    'mz': mz,
                    'intensity': intensity,
                }
    
    return fragment_data


def _generate_parent_child_edges(
    fragment_formulas: list[NDArray[np.int32]],
    fragment_mslevels: list[set[int]],
    water_absorption: bool,
    element_array_width: int,
) -> list[tuple[int, int]]:
    """
    Generate parent-child edges based on formula subtraction rules.
    
    A directed edge parent → child exists if:
    1. Direct loss: parent_formula - child_formula ≥ 0 (element-wise) and non-zero
    2. Water absorption (if enabled): parent_formula - child_formula - H₂O ≥ 0 and non-zero
    
    MS level filtering: A fragment in MS^N can only have parents that:
    - Appear in the same MS^N spectrum, OR
    - Are precursors at level MS^(N-1)
    
    Args:
        fragment_formulas: List of formula arrays for each node
        fragment_mslevels: List of MS level sets for each node
        water_absorption: If True, allow H₂O loss relationships
        element_array_width: Width of formula arrays
    
    Returns:
        List of (parent_idx, child_idx) tuples
    """
    edges = []
    num_fragments = len(fragment_formulas)
    
    # Water molecule formula: H₂O = [C=0, H=2, N=0, O=1, ...]
    # Assuming element order is [C, H, N, O, ...] based on typical usage
    h2o_formula = np.zeros(element_array_width, dtype=np.int32)
    if element_array_width >= 4:
        h2o_formula[1] = 2  # H
        h2o_formula[3] = 1  # O
    
    for parent_idx in range(num_fragments):
        parent_formula = fragment_formulas[parent_idx]
        parent_mslevels = fragment_mslevels[parent_idx]
        
        for child_idx in range(num_fragments):
            if parent_idx == child_idx:
                continue
            
            child_formula = fragment_formulas[child_idx]
            child_mslevels = fragment_mslevels[child_idx]
            
            # Check MS level compatibility
            # Why: A fragment in MS^N can only have parents from the same MS^N or precursors from MS^(N-1)
            if not _are_mslevels_compatible_for_edge(parent_mslevels, child_mslevels):
                continue
            
            # Check direct loss: parent - child ≥ 0 and non-zero
            loss = parent_formula - child_formula
            if np.all(loss >= 0) and np.any(loss > 0):
                edges.append((parent_idx, child_idx))
                continue
            
            # Check water absorption loss if enabled
            if water_absorption:
                loss_with_water = loss - h2o_formula
                if np.all(loss_with_water >= 0) and np.any(loss_with_water > 0):
                    edges.append((parent_idx, child_idx))
    
    return edges


def _are_mslevels_compatible_for_edge(
    parent_mslevels: set[int],
    child_mslevels: set[int],
) -> bool:
    """
    Check if parent and child MS levels are compatible for an edge.
    
    Rule: A fragment in MS^N can only have parents that:
    - Appear in the same MS^N spectrum (overlap in MS levels), OR
    - Are precursors (MS level 1 for the precursor node)
    
    Args:
        parent_mslevels: Set of MS levels where parent was observed
        child_mslevels: Set of MS levels where child was observed
    
    Returns:
        True if edge is valid based on MS level constraints
    """
    # If parent is the precursor (MS level 1), it can be parent to any fragment
    if 1 in parent_mslevels:
        return True
    
    # Otherwise, parent and child must share at least one MS level
    # Why: Fragments can only arise from co-occurring species in the same MS experiment
    return bool(parent_mslevels & child_mslevels)
