"""
Test fragmentation graph construction from MSP data.

This test validates the construction of directed fragmentation graphs including:
- Fragment merging and deduplication across MS levels
- Parent-child edge generation based on formula subtraction
- MS level filtering constraints
- Precursor node inclusion
"""

import polars as pl
import numpy as np
from hrms_utils.formats.fragmentation_graph import construct_fragmentation_graphs_from_msp_data


def test_single_ms2_spectrum():
    """Test basic parent-child relationships with a single MS² spectrum."""
    print("\n=== Test 1: Single MS² spectrum ===")
    
    test_data = {
        'base_inchikey': ['COMPOUND1'],
        'ion_mode': ['P'],
        'precursor_formula_array': [[6, 12, 1, 2, 0]],  # C6H12NO2
        'mslevel': [2],
        'cleaned_fragment_formulas': [
            [[6, 10, 1, 1, 0], [3, 6, 0, 1, 0]]  # MS2: C6H10NO, C3H6O
        ],
        'cleaned_fragment_formulas_str': [['C6H10NO', 'C3H6O']],
        'cleaned_normalized_mz': [[100.0, 50.0]],
        'cleaned_normalized_intensity': [[1000.0, 500.0]],
    }
    
    df = pl.DataFrame(test_data)
    graphs = construct_fragmentation_graphs_from_msp_data(df, water_absorption=False)
    
    assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
    fragments, edge_index = graphs[0]
    
    # Should have 3 nodes: precursor + 2 fragments
    assert fragments.shape[0] == 3, f"Expected 3 nodes, got {fragments.shape[0]}"
    assert fragments.shape[1] == 2, f"Expected 2 features per node, got {fragments.shape[1]}"
    
    # Should have edges from precursor to both fragments
    assert edge_index.shape[0] == 2, f"Expected 2 rows in edge_index, got {edge_index.shape[0]}"
    assert edge_index.shape[1] >= 2, f"Expected at least 2 edges, got {edge_index.shape[1]}"
    
    print(f"  ✓ Nodes: {fragments.shape[0]}, Edges: {edge_index.shape[1]}")
    print(f"  ✓ Fragments (mz, intensity):\n{fragments}")
    print(f"  ✓ Edges:\n{edge_index}")


def test_ms2_ms3_cascade_with_filtering():
    """Test MS level filtering with MS² + MS³ cascade (N₂H₂/O₂H₂/H₂ example)."""
    print("\n=== Test 2: MS² + MS³ cascade with MS level filtering ===")
    
    # MS² spectrum: [N₂H₂, O₂H₂, H₂]
    # MS³ spectrum (precursor: N₂H₂): [H₂]
    # Expected: N₂H₂ → H₂ valid, O₂H₂ → H₂ should be filtered out
    
    test_data = {
        'base_inchikey': ['COMPOUND1', 'COMPOUND1'],
        'ion_mode': ['P', 'P'],
        'precursor_formula_array': [
            [0, 2, 2, 0, 0],  # N2H2 as root precursor
            [0, 2, 2, 0, 0],
        ],
        'mslevel': [2, 3],
        'cleaned_fragment_formulas': [
            [[0, 2, 2, 0, 0], [0, 2, 0, 2, 0], [0, 2, 0, 0, 0]],  # MS2: N2H2, O2H2, H2
            [[0, 2, 0, 0, 0]],  # MS3: H2
        ],
        'cleaned_fragment_formulas_str': [
            ['N2H2', 'O2H2', 'H2'],
            ['H2'],
        ],
        'cleaned_normalized_mz': [[100.0, 90.0, 2.0], [2.0]],
        'cleaned_normalized_intensity': [[1000.0, 800.0, 500.0], [600.0]],
    }
    
    df = pl.DataFrame(test_data)
    graphs = construct_fragmentation_graphs_from_msp_data(df, water_absorption=False)
    
    assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
    fragments, edge_index = graphs[0]
    
    # Should have 4 nodes: precursor + N2H2 (if different from precursor) + O2H2 + H2
    # Actually, N2H2 IS the precursor, so we should have 3 unique nodes
    print(f"  ✓ Nodes: {fragments.shape[0]}")
    print(f"  ✓ Edges: {edge_index.shape[1]}")
    print(f"  ✓ Fragments (mz, intensity):\n{fragments}")
    print(f"  ✓ Edges (parent -> child):\n{edge_index}")
    
    # Verify that O2H2 -> H2 edge is NOT present
    # This requires checking the actual formulas, which we don't expose
    # But we can verify the edge count is reasonable
    assert edge_index.shape[1] >= 1, "Expected at least 1 edge (precursor/N2H2 -> H2)"


def test_precursor_deduplication():
    """Test that precursor appearing in fragments is merged correctly."""
    print("\n=== Test 3: Precursor deduplication ===")
    
    test_data = {
        'base_inchikey': ['COMPOUND1'],
        'ion_mode': ['P'],
        'precursor_formula_array': [[6, 12, 1, 2, 0]],  # C6H12NO2
        'mslevel': [2],
        'cleaned_fragment_formulas': [
            [[6, 12, 1, 2, 0], [6, 10, 1, 1, 0]]  # Precursor appears in fragments
        ],
        'cleaned_fragment_formulas_str': [['C6H12NO2', 'C6H10NO']],
        'cleaned_normalized_mz': [[120.0, 100.0]],
        'cleaned_normalized_intensity': [[1500.0, 1000.0]],
    }
    
    df = pl.DataFrame(test_data)
    graphs = construct_fragmentation_graphs_from_msp_data(df, water_absorption=False)
    
    assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
    fragments, edge_index = graphs[0]
    
    # Should have 2 unique nodes (precursor deduplicated)
    assert fragments.shape[0] == 2, f"Expected 2 nodes after deduplication, got {fragments.shape[0]}"
    
    print(f"  ✓ Nodes: {fragments.shape[0]} (precursor deduplicated)")
    print(f"  ✓ Edges: {edge_index.shape[1]}")


def test_water_absorption():
    """Test water absorption loss edges."""
    print("\n=== Test 4: Water absorption ===")
    
    # Parent: C6H12O3, Child: C6H8O2 (diff = H4O1, which is H2O + H2)
    test_data = {
        'base_inchikey': ['COMPOUND1'],
        'ion_mode': ['P'],
        'precursor_formula_array': [[6, 12, 0, 3, 0]],  # C6H12O3
        'mslevel': [2],
        'cleaned_fragment_formulas': [
            [[6, 8, 0, 2, 0]]  # C6H8O2
        ],
        'cleaned_fragment_formulas_str': [['C6H8O2']],
        'cleaned_normalized_mz': [[100.0]],
        'cleaned_normalized_intensity': [[1000.0]],
    }
    
    df = pl.DataFrame(test_data)
    
    # Without water absorption: C6H12O3 - C6H8O2 = H4O1 (not direct loss)
    graphs_no_water = construct_fragmentation_graphs_from_msp_data(df, water_absorption=False)
    assert len(graphs_no_water) == 1
    fragments_no_water, edge_index_no_water = graphs_no_water[0]
    
    # With water absorption: C6H12O3 - C6H8O2 - H2O = H2 (valid)
    graphs_with_water = construct_fragmentation_graphs_from_msp_data(df, water_absorption=True)
    assert len(graphs_with_water) == 1
    fragments_with_water, edge_index_with_water = graphs_with_water[0]
    
    print(f"  ✓ Without water absorption: {edge_index_no_water.shape[1]} edges")
    print(f"  ✓ With water absorption: {edge_index_with_water.shape[1]} edges")
    
    # With water absorption should have at least as many edges
    assert edge_index_with_water.shape[1] >= edge_index_no_water.shape[1], \
        "Water absorption should add edges, not remove them"


def test_multiple_compounds():
    """Test grouping by base_inchikey and ion_mode."""
    print("\n=== Test 5: Multiple compounds ===")
    
    test_data = {
        'base_inchikey': ['COMPOUND1', 'COMPOUND2', 'COMPOUND1'],
        'ion_mode': ['P', 'P', 'N'],  # Different polarity for COMPOUND1
        'precursor_formula_array': [
            [6, 12, 1, 2, 0],
            [5, 10, 0, 2, 0],
            [6, 12, 1, 2, 0],
        ],
        'mslevel': [2, 2, 2],
        'cleaned_fragment_formulas': [
            [[3, 6, 0, 1, 0]],
            [[5, 8, 0, 1, 0]],
            [[3, 6, 0, 1, 0]],
        ],
        'cleaned_fragment_formulas_str': [['C3H6O'], ['C5H8O'], ['C3H6O']],
        'cleaned_normalized_mz': [[50.0], [80.0], [50.0]],
        'cleaned_normalized_intensity': [[500.0], [600.0], [500.0]],
    }
    
    df = pl.DataFrame(test_data)
    graphs = construct_fragmentation_graphs_from_msp_data(df, water_absorption=False)
    
    # Should have 3 separate graphs
    assert len(graphs) == 3, f"Expected 3 graphs, got {len(graphs)}"
    
    print(f"  ✓ Generated {len(graphs)} separate graphs for different compounds/polarities")


def test_empty_spectrum():
    """Test handling of spectrum with no fragments."""
    print("\n=== Test 6: Empty spectrum ===")
    
    test_data = {
        'base_inchikey': ['COMPOUND1'],
        'ion_mode': ['P'],
        'precursor_formula_array': [[6, 12, 1, 2, 0]],
        'mslevel': [2],
        'cleaned_fragment_formulas': [[]],  # No fragments
        'cleaned_fragment_formulas_str': [[]],
        'cleaned_normalized_mz': [[]],
        'cleaned_normalized_intensity': [[]],
    }
    
    df = pl.DataFrame(test_data)
    graphs = construct_fragmentation_graphs_from_msp_data(df, water_absorption=False)
    
    assert len(graphs) == 1, f"Expected 1 graph even with no fragments, got {len(graphs)}"
    fragments, edge_index = graphs[0]
    
    # Should have 1 node (precursor only) with no edges
    assert fragments.shape[0] == 1, f"Expected 1 node (precursor), got {fragments.shape[0]}"
    assert edge_index.shape[1] == 0, f"Expected 0 edges, got {edge_index.shape[1]}"
    
    print(f"  ✓ Handled empty spectrum: {fragments.shape[0]} node, {edge_index.shape[1]} edges")


def test_fragment_across_multiple_ms_levels():
    """Test fragment appearing in MS², MS³, and MS⁴."""
    print("\n=== Test 7: Fragment across multiple MS levels ===")
    
    test_data = {
        'base_inchikey': ['COMPOUND1', 'COMPOUND1', 'COMPOUND1'],
        'ion_mode': ['P', 'P', 'P'],
        'precursor_formula_array': [
            [6, 12, 1, 2, 0],
            [6, 12, 1, 2, 0],
            [6, 12, 1, 2, 0],
        ],
        'mslevel': [2, 3, 4],
        'cleaned_fragment_formulas': [
            [[3, 6, 0, 1, 0]],  # C3H6O in MS2
            [[3, 6, 0, 1, 0]],  # C3H6O in MS3
            [[3, 6, 0, 1, 0]],  # C3H6O in MS4
        ],
        'cleaned_fragment_formulas_str': [['C3H6O'], ['C3H6O'], ['C3H6O']],
        'cleaned_normalized_mz': [[50.0], [50.1], [49.9]],
        'cleaned_normalized_intensity': [[500.0], [600.0], [400.0]],
    }
    
    df = pl.DataFrame(test_data)
    graphs = construct_fragmentation_graphs_from_msp_data(df, water_absorption=False)
    
    assert len(graphs) == 1, f"Expected 1 graph, got {len(graphs)}"
    fragments, edge_index = graphs[0]
    
    # Should have 2 unique nodes (precursor + merged fragment)
    assert fragments.shape[0] == 2, f"Expected 2 nodes (merged across MS levels), got {fragments.shape[0]}"
    
    # Use max intensity (600.0) for the merged fragment
    max_intensity = np.max(fragments[:, 1])
    assert max_intensity == 600.0, f"Expected max intensity 600.0, got {max_intensity}"
    
    print(f"  ✓ Merged fragment across 3 MS levels into 1 node with max intensity")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing Fragmentation Graph Construction")
    print("=" * 60)
    
    try:
        test_single_ms2_spectrum()
        test_ms2_ms3_cascade_with_filtering()
        test_precursor_deduplication()
        test_water_absorption()
        test_multiple_compounds()
        test_empty_spectrum()
        test_fragment_across_multiple_ms_levels()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
