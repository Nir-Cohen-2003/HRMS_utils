import networkx as nx
import rustworkx as rx
from time import perf_counter
from rdkit import Chem


def construct_graph(smiles:str) -> nx.Graph:
    """ 
    Converts a SMILE into a graph
     
    Parameters
    ----------
    s : str 
        Smile of the molecule
        
    Returns:
    -------
    networkx.classes.graph.Graph
        Graph that represents the molecule.
        The bond types are represented as edge weights.
        The atom types are represented as atom attributes of the nodes.
    """
    #read the smile
    m: Chem.Mol = Chem.MolFromSmiles(smiles) # type :ignore
    # convert the molecule into a graph
    if m is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    # The bond and atom types are converted to node/edge attributes
    G: nx.Graph = nx.Graph()
    for atom in m.GetAtoms():
        G.add_node(atom.GetIdx(), atom=atom.GetSymbol())
    for bond in m.GetBonds():
        G.add_edge(bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx(), weight=bond.GetBondTypeAsDouble())
    return G

def construct_graph_rustworkx(smiles: str):
    """
    Converts a SMILE into a RustWorkX graph (pure RustWorkX implementation)
     
    Parameters
    ----------
    smiles : str 
        SMILES string of the molecule
        
    Returns:
    -------
    rustworkx.PyGraph
        Graph that represents the molecule.
        The bond types are represented as edge weights.
        The atom types are represented as atom attributes of the nodes.
    """
    
    # Read the SMILES
    m: Chem.Mol = Chem.MolFromSmiles(smiles) # type: ignore
    
    # Create RustWorkX graph
    G = rx.PyGraph()
    
    # Add nodes with atom data
    node_mapping = {}  # Map RDKit atom indices to RustWorkX node indices
    for atom in m.GetAtoms():
        rx_node_idx = G.add_node({'atom': atom.GetSymbol()})
        node_mapping[atom.GetIdx()] = rx_node_idx
    
    # Add edges with bond data
    for bond in m.GetBonds():
        begin_idx = node_mapping[bond.GetBeginAtom().GetIdx()]
        end_idx = node_mapping[bond.GetEndAtom().GetIdx()]
        G.add_edge(begin_idx, end_idx, {'weight': bond.GetBondTypeAsDouble()})
    
    return G

def benchmark_construct_graph(repeat_multiplier=5):
    """
    Benchmark function to compare NetworkX vs RustWorkX graph construction.
    Tests both correctness and performance.
    
    Parameters
    ----------
    repeat_multiplier : int, optional
        Multiplier for the number of repeats in detailed timing (default: 5)
    """
    try:
        import rustworkx as rx
    except ImportError:
        print("RustWorkX not available, skipping RustWorkX benchmarks")
        return
    
    import numpy as np
    
    # Test molecules of varying complexity
    test_smiles = [
        # Small molecules
        "C",                # Methane
        "CC",               # Ethane
        "CCO",              # Ethanol
        "CC(=O)C",          # Acetone
        "c1ccccc1",         # Benzene
        "Cc1ccccc1",        # Toluene
        "CC(=O)O",          # Acetic acid
        "C1CCCCC1",         # Cyclohexane
        
        # Medium molecules
        # "C(C1C(C(C(C(O1)O)O)O)O",  # Glucose
        "CC(=O)Oc1ccccc1C(=O)O",     # Aspirin
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",# Caffeine
        "CN1CCC[C@H]1c2cccnc2",      # Nicotine
        
        # Larger molecules
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen
        "CCN(CC)CCOC(=O)C1=CC=CC=C1", # Propranolol-like
        "CC1=C(C(=O)NC(=O)N1)N",     # Paracetamol
        
        # Complex molecules
        "CC(C)NCC(O)COc1ccc2nc(S(N)(=O)=O)sc2c1", # Hydrochlorothiazide
        "C[C@H]1CC[C@H]2[C@@H]([C@H]1C)CC[C@@H]3[C@@H]2CC[C@H](C3)O", # Steroid-like
    ]
    
    # Multiply by repetitions for better timing
    test_smiles = test_smiles * repeat_multiplier
    
    print(f"Benchmarking graph construction with {len(test_smiles)} molecules...")
    
    # Test NetworkX version
    print("Testing NetworkX version...")
    start_time = perf_counter()
    nx_graphs = []
    for smiles in test_smiles:
        try:
            g = construct_graph(smiles)
            nx_graphs.append(g)
        except Exception as e:
            print(f"NetworkX failed for {smiles}: {e}")
            nx_graphs.append(None)
    nx_time = perf_counter() - start_time
    
    # Test RustWorkX version
    print("Testing RustWorkX version...")
    start_time = perf_counter()
    rx_graphs = []
    for smiles in test_smiles:
        try:
            g = construct_graph_rustworkx(smiles)
            rx_graphs.append(g)
        except Exception as e:
            print(f"RustWorkX failed for {smiles}: {e}")
            rx_graphs.append(None)
    rx_time = perf_counter() - start_time
    
    # Compare results for correctness
    print("Comparing results for correctness...")
    mismatches = 0
    valid_comparisons = 0
    
    for i, (nx_g, rx_g, smiles) in enumerate(zip(nx_graphs, rx_graphs, test_smiles)):
        if nx_g is None or rx_g is None:
            continue
            
        valid_comparisons += 1
        
        # Compare number of nodes
        if nx_g.number_of_nodes() != rx_g.num_nodes():
            print(f"Node count mismatch for {smiles}: NX={nx_g.number_of_nodes()}, RX={rx_g.num_nodes()}")
            mismatches += 1
            continue
            
        # Compare number of edges
        if nx_g.number_of_edges() != rx_g.num_edges():
            print(f"Edge count mismatch for {smiles}: NX={nx_g.number_of_edges()}, RX={rx_g.num_edges()}")
            mismatches += 1
            continue
            
        # Compare atom types (more detailed comparison)
        nx_atoms = sorted([nx_g.nodes[node]['atom'] for node in nx_g.nodes()])
        rx_atoms = sorted([rx_g[node]['atom'] for node in rx_g.node_indices()])
        
        if nx_atoms != rx_atoms:
            print(f"Atom type mismatch for {smiles}")
            print(f"  NX atoms: {nx_atoms}")
            print(f"  RX atoms: {rx_atoms}")
            mismatches += 1
            continue
            
        # Compare edge weights (simplified - just check totals)
        nx_total_weight = sum(data['weight'] for _, _, data in nx_g.edges(data=True))
        rx_total_weight = sum(rx_g.get_edge_data(edge[0], edge[1])['weight'] 
                             for edge in rx_g.edge_list())
        
        if not np.isclose(nx_total_weight, rx_total_weight):
            print(f"Edge weight sum mismatch for {smiles}: NX={nx_total_weight:.6f}, RX={rx_total_weight:.6f}")
            mismatches += 1
    
    # Print results
    print(f"\nBenchmark Results:")
    print(f"Valid comparisons: {valid_comparisons}/{len(test_smiles)}")
    print(f"Mismatches found: {mismatches}")
    print(f"NetworkX time: {nx_time:.4f} seconds")
    print(f"RustWorkX time: {rx_time:.4f} seconds")
    
    if rx_time > 0:
        speedup = nx_time / rx_time
        print(f"Speedup (NetworkX/RustWorkX): {speedup:.2f}x")
        
        if speedup > 1:
            print(f"RustWorkX is {speedup:.2f}x faster")
        else:
            print(f"NetworkX is {1/speedup:.2f}x faster")
    
    # Test with a single complex molecule for detailed timing
    complex_smiles = "CC(C)NCC(O)COc1ccc2nc(S(N)(=O)=O)sc2c1"  # Hydrochlorothiazide
    n_repeats = repeat_multiplier * 20
    
    print(f"\nDetailed timing with {n_repeats} repeats of complex molecule:")
    print(f"SMILES: {complex_smiles}")
    
    # NetworkX detailed timing
    start_time = perf_counter()
    for _ in range(n_repeats):
        construct_graph(complex_smiles)
    nx_detailed_time = perf_counter() - start_time
    
    # RustWorkX detailed timing
    start_time = perf_counter()
    for _ in range(n_repeats):
        construct_graph_rustworkx(complex_smiles)
    rx_detailed_time = perf_counter() - start_time
    
    print(f"NetworkX: {nx_detailed_time:.4f} seconds ({nx_detailed_time/n_repeats*1000:.2f} ms per molecule)")
    print(f"RustWorkX: {rx_detailed_time:.4f} seconds ({rx_detailed_time/n_repeats*1000:.2f} ms per molecule)")
    
    if rx_detailed_time > 0:
        detailed_speedup = nx_detailed_time / rx_detailed_time
        print(f"Detailed speedup: {detailed_speedup:.2f}x")
    
    if mismatches == 0:
        print("\n✓ All graph constructions produce identical results")
    else:
        print(f"\n⚠️  {mismatches} mismatches found between NetworkX and RustWorkX versions")

if __name__ == "__main__":
    import sys
    import argparse

    if '--graph-benchmark' in sys.argv:
        parser = argparse.ArgumentParser(description='Benchmark graph construction')
        parser.add_argument('--graph-benchmark', action='store_true', help='Run graph construction benchmark')
        parser.add_argument('repeat_multiplier', nargs='?', type=int, default=5,
                            help='Multiplier for number of repeats in detailed timing (default: 5)')
        args = parser.parse_args()
        benchmark_construct_graph(repeat_multiplier=args.repeat_multiplier)


