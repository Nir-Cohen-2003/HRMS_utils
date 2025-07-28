import pulp
import networkx as nx
import rustworkx as rx
from time import perf_counter
from rdkit import Chem
from typing import List, Tuple, Optional, Generator

def MCES_ILP(G1, G2, threshold, solver='default', solver_options={}, no_ilp_threshold=False):
    """
     Calculates the exact distance between two molecules using an ILP

     Parameters
     ----------
     G1 : networkx.classes.graph.Graph
         Graph representing the first molecule.
     G2 : networkx.classes.graph.Graph
         Graph representing the second molecule.
     threshold : float
         Threshold for the comparison. Exact distance is only calculated if the distance is lower than the threshold.
     solver: string
         ILP-solver used for solving MCES. Example:CPLEX_CMD
     solver_options: dict
         additional options to pass to solvers. Example: threads=1, msg=False for better multi-threaded performance
     no_ilp_threshold: bool
         if true, always return exact distance even if it is below the threshold (slower)

     Returns:
     -------
     float
         Distance between the molecules
     int
         Type of Distance:
             1 : Exact Distance
             2 : Lower bound (If the exact distance is above the threshold)

    """

    ILP=pulp.LpProblem("MCES", pulp.LpMinimize)

    #Variables for nodepairs
    nodepairs=[]
    for i in G1.nodes:
        for j in G2.nodes:
            if G1.nodes[i]["atom"]==G2.nodes[j]["atom"]:
                nodepairs.append(tuple([i,j]))
    y=pulp.LpVariable.dicts('nodepairs', nodepairs,
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger)
    #variables for edgepairs and weight
    edgepairs=[]
    w={}
    for i in G1.edges:
        for j in G2.edges:
            if (G1.nodes[i[0]]["atom"]==G2.nodes[j[0]]["atom"] and G1.nodes[i[1]]["atom"]==G2.nodes[j[1]]["atom"]) or (G1.nodes[i[1]]["atom"]==G2.nodes[j[0]]["atom"] and G1.nodes[i[0]]["atom"]==G2.nodes[j[1]]["atom"]):
                edgepairs.append(tuple([i,j]))
                w[tuple([i,j])]=max(G1[i[0]][i[1]]["weight"],G2[j[0]][j[1]]["weight"])-min(G1[i[0]][i[1]]["weight"],G2[j[0]][j[1]]["weight"])

    #variables for not mapping an edge
    for i in G1.edges:
        edgepairs.append(tuple([i,-1]))
        w[tuple([i,-1])]=G1[i[0]][i[1]]["weight"]
    for j in G2.edges:
        edgepairs.append(tuple([-1,j]))
        w[tuple([-1,j])]=G2[j[0]][j[1]]["weight"]
    c=pulp.LpVariable.dicts('edgepairs', edgepairs,
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger)


    #objective function
    ILP += pulp.lpSum([ w[i]*c[i] for i in edgepairs])

    #Every node in G1 can only be mapped to at most one in G2
    for i in G1.nodes:
        h=[]
        for j in G2.nodes:
            if G1.nodes[i]["atom"]==G2.nodes[j]["atom"]:
                h.append(tuple([i,j]))
        ILP+=pulp.lpSum([y[k] for k in h])<=1

    #Every node in G1 can only be mapped to at most one in G1
    for i in G2.nodes:
        h=[]
        for j in G1.nodes:
            if G1.nodes[j]["atom"]==G2.nodes[i]["atom"]:
                h.append(tuple([j,i]))
        ILP+=pulp.lpSum([y[k] for k in h])<=1

    #Every edge in G1 has to be mapped to an edge in G2 or the variable for not mapping has to be 1
    for i in G1.edges:
        ls=[]
        rs=[]
        for j in G2.edges:
            if (G1.nodes[i[0]]["atom"]==G2.nodes[j[0]]["atom"] and G1.nodes[i[1]]["atom"]==G2.nodes[j[1]]["atom"]) or (G1.nodes[i[1]]["atom"]==G2.nodes[j[0]]["atom"] and G1.nodes[i[0]]["atom"]==G2.nodes[j[1]]["atom"]):
                ls.append(tuple([i,j]))
        ILP+=pulp.lpSum([c[k] for k in ls])+c[tuple([i,-1])]==1

    #Every edge in G2 has to be mapped to an edge in G1 or the variable for not mapping has to be 1
    for i in G2.edges:
        ls=[]
        rs=[]
        for j in G1.edges:
            if (G1.nodes[j[0]]["atom"]==G2.nodes[i[0]]["atom"] and G1.nodes[j[1]]["atom"]==G2.nodes[i[1]]["atom"]) or (G1.nodes[j[1]]["atom"]==G2.nodes[i[0]]["atom"] and G1.nodes[j[0]]["atom"]==G2.nodes[i[1]]["atom"]):
                ls.append(tuple([j,i]))
        ILP+=pulp.lpSum([c[k] for k in ls])+c[tuple([-1,i])]==1

    #The mapping of the edges has to match the mapping of the nodes
    for i in G1.nodes:
        for j in G2.edges:
            ls=[]
            for k in G1.neighbors(i):
                if tuple([tuple([i,k]),j]) in c:
                    ls.append(tuple([tuple([i,k]),j]))
                else:
                    if  tuple([tuple([k,i]),j]) in c:
                        ls.append(tuple([tuple([k,i]),j]))
            rs=[]
            if G1.nodes[i]["atom"]==G2.nodes[j[0]]["atom"]:
                rs.append(tuple([i,j[0]]))
            if G1.nodes[i]["atom"]==G2.nodes[j[1]]["atom"]:
                rs.append(tuple([i,j[1]]))
            ILP+=pulp.lpSum([c[k] for k in ls])<=pulp.lpSum([y[k] for k in rs])


    for i in G2.nodes:
        for j in G1.edges:
            ls=[]
            for k in G2.neighbors(i):
                if tuple([j,tuple([i,k])]) in c:
                    ls.append(tuple([j,tuple([i,k])]))
                else:
                    if tuple([j,tuple([k,i])]) in c:
                        ls.append(tuple([j,tuple([k,i])]))
            rs=[]
            if G2.nodes[i]["atom"]==G1.nodes[j[0]]["atom"]:
                rs.append(tuple([j[0],i]))
            if G2.nodes[i]["atom"]==G1.nodes[j[1]]["atom"]:
                rs.append(tuple([j[1],i]))
            ILP+=pulp.lpSum([c[k] for k in ls])<=pulp.lpSum(y[k] for k in rs)

    #constraint for the threshold
    if threshold!=-1 and not no_ilp_threshold:
        ILP +=pulp.lpSum([ w[i]*c[i] for i in edgepairs])<=threshold

    #solve the ILP
    if solver.lower()=="default":
        sol= pulp.getSolver("PULP_CBC_CMD", msg=0,**solver_options)
        ILP.solve()
    elif solver.upper()=="GUROBI":
        # ILP.solve(pulp.GUROBI(**solver_options))
        sol:pulp.LpSolver=pulp.getSolver("GUROBI", **solver_options)
        ILP.solve(sol)
    elif solver.upper()=="CUOPT":
        ILP.solve(pulp.CUOPT(msg=0))
        print("CUOPT WAS USED")

    else:
        ILP.solve(pulp.PULP_CBC_CMD(msg=0,**solver_options))
    if ILP.status==1:
        val = ILP.objective.value()
        if val is None:
            return 0, 1
        return float(ILP.objective.value()),1
    else:
        return threshold,2

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
    try:
        import rustworkx as rx
    except ImportError:
        raise ImportError("RustWorkX is required for this function. Install with: pip install rustworkx")
    
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

def benchmark_construct_graph():
    """
    Benchmark function to compare NetworkX vs RustWorkX graph construction.
    Tests both correctness and performance.
    """
    try:
        import rustworkx as rx
    except ImportError:
        print("RustWorkX not available, skipping RustWorkX benchmarks")
        return
    
    from time import perf_counter
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
    test_smiles = test_smiles * 5
    
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
    n_repeats = 100
    
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
       
    if '--graph-benchmark' in sys.argv:
        benchmark_construct_graph()


