from typing import List, Tuple, Optional, Generator
from scipy.optimize import linear_sum_assignment
import networkx as nx
import numpy as np
from collections import defaultdict

def filter1(G1: nx.Graph, G2: nx.Graph) -> float:
    """
     Finds a lower bound for the distance based on degree

     Parameters
     ----------
     G1 : networkx.classes.graph.Graph
         Graph representing the first molecule.
     G2 : networkx.classes.graph.Graph
         Graph representing the second molecule.

     Returns:
     -------
     float
         Lower bound for the distance between the molecules

    """
    #Find all occuring atom types and partition by type
    atom_types1=[]
    for i in G1.nodes:
        if G1.nodes[i]["atom"] not in atom_types1:
            atom_types1.append(G1.nodes[i]["atom"])
    type_map1={}
    for i in atom_types1:
        type_map1[i]=list(filter(lambda x: i==G1.nodes[x]["atom"],G1.nodes))

    atom_types2=[]
    for i in G2.nodes:
        if G2.nodes[i]["atom"] not in atom_types2:
            atom_types2.append(G2.nodes[i]["atom"])
    type_map2={}
    for i in atom_types2:
        type_map2[i]=list(filter(lambda x: i==G2.nodes[x]["atom"],G2.nodes))

    #calculate lower bound
    difference=0
    #Every atom type is done seperately
    for i in atom_types1:
        if i in atom_types2:
            #number of nodes that can be mapped
            n=min(len(type_map1[i]),len(type_map2[i]))
            #sort by degree
            degreelist1=sorted(type_map1[i],key=lambda x:sum([G1[x][j]["weight"] for j in G1.neighbors(x)]),reverse=True)
            degreelist2=sorted(type_map2[i],key=lambda x:sum([G2[x][j]["weight"] for j in G2.neighbors(x)]),reverse=True)
            #map in order of sorted lists
            for j in range(n):
                deg1=sum([G1[degreelist1[j]][k]["weight"] for k in G1.neighbors(degreelist1[j])])
                deg2=sum([G2[degreelist2[j]][k]["weight"] for k in G2.neighbors(degreelist2[j])])
                difference+= abs(deg1-deg2)
            #nodes that are not mapped
            if len(degreelist1)>n:
                for j in range(n,len(degreelist1)):
                    difference+=sum([G1[degreelist1[j]][k]["weight"] for k in G1.neighbors(degreelist1[j])])
            if len(degreelist2)>n:
                for j in range(n,len(degreelist2)):
                    difference+=sum([G2[degreelist2[j]][k]["weight"] for k in G2.neighbors(degreelist2[j])])
        #atom type only in one of the graphs
        else:
            for j in type_map1[i]:
                difference+=sum([G1[j][k]["weight"] for k in G1.neighbors(j)])
    for i in atom_types2:
        if i not in atom_types1:
            for j in type_map2[i]:
                difference+=sum([G2[j][k]["weight"] for k in G2.neighbors(j)])
    return difference/2

def get_cost(G1: nx.Graph, G2: nx.Graph, i: int, j: int) -> float:
    """
     Calculates the cost for mapping node i to j based on neighborhood

     Parameters
     ----------
     G1 : networkx.classes.graph.Graph
         Graph representing the first molecule.
     G2 : networkx.classes.graph.Graph
         Graph representing the second molecule.
     i : int
         Node of G1
     j : int
         Node of G2

     Returns:
     -------
     float
         Cost of mapping i to j

    """
    #Find all occuring atom types in neighborhood
    atom_types1=[]
    for k in G1.neighbors(i):
        if G1.nodes[k]["atom"] not in atom_types1:
            atom_types1.append(G1.nodes[k]["atom"])
    type_map1={}
    for k in atom_types1:
        type_map1[k]=list(filter(lambda x: k==G1.nodes[x]["atom"],G1.neighbors(i)))


    atom_types2=[]
    for k in G2.neighbors(j):
        if G2.nodes[k]["atom"] not in atom_types2:
            atom_types2.append(G2.nodes[k]["atom"])
    type_map2={}
    for k in atom_types2:
        type_map2[k]=list(filter(lambda x: k==G2.nodes[x]["atom"],G2.neighbors(j)))

    #calculate cost
    difference=0.
    #Every atom type is handled seperately
    for k in atom_types1:
        if k in atom_types2:
            n=min(len(type_map1[k]),len(type_map2[k]))
            #sort by incident edges by weight
            edgelist1=sorted(type_map1[k],key=lambda x:G1[i][x]["weight"],reverse=True)
            edgelist2=sorted(type_map2[k],key=lambda x:G2[j][x]["weight"],reverse=True)
            #map in order of sorted lists
            for l in range(n):
                difference+=(max(G1[i][edgelist1[l]]["weight"],G2[j][edgelist2[l]]["weight"])-min(G1[i][edgelist1[l]]["weight"],G2[j][edgelist2[l]]["weight"]))/2
            #cost for not mapped edges
            if len(edgelist1)>n:
                for l in range(n,len(edgelist1)):
                    difference+=G1[i][edgelist1[l]]["weight"]/2
            if len(edgelist2)>n:
                for l in range(n,len(edgelist2)):
                    difference+=G2[j][edgelist2[l]]["weight"]/2
        else:
            for l in type_map1[k]:
                difference+=G1[i][l]["weight"]/2
    for k in atom_types2:
        if k not in atom_types1:
            for l in type_map2[k]:
                difference+=G2[j][l]["weight"]/2
    return difference

def filter2_from_lib(G1,G2):
    """
     Finds a lower bound for the distance based on neighborhood

     Parameters
     ----------
     G1 : networkx.classes.graph.Graph
         Graph representing the first molecule.
     G2 : networkx.classes.graph.Graph
         Graph representing the second molecule.

     Returns:
     -------
     float
         Lower bound for the distance between the molecules

    """
    # Find all occuring atom types
    atom_types1=[]
    for i in G1.nodes:
        if G1.nodes[i]["atom"] not in atom_types1:
            atom_types1.append(G1.nodes[i]["atom"])

    atom_types2=[]
    for i in G2.nodes:
        if G2.nodes[i]["atom"] not in atom_types2:
            atom_types2.append(G2.nodes[i]["atom"])

    atom_types=atom_types1

    for i in atom_types2:
        if i not in atom_types:
            atom_types.append(i)
    #calculate distance
    res=0
    #handle every atom type seperately
    for i in atom_types:
        #filter by atom type
        nodes1=list(filter(lambda x: i==G1.nodes[x]["atom"],G1.nodes))
        nodes2=list(filter(lambda x: i==G2.nodes[x]["atom"],G2.nodes))
        #Create new graph for and solve minimum weight full matching
        G=nx.Graph()
        #Add node for every node of type i in G1 and G2
        for j in nodes1:
            G.add_node(tuple([1,j]))
        for j in nodes2:
            G.add_node(tuple([2,j]))
        #Add edges between all nodes of G1 and G2
        for j in nodes1:
            for k in nodes2:
                if G1.nodes[j]["atom"]==G2.nodes[k]["atom"]:
                    G.add_edge(tuple([1,j]),tuple([2,k]),weight=get_cost(G1,G2,j,k))
        #Add nodes if one graph has more nodes of type i than the other
        if len(nodes1)<len(nodes2):
            diff=len(nodes2)-len(nodes1)
            for j in range(1,diff+1):
                G.add_node(tuple([1,-j]))
                for k in nodes2:
                    G.add_edge(tuple([1,-j]),tuple([2,k]),weight=sum([G2[l][k]["weight"] for l in G2.neighbors(k)])/2)
        if len(nodes2)<len(nodes1):
            diff=len(nodes1)-len(nodes2)
            for j in range(1,diff+1):
                G.add_node(tuple([2,-j]))
                for k in nodes1:
                    G.add_edge(tuple([1,k]),tuple([2,-j]),weight=sum([G1[l][k]["weight"] for l in G1.neighbors(k)])/2)
        #Solve minimum weight full matching
        h=nx.bipartite.minimum_weight_full_matching(G)
        #Add weight of the matching
        for k in h:
            if k[0]==1:
                res=res+G[k][h[k]]["weight"]

    return res


def filter2_v2(G1, G2):
    """
    Optimized version of filter2 that uses memoization and Hungarian algorithm for faster computation, written by Nir Cohen. still slower than filter2 (which is the v3), and the batch version is even faster.
    
    Parameters
    ----------
    G1 : networkx.classes.graph.Graph
        Graph representing the first molecule.
    G2 : networkx.classes.graph.Graph
        Graph representing the second molecule.
        
    Returns
    -------
    float
        Lower bound for the distance between the molecules
    """
    
    # Collect atom types and create mappings - faster than checking membership repeatedly
    atom_types1 = {}
    for i in G1.nodes:
        atom = G1.nodes[i]["atom"]
        if atom not in atom_types1:
            atom_types1[atom] = []
        atom_types1[atom].append(i)
    
    atom_types2 = {}
    for i in G2.nodes:
        atom = G2.nodes[i]["atom"]
        if atom not in atom_types2:
            atom_types2[atom] = []
        atom_types2[atom].append(i)
    
    # Pre-calculate node neighborhood information (memoization)
    neighbor_info1 = {}
    for node in G1.nodes:
        neighbors = list(G1.neighbors(node))
        neighbor_info1[node] = {
            'weights': [G1[node][n]['weight'] for n in neighbors],
            'atoms': [G1.nodes[n]['atom'] for n in neighbors],
            'total_weight': sum(G1[node][n]['weight'] for n in neighbors) / 2
        }
    
    neighbor_info2 = {}
    for node in G2.nodes:
        neighbors = list(G2.neighbors(node))
        neighbor_info2[node] = {
            'weights': [G2[node][n]['weight'] for n in neighbors],
            'atoms': [G2.nodes[n]['atom'] for n in neighbors],
            'total_weight': sum(G2[node][n]['weight'] for n in neighbors) / 2
        }
    
    # Function to calculate cost between nodes
    def node_cost(node1, node2):
        # Use pre-calculated neighborhood info
        ni1 = neighbor_info1[node1]
        ni2 = neighbor_info2[node2]
        
        # Get all atom types in both neighborhoods
        atoms1 = set(ni1['atoms'])
        atoms2 = set(ni2['atoms'])
        all_atoms = atoms1.union(atoms2)
        
        cost = 0.0
        for atom in all_atoms:
            # Find weights for this atom type in both neighborhoods
            weights1 = [w for w, a in zip(ni1['weights'], ni1['atoms']) if a == atom]
            weights2 = [w for w, a in zip(ni2['weights'], ni2['atoms']) if a == atom]
            
            # Sort weights (descending)
            weights1.sort(reverse=True)
            weights2.sort(reverse=True)
            
            # Match weights
            n = min(len(weights1), len(weights2))
            if n > 0:
                for i in range(n):
                    cost += abs(weights1[i] - weights2[i]) / 2
            
            # Add unmatched weights
            cost += sum(weights1[n:]) / 2 + sum(weights2[n:]) / 2
        
        return cost
    
    # Calculate total cost
    total_cost = 0.0
    all_types = set(atom_types1.keys()).union(set(atom_types2.keys()))
    
    for atom_type in all_types:
        nodes1 = atom_types1.get(atom_type, [])
        nodes2 = atom_types2.get(atom_type, [])
        
        if not nodes1:
            # All nodes of this type in G2 have no match
            for n2 in nodes2:
                total_cost += neighbor_info2[n2]['total_weight']
            continue
            
        if not nodes2:
            # All nodes of this type in G1 have no match
            for n1 in nodes1:
                total_cost += neighbor_info1[n1]['total_weight']
            continue
        
        # Create cost matrix
        cost_matrix = [[node_cost(n1, n2) for n2 in nodes2] for n1 in nodes1]
        
        # Make square matrix for Hungarian algorithm
        n1, n2 = len(nodes1), len(nodes2)
        if n1 < n2:
            # Add dummy rows
            for _ in range(n2 - n1):
                row = [neighbor_info2[nodes2[j]]['total_weight'] for j in range(n2)]
                cost_matrix.append(row)
        elif n2 < n1:
            # Add dummy columns
            for i in range(n1):
                for _ in range(n1 - n2):
                    cost_matrix[i].append(neighbor_info1[nodes1[i]]['total_weight'])
        
        # Use Hungarian algorithm with scipy
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment_cost = sum(cost_matrix[i][j] for i, j in zip(row_ind, col_ind))
        total_cost += assignment_cost
    
    return total_cost


def filter2(G1, G2):
    """
    Most Optimized version of filter2 based on profiling results, but still better to use filter2_batch for batch processing.
    version: v3
    Key optimizations:
    1. Pre-compute and cache all graph data to minimize NetworkX access
    2. Pre-sort neighbor weights by atom type
    3. Use more efficient data structures
    4. Minimize repeated computations
    """
    from collections import defaultdict
    
    # Pre-compute all graph data to avoid repeated NetworkX access
    def precompute_graph_data(G):
        atom_types = defaultdict(list)
        neighbor_data = {}
        
        for node in G.nodes():
            atom = G.nodes[node]["atom"]
            atom_types[atom].append(node)
            
            # Get all neighbor data in one pass
            neighbors = list(G.neighbors(node))
            if neighbors:
                neighbor_weights = [G[node][n]['weight'] for n in neighbors]
                neighbor_atoms = [G.nodes[n]['atom'] for n in neighbors]
                
                # Group by atom type and sort immediately
                atom_weights = defaultdict(list)
                for weight, atom in zip(neighbor_weights, neighbor_atoms):
                    atom_weights[atom].append(weight)
                
                # Sort once and store
                for atom in atom_weights:
                    atom_weights[atom].sort(reverse=True)
                
                neighbor_data[node] = {
                    'atom_weights': dict(atom_weights),
                    'total_weight': sum(neighbor_weights) / 2
                }
            else:
                neighbor_data[node] = {
                    'atom_weights': {},
                    'total_weight': 0.0
                }
        
        return dict(atom_types), neighbor_data
    
    atom_types1, neighbor_data1 = precompute_graph_data(G1)
    atom_types2, neighbor_data2 = precompute_graph_data(G2)
    
    # Optimized node cost using pre-sorted data
    def node_cost_optimized(node1, node2):
        weights1 = neighbor_data1[node1]['atom_weights']
        weights2 = neighbor_data2[node2]['atom_weights']
        
        cost = 0.0
        all_atoms = set(weights1.keys()) | set(weights2.keys())
        
        for atom in all_atoms:
            w1 = weights1.get(atom, [])  # Already sorted
            w2 = weights2.get(atom, [])  # Already sorted
            
            min_len = min(len(w1), len(w2))
            cost += sum(abs(a - b) for a, b in zip(w1[:min_len], w2[:min_len])) / 2
            cost += sum(w1[min_len:]) / 2 + sum(w2[min_len:]) / 2
        
        return cost
    
    # Main computation
    total_cost = 0.0
    all_types = set(atom_types1.keys()) | set(atom_types2.keys())
    
    for atom_type in all_types:
        nodes1 = atom_types1.get(atom_type, [])
        nodes2 = atom_types2.get(atom_type, [])
        
        if not nodes1:
            total_cost += sum(neighbor_data2[n2]['total_weight'] for n2 in nodes2)
            continue
            
        if not nodes2:
            total_cost += sum(neighbor_data1[n1]['total_weight'] for n1 in nodes1)
            continue
        
        n1, n2 = len(nodes1), len(nodes2)
        max_size = max(n1, n2)
        
        import numpy as np
        cost_matrix = np.full((max_size, max_size), 0.0)
        
        # Fill actual costs
        for i, node1 in enumerate(nodes1):
            for j, node2 in enumerate(nodes2):
                cost_matrix[i, j] = node_cost_optimized(node1, node2)
        
        # Fill dummy costs
        if n1 < n2:
            for i in range(n1, max_size):
                for j, node2 in enumerate(nodes2):
                    cost_matrix[i, j] = neighbor_data2[node2]['total_weight']
        elif n2 < n1:
            for i, node1 in enumerate(nodes1):
                for j in range(n2, max_size):
                    cost_matrix[i, j] = neighbor_data1[node1]['total_weight']
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment_cost = cost_matrix[row_ind, col_ind].sum()
        total_cost += assignment_cost
    
    return total_cost


def filter2_batch(graphs_list1, graphs_list2=None):
    """
    Batch processing version for computing many pairwise distances efficiently.
    This is optimized for all-to-all comparisons of molecule lists.
    
    Parameters
    ----------
    graphs_list1 : list of networkx.Graph
        First set of molecular graphs
    graphs_list2 : list of networkx.Graph, optional
        Second set of molecular graphs. If None, computes all-to-all within graphs_list1
        
    Returns
    -------
    numpy.ndarray
        Distance matrix where result[i,j] is the distance between graphs_list1[i] and graphs_list2[j]
        (or graphs_list1[j] if graphs_list2 is None)
    """
    import numpy as np
    from collections import defaultdict
    
    if graphs_list2 is None:
        graphs_list2 = graphs_list1
        symmetric = True
    else:
        symmetric = False
    
    n1, n2 = len(graphs_list1), len(graphs_list2)
    
    # Pre-compute all graph data
    all_graph_data = {}
    
    def precompute_graph(G, graph_id):
        atom_types = defaultdict(list)
        neighbor_data = {}
        
        for node in G.nodes():
            atom = G.nodes[node]["atom"]
            atom_types[atom].append(node)
            
            neighbors = list(G.neighbors(node))
            if neighbors:
                weights = [G[node][n]['weight'] for n in neighbors]
                atoms = [G.nodes[n]['atom'] for n in neighbors]
                
                # Group by atom type and sort immediately
                atom_weights = defaultdict(list)
                for w, a in zip(weights, atoms):
                    atom_weights[a].append(w)
                
                # Sort once
                for a in atom_weights:
                    atom_weights[a].sort(reverse=True)
                
                neighbor_data[node] = {
                    'atom_weights': dict(atom_weights),
                    'total_weight': sum(weights) / 2
                }
            else:
                neighbor_data[node] = {
                    'atom_weights': {},
                    'total_weight': 0.0
                }
        
        all_graph_data[graph_id] = {
            'atom_types': dict(atom_types),
            'neighbor_data': neighbor_data
        }
    
    # Pre-compute all graphs
    for i, G in enumerate(graphs_list1):
        precompute_graph(G, f"1_{i}")
    
    if not symmetric:
        for i, G in enumerate(graphs_list2):
            precompute_graph(G, f"2_{i}")
    
    def fast_node_cost(node1, data1, node2, data2):
        """Fast node cost using pre-sorted data."""
        nd1 = data1['neighbor_data'][node1]
        nd2 = data2['neighbor_data'][node2]
        
        atom_weights1 = nd1['atom_weights']
        atom_weights2 = nd2['atom_weights']
        
        cost = 0.0
        all_atoms = set(atom_weights1.keys()) | set(atom_weights2.keys())
        
        for atom in all_atoms:
            weights1 = atom_weights1.get(atom, [])
            weights2 = atom_weights2.get(atom, [])
            
            n = min(len(weights1), len(weights2))
            cost += sum(abs(w1 - w2) / 2 for w1, w2 in zip(weights1[:n], weights2[:n]))
            cost += sum(weights1[n:]) / 2 + sum(weights2[n:]) / 2
        
        return cost
    
    def compute_single_pair(data1, data2):
        """Compute distance between two pre-processed graphs."""
        total_cost = 0.0
        all_types = set(data1['atom_types'].keys()) | set(data2['atom_types'].keys())
        
        for atom_type in all_types:
            nodes1 = data1['atom_types'].get(atom_type, [])
            nodes2 = data2['atom_types'].get(atom_type, [])
            
            if not nodes1:
                total_cost += sum(data2['neighbor_data'][n2]['total_weight'] for n2 in nodes2)
                continue
                
            if not nodes2:
                total_cost += sum(data1['neighbor_data'][n1]['total_weight'] for n1 in nodes1)
                continue
            
            n1, n2 = len(nodes1), len(nodes2)
            max_size = max(n1, n2)
            
            cost_matrix = np.zeros((max_size, max_size))
            
            for i, node1 in enumerate(nodes1):
                for j, node2 in enumerate(nodes2):
                    cost_matrix[i, j] = fast_node_cost(node1, data1, node2, data2)
            
            # Fill dummy costs
            if n1 < n2:
                for i in range(n1, max_size):
                    for j, node2 in enumerate(nodes2):
                        cost_matrix[i, j] = data2['neighbor_data'][node2]['total_weight']
            elif n2 < n1:
                for i, node1 in enumerate(nodes1):
                    for j in range(n2, max_size):
                        cost_matrix[i, j] = data1['neighbor_data'][node1]['total_weight']
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignment_cost = cost_matrix[row_ind, col_ind].sum()
            total_cost += assignment_cost
        
        return total_cost
    
    # Compute results
    results = np.zeros((n1, n2))
    
    for i in range(n1):
        start_j = i if symmetric else 0
        for j in range(start_j, n2):
            data1 = all_graph_data[f"1_{i}"]
            data2 = all_graph_data[f"2_{j}"] if not symmetric else all_graph_data[f"1_{j}"]
            
            distance = compute_single_pair(data1, data2)
            results[i, j] = distance
            
            if symmetric and i != j:
                results[j, i] = distance
    
    return results

if __name__ == "__main__":
    import sys
    from time import perf_counter
    from typing import List
    from .lib import construct_graph, MCES_ILP

    if '--bounds-test' in sys.argv:
        # compare the bounds we get from filter1, filter2, filter2_from_lib, filter2_v3_optimized, filter2_batch, filter3, and filter4
        smiles_examples: List[str] = [
            # Small molecules
            "C",                # Methane
            "CC",               # Ethane
            "CCO",              # Ethanol
            "CC(=O)C",          # Acetone
            "c1ccccc1",         # Benzene
            "Cc1ccccc1",        # Toluene
            "CC(=O)O",          # Acetic acid
            "C1CCCCC1",         # Cyclohexane
            "C(C1C(C(C(C(O1)O)O)O)O)O",  # Glucose

            # Medications / larger molecules
            "CC(=O)Oc1ccccc1C(=O)O",     # Aspirin
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",# Caffeine
            "CN1CCC[C@H]1c2cccnc2",      # Nicotine
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen
            "CC1=C(C(=O)NC(=O)N1)N",     # Paracetamol (acetaminophen)
            "CC(C)NCC(O)COc1ccc2nc(S(N)(=O)=O)sc2c1", # Hydrochlorothiazide
            "CC(C)C1=CC(=C(C=C1)O)C(C)C(=O)O", # Naproxen
            "CCOC(=O)C1=CC=CC=C1N",      # Lidocaine
            "CC(C)C1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen (repeat for clarity)
            "CCN(CC)CCOC(=O)C1=CC=CC=C1", # Propranolol
            "CC(C)C1=CC(=O)NC(=O)N1",    # Paracetamol (repeat for clarity)
        ]*1
        graphs = [construct_graph(smiles) for smiles in smiles_examples]

        # time the filters
        start_time = perf_counter()
        filter1_results = [filter1(G1, G2) for G1 in graphs for G2 in graphs]
        time1 = perf_counter() - start_time

        start_time = perf_counter()
        filter2_v2_results = [filter2_v2(G1, G2) for G1 in graphs for G2 in graphs]
        time2 = perf_counter() - start_time

        start_time = perf_counter()
        filter2_from_lib_results = [filter2_from_lib(G1, G2) for G1 in graphs for G2 in graphs]
        time2_from_lib = perf_counter() - start_time

        start_time = perf_counter()
        filter2_v3_results = [filter2(G1, G2) for G1 in graphs for G2 in graphs]
        time2_v3 = perf_counter() - start_time

        start_time = perf_counter()
        # filter2_batch returns a matrix, flatten to match the other results
        import numpy as np
        batch_matrix = filter2_batch(graphs)
        filter2_batch_results = batch_matrix.flatten()
        time2_batch = perf_counter() - start_time

        
        # Compute true MCES distances (this will be the slowest)
        start_time = perf_counter()
        mces_results = []
        for i, G1 in enumerate(graphs):
            for j, G2 in enumerate(graphs):
                if i == j:
                    mces_results.append(0.0)  # Distance to self is 0
                else:
                    try:
                        distance, distance_type = MCES_ILP(G1, G2,threshold=1000,no_ilp_threshold=True,solver="default")
                        mces_results.append(distance)
                    except Exception as e:
                        print(f"MCES_ILP failed for graphs {i}, {j}: {e}")
                        mces_results.append(float('inf'))  # Mark as failed
        time_mces = perf_counter() - start_time

        # Check for invalid bounds (any filter result > true MCES distance)
        invalid_bounds_found = False
        for i, (f1, f2, f2_lib, f2_v3, f2_batch, mces) in enumerate(zip(
            filter1_results, filter2_v2_results, filter2_from_lib_results, 
            filter2_v3_results, filter2_batch_results, 
            mces_results)):
            
            if mces == float('inf'):  # Skip failed MCES computations
                continue
                
            filters = [("Filter1", f1), ("Filter2", f2), ("Filter2_lib", f2_lib), 
                      ("Filter2_v3", f2_v3), ("Filter2_batch", f2_batch)]
            
            for filter_name, filter_result in filters:
                if filter_result > mces + 1e-6:  # Small tolerance for numerical errors
                    print(f"INVALID BOUND ALERT! Test {i}: {filter_name} = {filter_result:.6f} > MCES = {mces:.6f}")
                    invalid_bounds_found = True

        # Print results where filter2 variants differ
        for i, (f1, f2, f2_lib, f2_v3, f2_batch, mces) in enumerate(zip(
            filter1_results, filter2_v2_results, filter2_from_lib_results, 
            filter2_v3_results, filter2_batch_results,  mces_results)):
            
            if not (np.isclose(f2, f2_lib) and np.isclose(f2, f2_v3) and np.isclose(f2, f2_batch)):
                print(f"Test {i}:")
                print(f"  Filter1: {f1:.6f}")
                print(f"  Filter2: {f2:.6f}")
                print(f"  Filter2 from lib: {f2_lib:.6f}")
                print(f"  Filter2 v3 optimized: {f2_v3:.6f}")
                print(f"  Filter2 batch: {f2_batch:.6f}")
                print(f"  MCES (true): {mces:.6f}")
                print("  ALERT: Filter2 variants differ!")

        # Print summary statistics
        valid_indices = [i for i, mces in enumerate(mces_results) if mces != float('inf')]
        
        if valid_indices:
            print(f"\nSummary for {len(valid_indices)} valid MCES computations:")
            
            if all(np.isclose(filter2_v2_results[i], filter2_from_lib_results[i]) and 
                   np.isclose(filter2_v2_results[i], filter2_v3_results[i]) and 
                   np.isclose(filter2_v2_results[i], filter2_batch_results[i])
                   for i in valid_indices):
                print("All results of filter2, filter2_from_lib, filter2_v3_optimized, and filter2_batch are consistent.")
                
                # Calculate average differences for valid results only
                avg_diff_f2 = sum((filter2_v2_results[i] - filter1_results[i]) for i in valid_indices) / len(valid_indices)
                
                
                # Calculate bound tightness (how close bounds are to true values)
                avg_gap_f1 = sum((mces_results[i] - filter1_results[i]) for i in valid_indices) / len(valid_indices)
                avg_gap_f2 = sum((mces_results[i] - filter2_v2_results[i]) for i in valid_indices) / len(valid_indices)

                
                print("Average improvement over filter1:")
                print(f"  Filter2 - Filter1: {avg_diff_f2:.4f}")

                
                print("Average gap to true MCES distance:")
                print(f"  MCES - Filter1: {avg_gap_f1:.4f}")
                print(f"  MCES - Filter2: {avg_gap_f2:.4f}")

                

        if not invalid_bounds_found:
            print("\n✓ All bounds are valid (no filter exceeded true MCES distance)")
        else:
            print("\n⚠️  INVALID BOUNDS DETECTED! Check the algorithms above.")

        print("\nTiming results:")
        print(f"Time for filter1: {time1:.2f} seconds")
        print(f"Time for filter2: {time2:.2f} seconds")
        print(f"Time for filter2 from lib: {time2_from_lib:.2f} seconds")
        print(f"Time for filter2_v3_optimized: {time2_v3:.2f} seconds")
        print(f"Time for filter2_batch: {time2_batch:.2f} seconds")
        print(f"Time for MCES_ILP (true): {time_mces:.2f} seconds")
