import pulp
import networkx as nx
from time import time
from rdkit import Chem
from scipy.optimize import linear_sum_assignment


def MCES(smiles1, smiles2, threshold=10, i=0, solver='default', solver_options={}, no_ilp_threshold=False, always_stronger_bound=True, catch_errors=False):
    """
    Calculates the distance between two molecules

    Parameters
    ----------
    smiles1 : str
        SMILES of the first molecule
    smiles2 : str
        SMILES of the second molecule
    threshold : float
        Threshold for the comparison. Exact distance is only calculated if the distance is lower than the threshold.
        If set to -1 the exact distance is always calculated.
    i : int
        index, mainly for parallelization
    solver: string
        ILP-solver used for solving MCES. Example:CPLEX_CMD
    solver_options: dict
        additional options to pass to solvers. Example: threads=1 for better multi-threaded performance
    no_ilp_threshold: bool
        if true, always return exact distance even if it is below the threshold (slower)
    always_stronger_bound: bool
        if true, always compute and use the second stronger bound

    Returns:
    -------
    int
        index
    float
        Distance between the molecules
    float
        Time taken for the calculation
    int
        Type of Distance:
            1 : Exact Distance
            2 : Lower bound (if the exact distance is above the threshold; bound chosen dynamically)
            4 : Lower bound (second lower bound was used)

    """
    start = time()
    # construct graph for both smiles.
    G1 = construct_graph(smiles1)
    G2 = construct_graph(smiles2)
    if threshold != -1:         # with `-1` always compute exact distance
        # filter out if distance is above the threshold
        try:
            distance, compute_mode = apply_filter(G1, G2, threshold, always_stronger_bound=always_stronger_bound)
            if distance > threshold:
                return i, distance, time() - start, compute_mode
        except Exception as e:
            # print('ERROR:', smiles1, smiles2, 'filter', e, file=sys.stderr)
            if (catch_errors):
                distance = -1
                compute_mode = 2
            else:
                raise e
    # calculate MCES
    try:
        distance, compute_mode = MCES_ILP(G1, G2, threshold, solver, solver_options=solver_options,
                                          no_ilp_threshold=no_ilp_threshold)
    except Exception as e:
        # print('ERROR:', smiles1, smiles2, 'exact', e, file=sys.stderr)
        if (catch_errors):
            distance = -1
            compute_mode = 1
        else:
            raise e
    return i, distance, time() - start, compute_mode


def MCES_ILP(G1, G2, threshold, solver='GUROBI', solver_options={}, no_ilp_threshold=False):
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
    if solver=="default":
        ILP.solve()
    else:
        sol=pulp.getSolver(solver, **solver_options)
        ILP.solve(sol)
    if ILP.status==1:
        val = ILP.objective.value()
        if val is None:
            return 0, 1
        return float(ILP.objective.value()),1
    else:
        return threshold,2

def construct_graph(smiles:str):
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
    m = Chem.MolFromSmiles(smiles)
    # convert the molecule into a graph
    # The bond and atom types are converted to node/edge attributes
    G=nx.Graph()
    for atom in m.GetAtoms():
        G.add_node(atom.GetIdx(),atom=atom.GetSymbol())
    for bond in m.GetBonds():
        G.add_edge(bond.GetBeginAtom().GetIdx(),bond.GetEndAtom().GetIdx(),weight=bond.GetBondTypeAsDouble())
    return G


def filter1(G1,G2):
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

def get_cost(G1,G2,i,j):
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


def filter2(G1, G2):
    """
    Optimized version of filter2 that uses memoization and Hungarian algorithm for faster computation, written by Nir Cohen
    
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

def apply_filter(G1,G2,threshold,always_stronger_bound=True):
    """
     Finds a lower bound for the distance

     Parameters
     ----------
     G1 : networkx.classes.graph.Graph
         Graph representing the first molecule.
     G2 : networkx.classes.graph.Graph
         Graph representing the second molecule.
     threshold : int
         Threshold for the comparison. We want to find a lower bound that is higher than the threshold
     always_stronger_bound : bool
         if true, always compute and use the second stronger bound



     Returns:
     -------
     float
         Lower bound for the distance between the molecules
     int
         Which lower bound was chosen: 2 - depending on threshold, 4 - second lower bound

    """
    if always_stronger_bound:
        d=filter2(G1,G2)
        return d, 4
    else:
        #calculate first lower bound
        d=filter1(G1,G2)
        #if below threshold calculate second lower bound
        if d<=threshold:
            d=filter2(G1,G2)
            if d<=threshold:
                return d, 2

        return d, 2