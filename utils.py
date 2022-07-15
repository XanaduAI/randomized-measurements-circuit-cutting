from typing import List, Optional, Tuple

import networkx as nx
import pennylane as qml
from pennylane import numpy as np


def clustered_chain_graph(
    n: int, r: int, k: int, q1: float, q2: float, seed: Optional[int] = None
) -> Tuple[nx.Graph, List[List[int]], List[List[int]]]:
    """
    Function to build clustered chain graph

    Args:
        n (int): number of nodes in each cluster
        r (int): number of clusters
        k (int): number of vertex separators between each cluster pair
        q1 (float): probability of an edge connecting any two nodes in a cluster
        q2 (float): probability of an edge connecting a vertex separator to any node in a cluster
        seed (Optional[int]=None): seed for fixing edge generation

    Returns:
        nx.Graph: clustered chain graph
    """

    if r <= 0 or not isinstance(r, int):
        raise ValueError("Number of clusters must be an integer greater than 0")

    clusters = []
    for i in range(r):
        _seed = seed * i if seed is not None else None
        cluster = nx.erdos_renyi_graph(n, q1, seed=_seed)
        nx.set_node_attributes(cluster, f"cluster_{i}", "subgraph")
        clusters.append(cluster)

    separators = []
    for i in range(r - 1):
        separator = nx.empty_graph(k)
        nx.set_node_attributes(separator, f"separator_{i}", "subgraph")
        separators.append(separator)

    G = nx.disjoint_union_all(clusters + separators)

    cluster_nodes = [
        [n[0] for n in G.nodes(data="subgraph") if n[1] == f"cluster_{i}"] for i in range(r)
    ]
    separator_nodes = [
        [n[0] for n in G.nodes(data="subgraph") if n[1] == f"separator_{i}"] for i in range(r - 1)
    ]

    rng = np.random.default_rng(seed)

    for i, separator in enumerate(separator_nodes):
        for s in separator:
            for c in cluster_nodes[i] + cluster_nodes[i + 1]:
                if rng.random() < q2:
                    G.add_edge(s, c)

    return G, cluster_nodes, separator_nodes


def get_qaoa_circuit(
    G: nx.Graph,
    cluster_nodes: List[List[int]],
    separator_nodes: List[List[int]],
    params: Tuple[Tuple[float]],
    layers: int = 1,
) -> qml.tape.QuantumTape:
    """
    Function to build QAOA max-cut circuit tape from graph including `WireCut` 
    operations
    
    Args:
        G (nx.Graph): problem graph to be solved using QAOA
        cluster_nodes (List[List[int]]): nodes of the clusters within the graph
        separator_nodes (List[List[int]]): nodes of the separators in the graph
        params (Tuple[Tuple[float]]): parameters of the QAOA circuit to be optimized
        layers (int): number of layer in the QAOA circuit
        
    Returns:
        QuantumTape: the QAOA tape containing `WireCut` operations
    """
    wires = len(G)
    r = len(cluster_nodes)

    cost, _ = qml.qaoa.maxcut(G)

    with qml.tape.QuantumTape() as tape:
        for w in range(wires):
            qml.Hadamard(wires=w)

        for l in range(layers):
            gamma, beta = params[l]

            for i, c in enumerate(cluster_nodes):
                if i == 0:
                    current_separator = []
                    next_separator = separator_nodes[0]
                elif i == r - 1:
                    current_separator = separator_nodes[-1]
                    next_separator = []
                else:
                    current_separator = separator_nodes[i - 1]
                    next_separator = separator_nodes[i]

                for cs in current_separator:
                    qml.WireCut(wires=cs)

                nodes = c + current_separator + next_separator
                subgraph = G.subgraph(nodes)

                for edge in subgraph.edges:
                    qml.IsingZZ(2*gamma, wires=edge) # multiply param by 2 for consistency with analytic cost

            # mixer layer
            for w in range(wires):
                qml.RX(2*beta, wires=w)


            # reset cuts
            if l < layers - 1:
                for s in separator_nodes:
                    qml.WireCut(wires=s)

        [qml.expval(op) for op in cost.ops if not isinstance(op, qml.ops.identity.Identity)]


    return tape
