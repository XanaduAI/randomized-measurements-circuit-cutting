from typing import Sequence
from itertools import combinations, product
import networkx as nx

def _insert_plot_properties(
    graph: nx.Graph,
    top: Sequence[int],
    middle: Sequence[int],
    bottom: Sequence[int]
):

    pos = {}
    color = {}
    loc = {}
    cut = {}
    d = (len(middle) - len(top))/2
    mt, mm, mb = [min(row) for row in [top, middle, bottom]]

    for n in graph:

        if n in top:
            pos[n] = (float(n - mt + d), 1.)
            color[n] = 'steelblue'
            loc[n] = 'side'
            cut[n] = False
        elif n in middle:
            pos[n] = (float(n - mm), 0.)
            color[n] = 'gold'
            loc[n] = 'middle'
            cut[n] = True
        else:
            pos[n] = (float(n - mb + d), -1.)
            color[n] = 'steelblue'
            loc[n] = 'side'
            cut[n] = False

    nx.set_node_attributes(graph, pos, name='pos')
    nx.set_node_attributes(graph, color, name='color')
    nx.set_node_attributes(graph, loc, name='loc')
    nx.set_node_attributes(graph, cut, name='cut')

    edge_color = {}

    for u, v in graph.edges():

        if u in top or v in top:
            edge_color[(u,v)] = 0
        elif u in bottom or v in bottom:
            edge_color[(u,v)] = 1

    nx.set_edge_attributes(graph, edge_color, name='color')

def make_graph(l: int, c: int):

    top = range(0, l)
    middle = range(l, l+c)
    bottom = range(l+c, c+2*l)

    graph = nx.Graph()
    graph.add_edges_from(combinations(top, 2), color=0)
    graph.add_edges_from(product(top, middle), color=0)
    graph.add_edges_from(product(middle, bottom), color=1)
    graph.add_edges_from(combinations(bottom, 2), color=1)

    _insert_plot_properties(graph, top, middle, bottom)

    return graph