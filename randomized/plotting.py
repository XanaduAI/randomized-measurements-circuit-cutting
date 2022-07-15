import networkx as nx
from matplotlib import pyplot as plt

def draw_interaction_graph(
    graph: nx.Graph,
    ax=None,
    *,
    figsize=(8,6),
    node_size=800,
    font_size=16,
    width=2,
    font_color='black',
    font_weight='bold',
    with_labels=True,
    **kwargs
):

    pos = nx.get_node_attributes(graph, 'pos')
    node_color = [d['color'] for _, d in graph.nodes(data=True)]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    nx.draw(
        graph,
        ax=ax,
        pos=pos,
        node_size=node_size,
        font_size=font_size,
        width=width,
        node_color=node_color,
        font_color=font_color,
        font_weight=font_weight,
        with_labels=with_labels,
        **kwargs
    )

    return ax