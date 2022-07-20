import pennylane as qml
from pennylane import numpy as np

def QAOACostLayer(graph, gamma, flip = False):

    edges = list(graph.edges(data=True))
    cut_wires = [n for n, d in graph.nodes(data=True) if d['cut']]

    if not flip:

        for u, v, d in edges:
            if d['color'] == 0:
                qml.MultiRZ(2*gamma, wires=[u, v])
        
        qml.WireCut(wires=cut_wires)

        for u, v, d in edges:
            if d['color'] == 1:
                qml.MultiRZ(2*gamma, wires=[u, v])

    else:

        for u, v, d in reversed(edges):
            if d['color'] == 1:
                qml.MultiRZ(2*gamma, wires=[u, v])

        qml.WireCut(wires=cut_wires)

        for u, v, d in reversed(edges):
            if d['color'] == 0:
                qml.MultiRZ(2*gamma, wires=[u, v])

def QAOAMixerLayer(graph, beta):
    for u in graph.nodes():
        qml.RX(2*beta, wires=u)

def QAOATemplate(graph, params):

    params = np.atleast_2d(params)
    n_layers = params.shape[0]

    for i in range(len(graph)):
        qml.Hadamard(wires=i)

    for i in range(n_layers):
        QAOACostLayer(graph, params[i,0], flip = i%2 == 1)
        QAOAMixerLayer(graph, params[i,1])

def cost_from_samples(graph, samples):
    B = np.atleast_2d(samples)
    z = (-1)**B[:, graph.edges()]
    cost = z.prod(axis=-1).sum(axis=-1)
    return cost/len(graph.edges)