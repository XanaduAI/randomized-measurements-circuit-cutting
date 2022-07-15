import pennylane as qml
from pennylane import numpy as np
from timeit import default_timer as timer
from datetime import datetime, timedelta

from utils import clustered_chain_graph, get_qaoa_circuit
import sys
import torch
import gc
import os

split = 1 # number of gpus per tape execution (can be fractional)

r = 2  # number of clusters
n = 25  # nodes in clusters
k = 1  # vertex separators

layers = 2


time_stamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
filename = f"./data/forward_pass/p={layers}_r={r}_n={n}_k={k}_{time_stamp}"
sys.stdout = open(filename, 'w')

print(f"\nProblem graph with {r} clusters of {n} nodes and {k} vertex separators")
print(f"\nNumber of layers = {layers}")

q1 = 0.7
q2 = 0.3

seed = 1967

G, cluster_nodes, separator_nodes = clustered_chain_graph(n, r, k, q1, q2, seed=seed)

#########################
# distribute computation
#########################
import ray
ray.init() # Should be updated according to system config

print("Nodes in the Ray cluster:", flush=True)
print(ray.nodes(), flush=True)

print(f"\ncluster resources: {ray.available_resources()}", flush=True)
print(f"\nresources: {ray.available_resources()}", flush=True)

frag_wires = n + (3*layers -1)*k  # number of wires on biggest fragment
print(f"\nSimulating {frag_wires} qubits for largest fragment")
print(f"\nTotal number of qubits = {r*n + k*(r - 1)}")

@ray.remote(num_gpus=split)
def execute_tapes(tape):
    dev = qml.device("lightning.gpu", wires=frag_wires)
    res = dev.execute(tape)
    del dev
    gc.collect()
    torch.cuda.empty_cache()
    return res
    
params = np.array([[7.20792567e-01, 1.02761748e-04]] * layers, requires_grad=True)
circuit = get_qaoa_circuit(G, cluster_nodes, separator_nodes, params, layers)

full_depth = circuit.specs["depth"]
print(f"\nDepth of full (uncut) circuit = {full_depth}")

print(f"\nresources: {ray.available_resources()}")

##################
# circuit cutting
##################
start_cut = timer()

fragment_configs, processing_fn = qml.cut_circuit(circuit, device_wires=range(frag_wires))

print(f"\nTotal number of frag tapes = {len(fragment_configs)}")

end_cut = timer()
elapsed_cut = end_cut - start_cut
format_cut = str(timedelta(seconds=elapsed_cut))
print(f"\nCircuit cutting time: {format_cut}")

def find_depth(tapes):
    # Assuming the same depth for all configurations of largest fragments
    largest_width = 0
    all_depths = []
    for tpe in tapes:
        all_depths.append(tpe.specs["depth"])
        wire_num = len(tpe.wires)
        if wire_num > largest_width:
            largest_width = wire_num
            largest_frag = tpe  
            
    return (largest_frag.specs["depth"], max(all_depths))

frag_depth, deepest_tape = find_depth(fragment_configs)
print(f"\nDepth of largest fragment = {frag_depth}", flush=True)
print(f"\nDepth of deepest tape = {deepest_tape}", flush=True)

################
# execute tapes
################
start_ray = timer()

f = [execute_tapes.remote(tape) for tape in fragment_configs]
output = [ray.get(res) for res in f] 

end_ray = timer()
elapsed_ray = end_ray - start_ray
format_ray = str(timedelta(seconds=elapsed_ray))
print(f"\nRay run time: {format_ray}") 

##############
# postprocess
##############
start_pp = timer()

pp_result = sum(processing_fn(output))

end_pp = timer()
elapsed_pp = end_pp - start_pp
format_pp = str(timedelta(seconds=elapsed_pp))
print(f"\nPostprocessing time: {format_pp}") 
print(f"\nSimulated cost value = {pp_result}")
