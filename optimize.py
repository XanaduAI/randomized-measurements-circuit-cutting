import sys
import os
import pennylane as qml
from pennylane import numpy as np
import torch
import gc
from timeit import default_timer as timer
from datetime import datetime, timedelta

from utils import clustered_chain_graph, get_qaoa_circuit

from ray import train

########################################################################
# Execution env setup
########################################################################
split = 1 # number of gpus per tape execution (can be fractional)

r = 3 # number of clusters
n = 20  # nodes in clusters
k = 1  # vertex separators

layers = 2 

time_stamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
filename = f"./data/optimisation/opt_p={layers}_r={r}_n={n}_k={k}_{time_stamp}"
sys.stdout = open(filename, 'w')

print(f"\nProblem graph with {r} clusters of {n} nodes and {k} vertex separators", flush=True)

q1 = 0.7
q2 = 0.3

seed = 1967

G, cluster_nodes, separator_nodes = clustered_chain_graph(n, r, k, q1, q2, seed=seed)

import ray
ray.init() # Should be updated according to system config

print("Nodes in the Ray cluster:", flush=True)
print(ray.nodes(), flush=True)

print(f"\ncluster resources: {ray.available_resources()}", flush=True)
print(f"\nresources: {ray.available_resources()}", flush=True)

frag_wires = n + (3*layers -1)*k  # number of wires on biggest fragment
print(f"\nSimulating {frag_wires} qubits for largest fragment\n", flush=True)

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

@ray.remote(num_gpus=split)
def execute_tape(tape):
    dev = qml.device("lightning.gpu", wires=frag_wires)
    res = dev.execute(tape)
    del dev
    gc.collect()
    torch.cuda.empty_cache()
    return res

@ray.remote(num_gpus=split)
def execute_tape_jac(tape):
    dev = qml.device("lightning.gpu", wires=frag_wires)
    return dev.adjoint_jacobian(tape)

########################################################################
# Add samples Ray calls for S/R and for circuit execution
########################################################################
class RayExecutor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, tape):
        ctx.tape = tape
        return execute_tape.remote(tape)

    @staticmethod
    def backward(ctx, dy):
        jac = torch.tensor(ray.get(execute_tape_jac.remote(ctx.tape)), requires_grad=True)
        return dy * jac, None
        
########################################################################
# Immitate NN functionality and register methods to autograd
########################################################################
class CircNetFull(torch.nn.Module):
    """
    Executes a QAOA circuit for a given set of parameters and returns a cost 
    (energy) value.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, params):
        circuit = get_qaoa_circuit(G, cluster_nodes, separator_nodes, params, layers)
        
        start_frag = timer()
        
        print(f"\nFinding fragments ... ", flush=True)
        fragment_configs, processing_fn = qml.cut_circuit(circuit, device_wires=range(frag_wires))
        end_frag = timer()
        elapsed_frag = end_frag - start_frag
        format_frag = str(timedelta(seconds=elapsed_frag))
        print(f"\nFragmentation time: {format_frag}")
        
        print(f"\nTotal number of fragment tapes = {len(fragment_configs)}", flush=True)
        
        frag_depth, deepest_tape = find_depth(fragment_configs)
        print(f"\nDepth of largest fragment = {frag_depth}", flush=True)
        print(f"\nDepth of deepest tape = {deepest_tape}", flush=True)
        
        start_cut = timer()
        results = ray.get([RayExecutor.apply(t.get_parameters(), t) for t in fragment_configs])
        
        end_cut = timer()
        elapsed_cut = end_cut - start_cut
        format_cut = str(timedelta(seconds=elapsed_cut))
        print(f"\nCircuit cutting time: {format_cut}", flush=True)
        
        return (sum(processing_fn(results)))


########################################################################
# Gradients 
########################################################################
def execute_grad(params, circuit):
    """
    Function to find and execute gradient tapes
    """

    start_grad = timer()
    delta = 0.001
    
    forward_tapes = []
    backward_tapes = []
    shifted = params.copy()
    
    for l in range(len(shifted)): # iterate over layers
        for i in range(len(shifted[l])): # iterate over params
            
            shifted[l][i] += delta / 2
            forward = get_qaoa_circuit(G, cluster_nodes, separator_nodes, shifted, layers)
            forward_tapes.append(forward)
            
            shifted[l][i] -= delta
            backward = get_qaoa_circuit(G, cluster_nodes, separator_nodes, shifted, layers)
            backward_tapes.append(backward)
    
    grad_circs = forward_tapes + backward_tapes
    print(f"\nTotal number of gradient circuits = {len(grad_circs)}", flush=True)
    print("\nFinding gradient fragments ...", flush=True)

    f_res = []
    for f_circ in forward_tapes:
        fragment_configs, processing_fn = qml.cut_circuit(f_circ, device_wires=range(frag_wires))
        f_results = ray.get([RayExecutor.apply(t.get_parameters(), t) for t in fragment_configs])
        f_res.append(sum(processing_fn(f_results)))
        
    b_res = []
    for b_circ in backward_tapes:
        fragment_configs, processing_fn = qml.cut_circuit(b_circ, device_wires=range(frag_wires))
        b_results = ray.get([RayExecutor.apply(t.get_parameters(), t) for t in fragment_configs])
        b_res.append(sum(processing_fn(b_results)))

    grads = []
    for fwd, bkwd in zip(f_res, b_res):
        val = (fwd - bkwd) /delta
        grads.append(val)
    
    end_grad = timer()
    elapsed_grad = end_grad - start_grad
    format_grad = str(timedelta(seconds=elapsed_grad))
    print(f"\nGradient evaluation time: {format_grad}", flush=True)
    
    return np.array(grads).reshape(params.shape)
        
def grad_descent():
    """
    Function to perform gradient gradient descent
    """
    init_params = np.array([[0.15, 0.2]] * layers, requires_grad=True)
    print(f"\nInitial params: {init_params}")
    circuit = get_qaoa_circuit(G, cluster_nodes, separator_nodes, init_params, layers)
    
    print(f"\nTotal number of qubits = {len(circuit.wires)}", flush=True)
    full_depth = circuit.specs["depth"]
    print(f"\nDepth of full (uncut) circuit = {full_depth}", flush=True)

    params = init_params
    start_opt = timer()
    
    for i in range(20):
        print(f"\nStep {i}:")
        print(f"\nNumber of params = {params.size}", flush=True)
        en = CircNetFull()(params)
        print(f"\nEnergy at step {i} = {en}", flush=True)
        grad = execute_grad(params, circuit)
        print(f"\nGrad len = {len(grad)}", flush=True)
        params -= 0.0001*grad
            
    end_opt = timer()
    elapsed_opt = end_opt - start_opt
    format_opt = str(timedelta(seconds=elapsed_opt))
    print(f"\nOptimisation time: {format_opt}", flush=True)
    
    print(f"\nFinal full parameters {params}", flush=True)
    print(f"\n Final cost = {en}", flush=True)
    
grad_descent()
