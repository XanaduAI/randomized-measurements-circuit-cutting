import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms import qcut
from pennylane.tape import QuantumTape

def _reshape_results(results, shots: int):
    results = [qml.math.flatten(r) for r in results]
    results = [results[i : i + shots] for i in range(0, len(results), shots)]
    results = list(map(list, zip(*results)))  # calculate list-based transpose
    return results

def qcut_processing_fn_mc(results, communication_graph, settings, shots: int, classical_processing_fn):

    res0 = results[0]
    results = _reshape_results(results, shots)
    out_degrees = [d for _, d in communication_graph.out_degree]

    evals = (0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5)
    expvals = []

    for result, setting in zip(results, settings.T):
        sample_terminal = []
        sample_mid = []

        for fragment_result, out_degree in zip(result, out_degrees):
            sample_terminal.append(fragment_result[: -out_degree or None])
            sample_mid.append(fragment_result[-out_degree or len(fragment_result) :])

        sample_terminal = np.hstack(sample_terminal)
        sample_mid = np.hstack(sample_mid)

        assert set(sample_terminal).issubset({np.array(0), np.array(1)})
        assert set(sample_mid).issubset({np.array(-1), np.array(1)})

        f = classical_processing_fn(sample_terminal)

        if not -1 <= f <= 1:
            raise ValueError(
                "The classical processing function supplied must "
                "give output in the interval [-1, 1]"
            )
        
        sigma_s = np.prod(sample_mid)
        t_s = f * sigma_s
        c_s = np.prod([evals[s] for s in setting])
        K = len(sample_mid)
        expvals.append(8**K * c_s * t_s)

    means = qml.math.convert_like(np.mean(expvals), res0)
    stds = qml.math.convert_like(np.std(expvals, ddof=1)/np.sqrt(shots), res0)

    return means, stds

def pauli_cut(tape: QuantumTape, n_shots: int, device_name: str = 'default.qubit'):

    num_extra_wires = sum(len(op.wires) for op in tape if isinstance(op, qml.WireCut))
    wires = len(tape.wires) + num_extra_wires

    device = qml.device(device_name, wires=wires, shots=1)

    g = qcut.tape_to_graph(tape)
    qcut.replace_wire_cut_nodes(g)
    fragments, comm_graph = qcut.fragment_graph(g)
    fragment_tapes = [qcut.graph_to_tape(f) for f in fragments]
    fragment_tapes = [qcut.remap_tape_wires(t, device.wires) for t in fragment_tapes]

    configurations, settings = qcut.expand_fragment_tapes_mc(
        fragment_tapes, comm_graph, shots=n_shots
    )

    tapes = tuple(tape for c in configurations for tape in c)
    samples = qml.execute(tapes, device, cache=False, gradient_fn=None)

    return samples, comm_graph, settings

def pauli_estimator(obs, samples, comm_graph, settings, n_shots):
    n_samples_in = len(samples) // len(comm_graph)
    samples_split = [samples[i: i+n_samples_in] for i in range(0, len(samples), n_samples_in)]
    samples_cut = [s[:n_shots] for s in samples_split]
    samples_recombined = [sample for samples_frag in samples_cut for sample in samples_frag]
    settings = settings[:,:n_shots]
    return qcut_processing_fn_mc(samples_recombined, comm_graph, settings, n_shots, obs)