from typing import Callable, Optional
from pennylane import numpy as np

import pennylane as qml
from pennylane.tape import QuantumTape

def make_kraus_ops(num_wires: int):

    d = 2**num_wires

    kraus0 = np.identity(d**2).reshape(d**2, d, d)
    kraus0 = np.concatenate([kraus0, np.identity(d)[None,:,:]], axis=0)
    kraus0 /= np.sqrt(d+1)
    
    kraus1 = np.identity(d**2).reshape(d**2, d, d)
    kraus1 /= np.sqrt(d)

    return list(kraus0.astype(complex)), list(kraus1.astype(complex))

def split_tape(tape: QuantumTape):

    split = False
    has_more_cuts = False
    probs = (1.,1.)

    with QuantumTape(do_queue=False) as tape0, QuantumTape(do_queue=False) as tape1:

        for op in tape:

            if isinstance(op, qml.WireCut):

                if not split:
                    
                    k = len(op.wires)
                    d = 2**k
                    
                    K0, K1 = make_kraus_ops(k)
                    probs = (d+1)/(2*d+1), d/(2*d+1)

                    qml.apply(qml.QubitChannel(K0, wires=op.wires, do_queue=False, id=op.id), context=tape0)
                    qml.apply(qml.QubitChannel(K1, wires=op.wires, do_queue=False, id=op.id), context=tape1)

                    split = True
                    
                else:
                    has_more_cuts = True
                    qml.apply(op, context=tape0)
                    qml.apply(op, context=tape1)

            else:
                qml.apply(op, context=tape0)
                qml.apply(op, context=tape1)

    tapes = (tape0, tape1)

    return tapes, probs, has_more_cuts

def make_all_tapes(tape: QuantumTape):

    tapes = [tape]
    probs = [1.0]
    signs = [1]
    has_wire_cuts = True

    while has_wire_cuts:

        tapes_ = []
        probs_ = []
        signs_ = []

        for tape, prob, sign in zip(tapes, probs, signs):

            branch_tapes, (p0, p1), has_wire_cuts = split_tape(tape)

            tapes_.extend(branch_tapes)
            probs_.extend([prob*p0, prob*p1])
            signs_.extend([sign, -sign])

        tapes = tapes_
        probs = probs_
        signs = signs_

    return tapes, np.array(probs), np.array(signs)

def randomized_cut(tape: QuantumTape, n_shots: int, device_name: str = 'cirq.mixedsimulator', seed: Optional[int] = None):

    device = qml.device(device_name, wires=tape.wires)
    rng = np.random.default_rng(seed)
    ks = [len(op.wires) for op in tape if isinstance(op, qml.WireCut)]

    if len(ks) == 0:
        raise ValueError('No wire cuts found in tape!')

    samples = np.zeros((n_shots, len(tape.wires)), dtype=int)

    tapes, probs, signs = make_all_tapes(tape)
    choices = rng.choice(len(tapes), size=n_shots, p=probs)

    configs, shot_counts = np.unique(choices, return_counts=True)

    for i, ns, tp in zip(configs, shot_counts, tapes):
        device.shots = ns.item()
        shots, = qml.execute([tp], device=device, cache=False, gradient_fn=None)
        samples[choices==i] = shots

    return samples, (signs[choices], ks)

def estimator(obs: Callable, samples: np.ndarray, settings: dict, vectorized: bool = False, return_error: bool = False, **kwargs):

    samples = np.atleast_2d(samples)
    n_samples = samples.shape[0]

    if vectorized:
        fs = obs(samples, **kwargs)[:n_samples]
    else:
        fs = np.stack([obs(sample, **kwargs) for sample in samples])[:n_samples]

    signs, ks = settings

    signs = np.asarray(signs)[:n_samples]
    ds = 2**np.asarray(ks)

    rv = np.prod(2*ds+1) * signs * fs
    mean = np.mean(rv)
    
    if return_error:
        err = np.std(rv, ddof=1)/np.sqrt(n_samples)
        return mean, err
    else:
        return mean           