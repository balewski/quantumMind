#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
QPE for a synthetic multi-controlled Z-with-phase unitary U.

What it does:
- Prepares a computational-basis state on `--nq_system` that satisfies the MCZ control condition.
- Runs QPE with `--nq_phase` phase qubits.
- Two implementations of U^(2^k):
  * --usePower: scale phase by 2^k once per k
  * default: repeat the same base gate 2^k times
- Prints timing and count stats, predicts the expected n-bit key, and plots a histogram with the expected key highlighted.

Phase conventions and units:
- `--true_phase` is φ in radians.
- QPE measures the unitless fraction x = φ / (2π) mod 1; printed "true" and "reco" are fractions of 2π.
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import time

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCPhaseGate
from qiskit.visualization import plot_histogram



def plot_counts(ax, counts, expected_key):
    title = f'QPE counts (expected: {expected_key})'
    # Use deterministic order so we can locate the expected_key on the x-axis
    plot_histogram(counts, ax=ax, title=title, sort='asc')
    labels = [t.get_text() for t in ax.get_xticklabels()]
    idx = labels.index(expected_key)
    xticks = ax.get_xticks()
    xpos = xticks[idx]
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=xpos, ymin=ymin, ymax=ymax, colors='red', linestyles='dashed')

# -------------------------
# 4. Build QPE Circuit
# -------------------------
def qpe_circuit(true_phase,nq_phase,qc_ini,usePower,nq_system ):
    """
    Build the QPE circuit.
    nq_phase: number of phase qubits
    """
    
    qc = QuantumCircuit(nq_phase + nq_system, nq_phase)
    
    # Step 1: Hadamards on phase register
    qc.h(range(nq_phase))
    
    # Step 2: Prepare eigenstate in system register
    qc.append(qc_ini, range(nq_phase, nq_phase + nq_system))

    #  assume U is a multi-controlled Z with phase true_phase acting on the system
    #  QPE applies controlled-U^(2^k): either scale phase by 2^k (usePower)
    #  or repeat the same gate 2^k times (not usePower)
    for k in range(nq_phase):
        # Use all system qubits; make the last provided qubit the target
        system_qubits = list(range(nq_phase, nq_phase + nq_system))
        controls_system = system_qubits[1:] if nq_system > 1 else []
        target_qubits = controls_system + [system_qubits[0]]
        if usePower:
            angle = true_phase * (1 << k)
            cevo = MCPhaseGate(angle, num_ctrl_qubits=nq_system)
            qc.append(cevo, [k] + target_qubits)
        else:
            angle = true_phase
            cevo = MCPhaseGate(angle, num_ctrl_qubits=nq_system)
            for _ in range(1 << k):
                qc.append(cevo, [k] + target_qubits)

    
    
    # Step 4: Inverse QFT on phase register
    qft_gate = QFTGate(nq_phase).inverse()
    qc.append(qft_gate, range(nq_phase))

    # Step 5: Measurement of phase register
    for i in range(nq_phase):
        qc.measure(i , i)

    return qc

def eval_phase(countD,args):

    phase = args.true_phase / (2 * np.pi)  # phase fraction in [0,1) units of 2π
    xtrue= phase % 1.0

    nqp=args.nq_phase
    max_key, max_value = max(countD.items(), key=lambda x: x[1])
    print('num keys:',len(countD),max_key, max_value)
    prob=max_value/args.shots
    ikey=int(max_key,2)
    xrec=ikey/2**nqp
    xerr=1/2**nqp
    # Predict the n-bit key that should be most likely (nearest integer to xtrue*2^nqp)
    ipred = int((xtrue * (2**nqp)) + 0.5) % (2**nqp)
    kpred = format(ipred, '0%db' % nqp)
    print('key=%s ikey=%d  prob=%.3f X: true=%.3f reco=%.3f +/- %.3f expected_key~%s'%(max_key, ikey, prob,xtrue,xrec,xerr,kpred))
    return kpred


# -------------------------
# 6. Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-q',"--nq_phase", type=int, default=4, help="Number of phase qubits")
    parser.add_argument('-x',"--true_phase", type=float, default=1.28, help="phase value for CZ")
    parser.add_argument('-s',"--nq_system", type=int, default=2, help="Number of system qubits for U")
    parser.add_argument('-p',"--usePower", action='store_true', help="If set: angle*2^k; else: repeat gate 2^k times")

    parser.add_argument("--shots", type=int, default=10_000, help="Number of shots")
    args = parser.parse_args()
    print(vars(args))
   
    
    # 4. Prepare the Qiskit circuit for this state
    print("\nPreparing Qiskit circuit to generate the eigenstate")
    qc_ini = QuantumCircuit(args.nq_system)
    # Prepare a computational-basis eigenstate that picks up the phase:
    # set the target (system qubit 0) to |1>, and set all system controls to |1>
    for iq in range(args.nq_system):
        qc_ini.x(iq)
    qc_ini.name='eigen'

    print("Initial state:")
    print(qc_ini)

    # Build QPE circuit
    qc = qpe_circuit(args.true_phase,args.nq_phase,qc_ini,args.usePower,args.nq_system)
    print(qc); print(qc.count_ops())
    # Run the QPE algorithm
    print(f"\nRunning QPE with {args.shots} shots...")
    backend = AerSimulator()
    transpile_start = time.time()
    qcT = transpile(qc, backend)
    transpile_end = time.time()
    print('Transpile (QPE) elapsed=%.3f s' % (transpile_end - transpile_start))
    len2q=qcT.depth(filter_function=lambda x: x.operation.num_qubits == 2 )
    n2q_g=qcT.num_nonlocal_gates()
    print('Transpiled, ops:',qcT.count_ops(),'\n  num2q:',n2q_g,'len2q:',len2q)
    sampler = Sampler(backend)
    run_start = time.time()
    job = sampler.run([qcT], shots=args.shots)
    result = job.result()
    run_end = time.time()
    print('Sampler run+result elapsed=%.3f s' % (run_end - run_start))

    # Post-process and plot the results
    counts = result[0].data.c.get_counts()
    

    expected_key = eval_phase(counts, args)
    print('expected_key:',expected_key)
    # Plot histogram
    if 1:
        fig, ax = plt.subplots()
        plot_counts(ax, counts, expected_key)
        outF='out/qs%d_ek%s.png'%(args.nq_system,expected_key)
        plt.savefig(outF)
        print('saved:',outF)


if __name__ == '__main__':
    main()
  
