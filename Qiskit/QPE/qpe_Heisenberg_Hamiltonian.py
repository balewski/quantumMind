#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
"""
Quantum Phase Estimation (QPE) for a Heisenberg-type Hamiltonian.

Features:
- Builds a Hamiltonian of size `--nq_system` using `prep_Heisenberg_Hamiltonian.py`.
- Selects an eigenstate (by `--evIdx`) and prepares it as an initial system state.
- Runs QPE with `--nq_phase` phase qubits and trotterized time evolution (Suzuki-Trotter reps).
- Two QPE implementations are supported: repeat U 2^k times (fixed time per layer).
- Outputs timing info, counts, predicted most-likely bitstring, and a histogram PNG tagged with the expected key.

Key args:
- `-s/--nq_system` number of system qubits; `-q/--nq_phase` number of phase qubits
- `-t/--time` evolution time per base layer; `--trotterSteps` Suzuki-Trotter repetitions
- `--shots` sampler shots; `--isEigen` verifies eigenstate; `-v/--verb` printing level
"""
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
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.visualization import plot_histogram
from prep_Heisenberg_Hamiltonian import create_heisenberg_hamiltonian, compute_eigenstates, prepare_eigenstate_circuit, check_if_eigenstate



# -------------------------
# 4. Build QPE Circuit
# -------------------------
def qpe_circuit(nq_phase, hamiltonian,qc_initState,time,trotterSteps):
    """
    Build the QPE circuit.
    nq_phase: number of phase qubits
    hamiltonian: SparsePauliOp 
    time: evolution time per base layer
    """
    
    nq_system = hamiltonian.num_qubits
    
    qc = QuantumCircuit(nq_phase + nq_system, nq_phase)
    
    # Step 1: Hadamards on phase register
    qc.h(range(nq_phase))

    # Step 2: Prepare eigenstate in system register
    qc.append(qc_initState, range(nq_phase, nq_phase + nq_system))

    # Step 3: Apply controlled-U^(2^k)
    #  Apply controlled exp(-i * time * 2^k * H) on target_qubits controlled by control_qubit.
    from qiskit.synthesis import SuzukiTrotter

    # Define the Trotterization strategy with the desired number of repetitions
    trotter_synth = SuzukiTrotter(reps=trotterSteps)

    for k in range(nq_phase):
        target_qubits=list(range(nq_phase, nq_phase + nq_system))
        evo = PauliEvolutionGate(hamiltonian, time=time , synthesis=trotter_synth)
        for _ in range(1<<k):
            qc.append(evo.control(1), [k] + target_qubits)
   

    
        
    # Step 4: Inverse QFT on phase register
    qft_gate = QFTGate(nq_phase).inverse()
    qc.append(qft_gate, range(nq_phase))

    # Step 5: Measurement of phase register
    for i in range(nq_phase):
        qc.measure(i , i)

    return qc

def eval_phase(countD,args,eigenvalue):
    phase = (-args.time  * eigenvalue) / (2 * np.pi)  # normalized phase fraction
    xtrue= phase % 1.0

    nqp=args.nq_phase
    max_key, max_value = max(countD.items(), key=lambda x: x[1])
    print('num keys:',len(countD),max_key, max_value)
    prob=max_value/args.shots
    ikey=int(max_key,2)
    xrec=ikey/2**nqp
    xerr=1/2**nqp
    ipred = int((xtrue * (2**nqp)) + 0.5) % (2**nqp)
    kpred = format(ipred, '0%db' % nqp)
    print('key=%s ikey=%d  prob=%.3f X: true=%.3f reco=%.3f +/- %.3f expected_key~%s'%(max_key, ikey, prob,xtrue,xrec,xerr,kpred))
    return kpred

def plot_counts(ax, counts, expected_key):
    title = f'QPE counts (expected: {expected_key})'
    plot_histogram(counts, ax=ax, title=title, sort='asc')
    labels = [t.get_text() for t in ax.get_xticklabels()]
    idx = labels.index(expected_key)
    xticks = ax.get_xticks()
    xpos = xticks[idx]
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=xpos, ymin=ymin, ymax=ymax, colors='red', linestyles='dashed')


# -------------------------
# 6. Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',"--nq_system", type=int, default=3, help="Number of qubits in the Hamiltonian (system)")
    parser.add_argument('-q',"--nq_phase", type=int, default=5, help="Number of phase qubits")
    parser.add_argument("--evIdx", type=int, default=0, help="Index of the eigenstate to prepare (0 for ground state).")

    parser.add_argument("--isEigen", action='store_true', help="Check if the initial state is an eigenstate")
    parser.add_argument('-t',"--time", type=float, default=1.2, help="evolution time (a.u.)")
    
    parser.add_argument("--shots", type=int, default=10_000, help="Number of shots")
    parser.add_argument("--trotterSteps", type=int, default=10, help="Number of Trotter steps")
    parser.add_argument('-v',"--verb", type=int, default=1, help="verbosity level; >1 prints full circuit")
    args = parser.parse_args()
    print(vars(args))
  

   # 1. Create Hamiltonian
    try:
        hamiltonian = create_heisenberg_hamiltonian(args.nq_system)
        print("Hamiltonian created:\n", hamiltonian)
    except NotImplementedError as e:
        print(f"Error: {e}")
        return

    # 2. Compute eigenstates
    eigenvalues, eigenvectors = compute_eigenstates(hamiltonian)
    print('all %d eigenvalues:'%(len(eigenvalues)),eigenvalues)
    
    # 3. Select the desired eigenstate
    if not (0 <= args.evIdx < len(eigenvalues)):
        print(f"Error: --evIdx must be between 0 and {len(eigenvalues) - 1}. You provided {args.evIdx}.")
        return
        
    selected_eigenvalue = eigenvalues[args.evIdx]
    # In numpy's eigh, eigenvectors are stored as columns in the matrix
    selected_eigenvector = eigenvectors[:, args.evIdx]

    print(f"\nSelected eigenstate (index {args.evIdx}):")
    print(f"  - Eigenvalue (Energy): {selected_eigenvalue.real:.6f}")
    
    # 4. Prepare the Qiskit circuit for this state
    print("\nPreparing Qiskit circuit to generate the eigenstate from |0...0>...")
    qc_initState = prepare_eigenstate_circuit(selected_eigenvector)
    
    # 5. Display results
    print("Circuit diagram:")
    print(qc_initState)

    qcT_start = time.time()
    qcT = transpile(qc_initState, basis_gates=['u','cx'])
    qcT_end = time.time()
    

    if args.isEigen:
        # Final check using the transpiled circuit and the Hamiltonian
        check_if_eigenstate(hamiltonian, qcT)
        exit(0)
     
    # Build QPE circuit
    qc = qpe_circuit(args.nq_phase, hamiltonian,qc_initState,args.time,args.trotterSteps)
    if args.verb>1:
        print(qc)
    print(qc.count_ops())
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
    #pprint(counts)

    expected_key = eval_phase(counts, args, selected_eigenvalue )
    print('expected_key:',expected_key)
    # Plot histogram
    if 1:
        fig, ax = plt.subplots()
        plot_counts(ax, counts, expected_key)
        outF='out/hs%d_ek%s.png'%(hamiltonian.num_qubits,expected_key)
        plt.savefig(outF)
        print('saved:',outF)


if __name__ == '__main__':
    main()
  
