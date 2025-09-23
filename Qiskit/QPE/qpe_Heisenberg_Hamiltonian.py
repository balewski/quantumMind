#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
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
from prep_Heisenberg_Hamiltonian import create_heisenberg_hamiltonian, compute_eigenstates, prepare_eigenstate_circuit, check_if_eigenstate



# -------------------------
# 4. Build QPE Circuit
# -------------------------
def qpe_circuit(nq_phase, hamiltonian,qc_initState,time,trotterSteps):
    """
    Build the QPE circuit.
    nq_phase: number of phase qubits
    hamiltonian: SparsePauliOp 
    theta: evolution time scaling
    init_state_label: eigenstate label for last 3 qubits
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

    ''' 1 c-U per phase qubit
    for k in range(nq_phase):
        evo = PauliEvolutionGate(hamiltonian, time=time * (2 ** k), synthesis=trotter_synth)
        target_qubits=list(range(nq_phase, nq_phase + nq_system))
        qc.append(evo.control(1), [k] + target_qubits)
    '''

    # fixed trotter steps per time
    for k in range(nq_phase):
        nrep=1<<k
        for j in range(nrep):
            evo = PauliEvolutionGate(hamiltonian, time=time , synthesis=trotter_synth)
            cevo=evo.control(1)
            target_qubits=list(range(nq_phase, nq_phase + nq_system))            
            qc.append(cevo, [k] + target_qubits)

    #print(qc)  ;aa  
        
    # Step 4: Inverse QFT on phase register
    qft_gate = QFTGate(nq_phase).inverse()
    qc.append(qft_gate, range(nq_phase))

    # Step 5: Measurement of phase register
    for i in range(nq_phase):
        qc.measure(i , i) #nq_phase-1-i)

    return qc

def eval_phase(countD,args,eigenvalue):
    phase = (-args.time  * eigenvalue) / (2 * np.pi)  # normalized phase in [0,1)
    xtrue= phase % 1.0

    nqp=args.nq_phase
    max_key, max_value = max(countD.items(), key=lambda x: x[1])
    print('num keys:',len(countD),max_key, max_value)
    prob=max_value/args.shots
    ikey=int(max_key,2)
    xrec=ikey/2**nqp
    xerr=1/2**nqp
    print('key=%s ikey=%d  prob=%.3f X: true=%.3f reco=%.3f +/- %.3f'%(max_key, ikey, prob,xtrue,xrec,xerr))


# -------------------------
# 6. Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nq_ham", type=int, default=3, help="Number of qubits in the Hamiltonian (default: 3).")
    parser.add_argument("--nq_phase", type=int, default=3, help="Number of phase qubits")
    parser.add_argument("--evIdx", type=int, default=0, help="Index of the eigenstate to prepare (0 for ground state).")

    parser.add_argument("--isEigen", action='store_true', help="Check if the initial state is an eigenstate")
    parser.add_argument("--time", type=float, default=1.5, help="evolution time (a.u.)")
    parser.add_argument("--shots", type=int, default=10_000, help="Number of shots")
    parser.add_argument("--trotterSteps", type=int, default=200, help="Number of Trotter steps")
    args = parser.parse_args()
    print(vars(args))
  

   # 1. Create Hamiltonian
    try:
        hamiltonian = create_heisenberg_hamiltonian(args.nq_ham)
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
    print('Transpile (initState) elapsed=%.3f s' % (qcT_end - qcT_start))
    print(qcT.draw())

    if args.isEigen:
        # Final check using the transpiled circuit and the Hamiltonian
        check_if_eigenstate(hamiltonian, qcT)
        exit(0)
     
    # Build QPE circuit
    qc = qpe_circuit(args.nq_phase, hamiltonian,qc_initState,args.time,args.trotterSteps)
    print(qc); print(qc.count_ops())
    # Run the QPE algorithm
    print(f"\nRunning QPE with {args.shots} shots...")
    backend = AerSimulator()
    transpile_start = time.time()
    qcT = transpile(qc, backend,basis_gates=['u','cz'])
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
    pprint(counts)

    eval_phase(counts, args, selected_eigenvalue )
    # Plot histogram
    if 0:
        fig, ax = plt.subplots()
        plot_counts(ax, counts, args.nq_phase, args.theta)
        plt.show()


if __name__ == '__main__':
    main()
  
