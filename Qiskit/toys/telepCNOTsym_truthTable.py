#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
CNOT teleportation truth table

Computes 4x4 truth table for CNOT teleportation:
- Input states: 00, 01, 10, 11 (computational basis only)
- Output: probability distribution over 4 possible output states
- Supports stacking multiple CNOT teleportations (CNOT^n)

Usage:
  ./telepCNOTsym_truthTable.py --nshot 50000 --backendType 0
  ./telepCNOTsym_truthTable.py --stack 2 --nshot 10000
  ./telepCNOTsym_truthTable.py --draw 10 --stack 3 --nshot 10000

Backend types:
  0 = AerSimulator (ideal, perfect)
  1 = FakeTorino (medium noise)
  2 = FakeCusco (high noise)

Register naming:
  qctr: control qubit
  qtrg: target qubit
  anc0, anc1, ...: ancilla pairs for each teleportation
  ab0, ab1, ...: classical registers for teleportation measurements
  ct: final measurement register
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeCusco
import matplotlib.pyplot as plt
import argparse
import time

def create_cnot_teleport_circuit(inp_ctrl, inp_targ, n_stack=1):
    """
    Create stacked CNOT teleportation circuit with computational basis input
    inp_ctrl, inp_targ: 0 or 1 for input computational basis state
    n_stack: number of CNOT teleportations to stack (default=1)
    """
    # Calculate total qubits needed: 2 (ctrl, trg) + 2*n_stack (ancillas)
    n_qubits = 2 + 2 * n_stack
    
    # Create control and target registers
    qctr = QuantumRegister(1, 'qctr')
    qtrg = QuantumRegister(1, 'qtrg')
    
    # Create ancilla registers for each teleportation
    anc_regs = [QuantumRegister(2, f'anc{i}') for i in range(n_stack)]
    
    # Create classical registers for teleportation measurements
    c_regs = [ClassicalRegister(2, f'ab{i}') for i in range(n_stack)]
    
    # Create classical register for final measurements
    freg = ClassicalRegister(2, 'ct')
    
    # Build circuit with all registers
    qc = QuantumCircuit(qctr, qtrg, *anc_regs, *c_regs, freg)
    
    # Prepare input states in computational basis
    if inp_ctrl:
        qc.x(qctr[0])  # control qubit = |1>
    if inp_targ:
        qc.x(qtrg[0])  # target qubit = |1>
    qc.barrier()
    
    # Stack n_stack teleported CNOTs
    for k in range(n_stack):
        anc = anc_regs[k]
        creg = c_regs[k]
        
        # Create entangled pair between processors
        qc.h(anc[0])
        qc.cx(anc[0], anc[1])
        qc.barrier()
        
        # Teleport CNOT gate
        qc.cx(qctr[0], anc[0])
        qc.cx(anc[1], qtrg[0])
        
        qc.h(anc[1])
        qc.measure(anc[0], creg[0])
        qc.measure(anc[1], creg[1])
        
        with qc.if_test((creg[0], 1)):
            qc.x(qtrg[0])  # X correction
        
        with qc.if_test((creg[1], 1)):
            qc.z(qctr[0])  # Z correction
        qc.barrier()
    
    # Measure the qubits in computational basis
    qc.measure(qctr[0], freg[0]) 
    qc.measure(qtrg[0], freg[1]) 
    
    return qc

def compute_ideal_cnot_truth_table(n_stack=1):
    """Compute ideal CNOT truth table (4x4) for stacked CNOTs
    
    Note: CNOT^2 = Identity, so:
    - odd n_stack: equivalent to single CNOT
    - even n_stack: equivalent to Identity
    """
    # CNOT gate matrix
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
    
    # Identity matrix
    I = np.eye(4)
    
    # Apply stacked CNOTs (CNOT^n_stack)
    # Since CNOT^2 = I, we only need to check if n_stack is odd or even
    if n_stack % 2 == 1:
        gate = CNOT
    else:
        gate = I
    
    # Compute truth table: P(output|input)
    # Each row is an input state, each column is an output state
    truth_table = np.zeros((4, 4))
    
    for inp_state in range(4):
        # Create input state vector
        state_vec = np.zeros(4)
        state_vec[inp_state] = 1
        
        # Apply gate
        out_state_vec = gate @ state_vec
        
        # Output probabilities
        truth_table[inp_state, :] = np.abs(out_state_vec)**2
    
    return truth_table

def compute_measured_truth_table(backend, nshot, n_stack=1):
    """Run 4 circuits and compute measured truth table"""
    # Create circuits for all 4 input states
    circuits = []
    input_states = []
    
    for inp_ctrl in [0, 1]:
        for inp_targ in [0, 1]:
            qc = create_cnot_teleport_circuit(inp_ctrl, inp_targ, n_stack)
            circuits.append(qc)
            input_states.append((inp_ctrl, inp_targ))
    
    print(f'Created {len(circuits)} circuits for truth table (stack={n_stack})')
    
    # Transpile all circuits
    qcT = transpile(circuits, backend=backend, optimization_level=3, seed_transpiler=42)
    print(f'Transpiled circuits for {backend.name}')
    print(f'Number of qubits per circuit: {qcT[0].num_qubits}')
    
    # Run all circuits
    t0 = time.time()
    job = backend.run(qcT, shots=nshot)
    results = job.result()
    elapsed = time.time() - t0
    print(f'Simulation completed in {elapsed:.2f} seconds')
    
    # Process results to compute probability distribution
    truth_table = np.zeros((4, 4))
    
    for idx, (inp_ctrl, inp_targ) in enumerate(input_states):
        counts = results.get_counts(idx)
        
        # Sum over ancilla bits and compute output probabilities
        ct_counts = {}
        for bitstring, count in counts.items():
            parts = bitstring.split()
            ct_bits = parts[0]
            if ct_bits not in ct_counts:
                ct_counts[ct_bits] = 0
            ct_counts[ct_bits] += count
        
        # Convert to probabilities
        total_shots = sum(ct_counts.values())
        inp_idx = inp_ctrl * 2 + inp_targ
        
        for ct_bits, count in ct_counts.items():
            ctrl_bit = int(ct_bits[1])
            targ_bit = int(ct_bits[0])
            out_idx = ctrl_bit * 2 + targ_bit
            truth_table[inp_idx, out_idx] = count / total_shots
    
    print('Truth table measurement completed')
    return truth_table

def plot_truth_table(truth_table, outF, backendName, nshot, n_stack=1):
    """Plot the 4x4 CNOT truth table"""
    labels = ['00', '01', '10', '11']
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    im = ax.imshow(truth_table, cmap='Purples', vmin=0, vmax=1, aspect='auto')
    
    # Set ticks and labels on all 4 axes
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels)
    
    # Add axis labels
    ax.set_xlabel('Output state (ctr,trg)', fontsize=12)
    ax.set_ylabel('Input state (ctr,trg)', fontsize=12)
    
    # Add labels on top and right axes as well
    ax.tick_params(top=True, right=True, labeltop=True, labelright=True)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    # Add colorbar with more space from the plot
    cbar = plt.colorbar(im, ax=ax, pad=0.1)
    cbar.set_label('Probability', rotation=270, labelpad=15)
    
    # Add grid
    ax.set_xticks(np.arange(4) - 0.5, minor=True)
    ax.set_yticks(np.arange(4) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Add probability values as text in each cell
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{truth_table[i, j]:.3f}',
                          ha="center", va="center", color="lime", fontsize=10)
    
    stack_str = f'x{n_stack}' if n_stack > 1 else ''
    ax.set_title(f'Teleported CNOT{stack_str} Truth Table, shots = {nshot}, {backendName}', pad=10)
     
    plt.tight_layout()
    plt.savefig(outF)
    print(f'\nSaved plot: {outF}')
    plt.show()

def get_parser():
    parser = argparse.ArgumentParser(
        description='CNOT teleportation truth table',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-n','--nshot', type=int, default=1_000,
                        help='Number of shots for circuit execution')
    parser.add_argument('-b','--backendType', type=int, default=0, choices=[0, 1, 2],
                        help='Backend type: 0=ideal (AerSimulator), 1=FakeTorino, 2=FakeCusco')
    parser.add_argument('-s','--stack', type=int, default=1,
                        help='Number of CNOT teleportations to stack (CNOT^n)')
    parser.add_argument('-d','--draw', type=str, default=None,
                        help='Draw specific circuit for input state (e.g., 00, 01, 10, or 11)')
    
    return parser

def main(args):
    nshot = args.nshot
    n_stack = args.stack
    
    # Select backend based on backendType argument
    if args.backendType == 0:
        backend = AerSimulator()
        print('Backend: AerSimulator (ideal)')
    elif args.backendType == 1:
        backend = FakeTorino()
        print('Backend: FakeTorino')
    elif args.backendType == 2:
        backend = FakeCusco()
        print('Backend: FakeCusco')
    
    print(f'Stacking {n_stack} CNOT teleportation(s)')
    
    # Handle --draw option to show a specific circuit
    if args.draw is not None:
        inp_state_str = args.draw
        
        if len(inp_state_str) != 2 or not all(c in '01' for c in inp_state_str):
            print(f'Error: input state must be 2-bit binary string (00, 01, 10, or 11)')
            return
        
        inp_ctrl = int(inp_state_str[0])
        inp_targ = int(inp_state_str[1])
        
        print(f'\nCreating circuit for input state: {inp_state_str}')
        print(f'  Control qubit: |{inp_ctrl}>')
        print(f'  Target qubit:  |{inp_targ}>')
        
        qc = create_cnot_teleport_circuit(inp_ctrl, inp_targ, n_stack)
        print('\n' + qc.draw(output='text', fold=-1).__str__())
        print(f'\nCircuit depth: {qc.depth()}')
        print(f'Circuit ops: {qc.count_ops()}')
        print(f'Total qubits: {qc.num_qubits} (2 data + {2*n_stack} ancillas)')
        
        # Run the circuit on the selected backend
        print(f'\nRunning circuit on {backend.name} with {nshot} shots...')
        qcT = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42)
        print(f'Transpiled circuit depth: {qcT.depth()}')
        print(f'Transpiled circuit ops: {qcT.count_ops()}')
        print(f'Number of qubits: {qcT.num_qubits}')
        
        t0 = time.time()
        job = backend.run(qcT, shots=nshot)
        result = job.result()
        elapsed = time.time() - t0
        print(f'Simulation completed in {elapsed:.2f} seconds')
        
        counts = result.get_counts()
        
        # Sum over ab bits, only show ct bits
        ct_counts = {}
        for bitstring, count in counts.items():
            # bitstring format: 'ct ab0 ab1 ...' where ct are final measurements
            parts = bitstring.split()
            ct_bits = parts[0]
            if ct_bits not in ct_counts:
                ct_counts[ct_bits] = 0
            ct_counts[ct_bits] += count
        
        total_shots = sum(ct_counts.values())
        print(f'\nMeasurement results (summed over ancilla bits, shots={total_shots}):')
        print('Output state : Count    Probability  Stat Error')
        for ct_bits, count in sorted(ct_counts.items()):
            prob = count / total_shots
            # Calculate statistical error
            if prob > 0 and prob < 1:
                std_err = np.sqrt(prob * (1 - prob) / total_shots)
            else:
                std_err = 1.0 / total_shots
            print(f'{ct_bits:12s} : {count:5d}    {prob:.4f}      {std_err:.4f}')
        
        # Compare with ideal stacked CNOT
        if n_stack % 2 == 1:
            ideal_output = f'{inp_ctrl}{inp_ctrl ^ inp_targ}'  # CNOT truth table
            gate_name = 'CNOT'
        else:
            ideal_output = f'{inp_ctrl}{inp_targ}'  # Identity
            gate_name = 'Identity'
        print(f'\nIdeal output for {gate_name}({inp_state_str}): {ideal_output}')
        
        return
    
    # Compute ideal truth table
    print(f'\nComputing ideal truth table for CNOT^{n_stack}...')
    truth_ideal = compute_ideal_cnot_truth_table(n_stack)
    print('Ideal truth table computed')
    print('\nIdeal truth table:')
    print('  Input -> Output probabilities')
    for i, inp in enumerate(['00', '01', '10', '11']):
        print(f'  {inp}: {truth_ideal[i]}')
    
    # Compute measured truth table
    print(f'\nRunning 4 circuits for measured truth table...')
    truth_measured = compute_measured_truth_table(backend, nshot, n_stack)
    
    # Plot the measured truth table
    outF = f'out/cnot_truthTable_s{n_stack}_b{args.backendType}.png'
    plot_truth_table(truth_measured, outF, backend.name, nshot, n_stack)
    
    # Print comparison statistics
    diff = truth_measured - truth_ideal
    print(f'\nComparison with ideal:')
    print(f'  Mean difference: {np.mean(diff):.4f}')
    print(f'  Std difference: {np.std(diff):.4f}')
    print(f'  Max abs difference: {np.max(np.abs(diff)):.4f}')
    
    # Print measured truth table with statistical errors
    print(f'\nMeasured truth table (shots={nshot}):')
    print('Input -> Output probabilities [± stat error]')
    for i, inp in enumerate(['00', '01', '10', '11']):
        probs_str = []
        for j in range(4):
            prob = truth_measured[i, j]
            # Calculate statistical error
            if prob > 0 and prob < 1:
                std_err = np.sqrt(prob * (1 - prob) / nshot)
            else:
                std_err = 1.0 / nshot
            probs_str.append(f'{prob:.3f}±{std_err:.3f}')
        print(f'  {inp}: [{", ".join(probs_str)}]')

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

