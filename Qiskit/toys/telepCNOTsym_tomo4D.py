#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
CNOT teleportation with 4D tomography

Performs 16x16=256 measurements for all possible inputs and measurement bases.

Usage:
  ./telepCNOTsym_tomo4D.py --nshot 50000 --backendType 0

Backend types:
  0 = AerSimulator (ideal, perfect)
  1 = FakeTorino (medium noise)
  2 = FakeCusco (high noise)
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeCusco
import matplotlib.pyplot as plt
import argparse
import time

def add_measurement_basis(qc, qubit, basis):
    """Add rotation gates to measure in different Pauli bases"""
    if basis == 'X':
        qc.h(qubit)
    elif basis == 'Y':
        qc.sdg(qubit)
        qc.h(qubit)
    # Z basis needs no rotation (computational basis)
    return qc

def add_state_preparation(qc, qubit, basis):
    """Add gates to prepare qubit in eigenstate of given Pauli operator"""
    if basis == 'X':
        qc.h(qubit)
    elif basis == 'Y':
        qc.h(qubit)
        qc.s(qubit)
    elif basis == 'Z':
        pass  # Already in Z eigenstate (|0>)
    # I: do nothing, keep in |0> state
    return qc

def create_cnot_tomo_circuitV2(prep_ctrl, prep_targ, meas_ctrl, meas_targ):
    """
    Create CNOT teleportation circuit with tomography state prep and measurements
    prep_ctrl, prep_targ: 'I', 'X', 'Y', or 'Z' for input state preparation
    meas_ctrl, meas_targ: 'I', 'X', 'Y', or 'Z' for measurement basis
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    
    # Define quantum and classical registers
    qreg = QuantumRegister(4, 'q')
    creg = ClassicalRegister(2, 'ab')
    freg = ClassicalRegister(2, 'ct')
    qc = QuantumCircuit(qreg, creg, freg)
    
    # Prepare input states in specified bases
    add_state_preparation(qc, qreg[0], prep_ctrl)  # control qubit
    add_state_preparation(qc, qreg[3], prep_targ)  # target qubit
    qc.barrier()
    
    # Create entangled pair between processors
    qc.h(qreg[1])
    qc.cx(qreg[1], qreg[2])
    qc.barrier()
    
    # Teleport CNOT gate
    qc.cx(qreg[0], qreg[1])
    qc.cx(qreg[2], qreg[3])
    
    qc.h(qreg[2])
    qc.measure(qreg[1], creg[0])
    qc.measure(qreg[2], creg[1])
    
    with qc.if_test((creg[0], 1)):
        qc.x(qreg[3])  # X correction
    
    with qc.if_test((creg[1], 1)):
        qc.z(qreg[0])  # Z correction
    qc.barrier()
    
    # Add measurement basis rotations for control qubit (q0) and target qubit (q3)
    if meas_ctrl != 'I':
        add_measurement_basis(qc, qreg[0], meas_ctrl)
    if meas_targ != 'I':
        add_measurement_basis(qc, qreg[3], meas_targ)
    
    # Measure the qubits
    qc.measure(qreg[0], freg[1])  # control qubit to freg[1]
    qc.measure(qreg[3], freg[0])  # target qubit to freg[0]
    
    return qc

def compute_ideal_cnot_tomography():
    """Compute ideal CNOT gate tomography matrix (16x16)"""
    pauli_bases = ['I', 'X', 'Y', 'Z']
    n_bases = len(pauli_bases)
    
    # Pauli matrices
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    
    # CNOT gate matrix
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
    
    # Compute tomography matrix
    tomo_matrix = np.zeros((16, 16))
    
    for i, basis_in_ctrl in enumerate(pauli_bases):
        for j, basis_in_targ in enumerate(pauli_bases):
            for k, basis_out_ctrl in enumerate(pauli_bases):
                for l, basis_out_targ in enumerate(pauli_bases):
                    # Input operator
                    op_in = np.kron(pauli_dict[basis_in_ctrl], pauli_dict[basis_in_targ])
                    # Output operator
                    op_out = np.kron(pauli_dict[basis_out_ctrl], pauli_dict[basis_out_targ])
                    # Expectation value: Tr(CNOT @ op_in @ CNOT^dag @ op_out) / 4
                    val = np.trace(CNOT @ op_in @ CNOT.conj().T @ op_out) / 4
                    
                    row_idx = i * n_bases + j
                    col_idx = k * n_bases + l
                    tomo_matrix[row_idx, col_idx] = val.real
    
    return tomo_matrix

def compute_measured_tomography(backend, nshot):
    """Run all 256 circuits and compute measured tomography matrix"""
    pauli_bases = ['I', 'X', 'Y', 'Z']
    n_bases = len(pauli_bases)
    
    # Create all circuits for all prep and measurement basis combinations
    circuits = []
    circuit_indices = []
    
    for prep_ctrl in pauli_bases:
        for prep_targ in pauli_bases:
            for meas_ctrl in pauli_bases:
                for meas_targ in pauli_bases:
                    qc = create_cnot_tomo_circuitV2(prep_ctrl, prep_targ, meas_ctrl, meas_targ)
                    circuits.append(qc)
                    circuit_indices.append((prep_ctrl, prep_targ, meas_ctrl, meas_targ))
    
    print(f'Created {len(circuits)} circuits for tomography')
    print(f'Number of qubits per circuit: {circuits[0].num_qubits}') 
    # Transpile all circuits
    qcT = transpile(circuits, backend=backend, optimization_level=3, seed_transpiler=42)
    print(f'Transpiled circuits for {backend.name}')
   
    
    # Run all circuits
    t0 = time.time()
    job = backend.run(qcT, shots=nshot)
    results = job.result()
    elapsed = time.time() - t0
    print(f'Simulation completed in {elapsed:.2f} seconds')
    
    # Process results to compute expectation values
    tomo_matrix = np.zeros((16, 16))
    
    for idx, (prep_ctrl, prep_targ, meas_ctrl, meas_targ) in enumerate(circuit_indices):
        counts = results.get_counts(idx)
        
        # Compute joint expectation value <Z_ctrl ⊗ Z_targ>
        # After basis rotations, this measures <Pauli_ctrl ⊗ Pauli_targ>
        exp_val = 0
        
        for bitstring, count in counts.items():
            # bitstring format: 'ct ab' where ct are final measurements, ab are teleportation measurements
            parts = bitstring.split()
            ct_bits = parts[0]
            ctrl_bit = int(ct_bits[1])  # control is freg[1]
            targ_bit = int(ct_bits[0])  # target is freg[0]
            
            # Compute parity: (-1)^(ctrl_bit + targ_bit)
            parity = (-1) ** (ctrl_bit + targ_bit)
            exp_val += parity * count
        
        exp_val /= sum(counts.values())
        
        # Determine matrix indices
        prep_idx = pauli_bases.index(prep_ctrl) * n_bases + pauli_bases.index(prep_targ)
        meas_idx = pauli_bases.index(meas_ctrl) * n_bases + pauli_bases.index(meas_targ)
        
        tomo_matrix[prep_idx, meas_idx] = exp_val
        
    print('Tomography measurement completed')
    return tomo_matrix

def plot_tomography_matrix(tomo_matrix, outF, backendName, nshot):
    """Plot the 16x16 tomography correlation matrix"""
    pauli_bases = ['I', 'X', 'Y', 'Z']
    labels = [f'{b1}{b2}' for b1 in pauli_bases for b2 in pauli_bases]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    im = ax.imshow(tomo_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels on all 4 axes
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels)
    
    # Add axis labels
    ax.set_xlabel('Output', fontsize=12)
    ax.set_ylabel('Input', fontsize=12)
    
    # Add labels on top and right axes as well
    ax.tick_params(top=True, right=True, labeltop=True, labelright=True)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    # Add colorbar with more space from the plot
    cbar = plt.colorbar(im, ax=ax, pad=0.1)
    cbar.set_label('Component value', rotation=270, labelpad=15)
    
    # Add grid
    ax.set_xticks(np.arange(16) - 0.5, minor=True)
    ax.set_yticks(np.arange(16) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    ax.set_title(f'CNOT Teleport, shots = {nshot}, {backendName}', pad=10)
     
    plt.tight_layout()
    plt.savefig(outF)
    print(f'\nSaved plot: {outF}')
    plt.show()

def get_parser():
    parser = argparse.ArgumentParser(
        description='CNOT teleportation with 4D tomography',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-n','--nshot', type=int, default=10_000,
                        help='Number of shots for circuit execution')
    parser.add_argument('-b','--backendType', type=int, default=0, choices=[0, 1, 2],
                        help='Backend type: 0=ideal (AerSimulator), 1=FakeTorino, 2=FakeCusco')
    parser.add_argument('-d','--draw', type=str, nargs=2, default=None,
                        help='Draw specific circuit: two basis strings (e.g., xi zz) for input and output')
    
    return parser

def main(args):
    nshot = args.nshot
    
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
    
    # Handle --draw option to show a specific circuit
    if args.draw is not None:
        inp_basis_str = args.draw[0].upper()
        out_basis_str = args.draw[1].upper()
        
        if len(inp_basis_str) != 2 or len(out_basis_str) != 2:
            print(f'Error: basis strings must be 2 characters each (e.g., XI ZZ)')
            return
        
        prep_ctrl = inp_basis_str[0]
        prep_targ = inp_basis_str[1]
        meas_ctrl = out_basis_str[0]
        meas_targ = out_basis_str[1]
        
        valid_bases = ['I', 'X', 'Y', 'Z']
        if prep_ctrl not in valid_bases or prep_targ not in valid_bases or \
           meas_ctrl not in valid_bases or meas_targ not in valid_bases:
            print(f'Error: basis must be one of I, X, Y, Z')
            return
        
        print(f'\nCreating circuit for:')
        print(f'  Input basis:  {prep_ctrl}{prep_targ} (control={prep_ctrl}, target={prep_targ})')
        print(f'  Output basis: {meas_ctrl}{meas_targ} (control={meas_ctrl}, target={meas_targ})')
        
        qc = create_cnot_tomo_circuitV2(prep_ctrl, prep_targ, meas_ctrl, meas_targ)
        print('\n' + qc.draw(output='text', fold=-1).__str__())
        print(f'\nCircuit depth: {qc.depth()}')
        print(f'Circuit ops: {qc.count_ops()}')
        
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
            # bitstring format: 'ct ab' where ct are final measurements, ab are teleportation measurements
            parts = bitstring.split()
            ct_bits = parts[0]
            if ct_bits not in ct_counts:
                ct_counts[ct_bits] = 0
            ct_counts[ct_bits] += count
        
        print(f'\nMeasurement results (summed over ancilla bits):')
        print('ct bits : Count')
        for ct_bits, count in sorted(ct_counts.items(), key=lambda x: x[1], reverse=True):
            print(f'{ct_bits:7s} : {count:5d}')
        
        return
    
    # Compute ideal tomography matrix
    print('Computing ideal CNOT tomography matrix...')
    tomo_ideal = compute_ideal_cnot_tomography()
    print('Ideal matrix computed')
    
    # Compute measured tomography matrix
    print('\nRunning 256 circuits for measured tomography...')
    tomo_measured = compute_measured_tomography(backend, nshot)
    
    # Plot the measured tomography matrix
    outF = f'out/cnot_tomo_measured_b{args.backendType}.png'
    plot_tomography_matrix(tomo_measured, outF, backend.name, nshot)
    
    # Print comparison statistics
    diff = tomo_measured - tomo_ideal
    print(f'\nComparison with ideal:')
    print(f'  Mean difference: {np.mean(diff):.4f}')
    print(f'  Std difference: {np.std(diff):.4f}')
    print(f'  Max abs difference: {np.max(np.abs(diff)):.4f}')

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

