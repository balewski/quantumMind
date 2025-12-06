#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
Code Summary
This script is a Qiskit simulation of the 5-qubit phase-flip code, 
a quantum error-correcting code designed specifically to protect against phase-flip (Z) errors.
The logic mirrors the 5-qubit bit-flip code but operates in the Hadamard basis.
The logical states are |0_L> = |+++++> and |1_L> = |----->.
'''

import argparse
import numpy as np
from itertools import combinations
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from pprint import pprint

# --- 1. Circuit Component Functions ---

def get_encoder_circuit():
    """Returns a QuantumCircuit that encodes the logical state for phase-flip code."""
    qr = QuantumRegister(5, name='q')
    qc = QuantumCircuit(qr, name='Encoder')
    # Standard repetition code encoding
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 4)
    # Move to X-basis for phase-flip protection
    qc.h(qr)
    qc.barrier()
    return qc

def get_syndrome_circuit():
    """Returns a circuit for measuring the 4 stabilizers: X0X1, X1X2, X2X3, X3X4."""
    q = QuantumRegister(5, name='q')
    a = QuantumRegister(4, name='a')
    qc = QuantumCircuit(q, a, name='Syndrome')
    
    # Basis change: H on data to map Z errors to X errors
    qc.h(q)
    
    # Measure Z-parity in this new basis (detecting original Z-flips)
    qc.cx(q[0], a[0])
    qc.cx(q[1], a[0])
    qc.cx(q[1], a[1])
    qc.cx(q[2], a[1])
    qc.cx(q[2], a[2])
    qc.cx(q[3], a[2])
    qc.cx(q[3], a[3])
    qc.cx(q[4], a[3])
    
    # Basis change back (optional, but keeps logical info consistent if we continue)
    qc.h(q)
    qc.barrier()
    return qc

# --- 2. The Decoder Map ---

def build_decoder_map():
    """Builds the mapping from a 4-bit syndrome string to the required correction."""
    decoder_map = {}
    # Syndrome mapping logic is identical to bit-flip code
    syndrome_to_int = {0: 0b0001, 1: 0b0011, 2: 0b0110, 3: 0b1100, 4: 0b1000}
    decoder_map['0000'] = ('No error', None)
    for qubit_idx, syndrome_int in syndrome_to_int.items():
        decoder_map[f'{syndrome_int:04b}'] = ('Z', qubit_idx)
    for q1, q2 in combinations(range(5), 2):
        combined_syndrome_int = syndrome_to_int[q1] ^ syndrome_to_int[q2]
        decoder_map[f'{combined_syndrome_int:04b}'] = ('Z', [q1, q2])
    return decoder_map

def init_state(qc, data_qubits, args):
    """Initializes the state of the data qubits based on arguments."""
    if args.initState:
        alpha, beta = args.initState
        norm = np.sqrt(alpha**2 + beta**2)
        initial_state = [alpha / norm, beta / norm]
        print(f"--- Initializing qubit 0 to {initial_state[0]:.3f}|0> + {initial_state[1]:.3f}|1> ---")
        qc.initialize(initial_state, data_qubits[0])
        qc.barrier()
    elif args.initOne:
        print("--- Initializing to logical state |1_L> ---")
        qc.x(data_qubits[0])
        qc.barrier()
    else:
        print("--- Initializing to logical state |0_L> ---")

# --- 3. Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description='5-Qubit Phase-Flip Code Simulation (2-Error Correcting)')
    parser.add_argument('-z', '--zq', type=int, nargs='+', default=[1], help='List of qubits to apply Z error on (e.g., -z 1 3)')
    parser.add_argument('-1', '--initOne', action='store_true', help='Initialize the logical state to |1_L>')
    parser.add_argument('-i','--initState', type=float, nargs=2, metavar=('ALPHA', 'BETA'), help='Initialize qubit 0 to alpha|0> + beta|1>')
    parser.add_argument("-v", "--verb", type=int, help="Increase debug verbosity", default=1)
    args = parser.parse_args()

    assert not (args.initOne and args.initState), "Error: Cannot use both --initOne and --initState simultaneously!"
    
    decoder_map = build_decoder_map()
    print('Mapping from classical syndrome to correction:\n syndrom:  action'); pprint(decoder_map)

    data_qubits = QuantumRegister(5, name='q')
    anc_qubits = QuantumRegister(4, name='a')
    synd_bits = ClassicalRegister(4, name='s')
    qc = QuantumCircuit(data_qubits, anc_qubits, synd_bits)

    # Step 1: Prepare and Encode
    init_state(qc, data_qubits, args)
    qc.append(get_encoder_circuit().to_instruction(), data_qubits)
   
    # Step 2: Inject errors
    z_error_qubits = args.zq
    if z_error_qubits:
        print(f"--- Injecting Z error(s) on qubit(s): {z_error_qubits} ---")
        for q in z_error_qubits:
            if 0 <= q < 5:
                qc.z(q)
            else:
                print(f"Warning: Qubit {q} out of range (0-4), skipping.")
        qc.barrier()

    # Step 3: Measure syndromes
    qc.append(get_syndrome_circuit().to_instruction(), qc.qubits)
    qc.measure(anc_qubits, synd_bits)
    if args.verb <= 1: print(qc)
    else: print(qc.decompose())

    # Step 4: Simulate and Decode
    simulator = AerSimulator()
    pm = generate_preset_pass_manager(backend=simulator, optimization_level=1)
    result = simulator.run(pm.run(qc), shots=1).result()
    counts = result.get_counts()
    measured_syndrome = list(counts.keys())[0]
    print(f"\nMeasured Syndrome: {measured_syndrome}")

    correction = decoder_map.get(measured_syndrome)
    gate, detected_qubits = (None, None) if not correction else correction
    
    if not correction:
        print("Error: Syndrome not found in decoder map! Cannot correct.")
    else:
        print(f"Decoded Correction: Apply '{gate}' gate to qubit(s) {detected_qubits}")

    # --- 5. Decoder Verification ---
    print("\n--- Verifying Decoder Correctness ---")
    
    # Standardize the injected error locations for comparison by sorting them.
    injected_errors_sorted = sorted(z_error_qubits) if z_error_qubits else []
    
    # Standardize the decoded error locations.
    decoded_errors_sorted = []
    if detected_qubits is not None:
        if isinstance(detected_qubits, list):
            decoded_errors_sorted = sorted(detected_qubits)
        else: # It's a single integer for a one-qubit error
            decoded_errors_sorted = [detected_qubits]

    print(f"Injected error location(s): {injected_errors_sorted if injected_errors_sorted else 'None'}")
    print(f"Decoded error location(s):  {decoded_errors_sorted if decoded_errors_sorted else 'None'}")
    
    # Compare the standardized lists.
    if injected_errors_sorted == decoded_errors_sorted:
        print("\n*** DECODER VERIFICATION PASSED ***")
    else:
        print("\n*** DECODER VERIFICATION FAILED ***")

if __name__ == "__main__":
    main()
