#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
Code Summary
This script is a Qiskit simulation of the 9-qubit Shor code, a quantum error-correcting
code capable of protecting against any single-qubit error (X, Z, or Y).
The program encodes a logical state into nine physical qubits, injects a user-defined error,
and then uses syndrome measurement to detect the error's type and location. The final
verification step checks if the decoder correctly identified the injected error.

Command-Line Arguments
--xq [QUBIT]: Specifies the qubit (0-8) to apply a bit-flip (X) error to.
--zq [QUBIT]: Specifies the qubit (0-8) to apply a phase-flip (Z) error to.
--initOne: Sets the initial logical state to |1_L>. Defaults to |0_L>.
'''

import argparse
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from pprint import pprint

# --- 1. Shor Code Circuit Component Functions ---

def get_encoder_circuit():
    """Returns a QuantumCircuit that encodes the logical state for the Shor code."""
    qr = QuantumRegister(9, name='q')
    qc = QuantumCircuit(qr, name='Encoder')
    qc.cx(0, 3)
    qc.cx(0, 6)
    qc.h([0, 3, 6])
    #qc.barrier()
    for i in [0, 3, 6]:
        qc.cx(i, i + 1)
        qc.cx(i, i + 2)
    qc.barrier(label='my err')
    return qc

def get_syndrome_circuit():
    """Returns a circuit for measuring the 8 stabilizers of the Shor code."""
    q = QuantumRegister(9, name='q')
    ax = QuantumRegister(6, name='ax')
    az = QuantumRegister(2, name='az')
    qc = QuantumCircuit(q, ax, az, name='Syndrome')

    # Part 1: Z-type stabilizers to detect X errors (using ax)
    qc.barrier(label='z-stab')
    qc.h(ax)  # Prepare ancillas in |+>
 
    # Measure stabilizers for X errors
    for i in range(3):
        block_start = i * 3
        anc_start = i * 2
        qc.cz(q[block_start], ax[anc_start])  # Z_i Z_i+1
        qc.cz(q[block_start + 1], ax[anc_start])  # Z_i Z_i+1
        qc.cz(q[block_start + 1], ax[anc_start + 1])  # Z_i+1 Z_i+2
        qc.cz(q[block_start + 2], ax[anc_start + 1])  # Z_i+1 Z_i+2
    #qc.barrier()
    
    # Apply Hadamards to undo the measurement basis change
    qc.h(ax)
    qc.barrier(label='x-stab')
    
    # Part 2: X-type stabilizers to detect Z errors (using az)
    qc.h(az)  # Prepare ancillas for X error detection
    # Stabilizer 1: X0X1X2 on az[0]
    for i in range(6): 
        qc.cx(az[0], q[i])
    # Stabilizer 2: X3X4X5X6X7X8 on az[1]
    for i in range(3, 9):
        qc.cx(az[1], q[i])
    qc.h(az)  # Measure ancillas during X type
    qc.barrier()

    return qc

# --- 2. The Shor Code Decoder Map ---

def build_decoder_map():
    """ Builds separate mappings for X and Z error syndromes. """
    x_err_map = {'000000': None}
    z_err_map = {'00': None}

    # Define the individual syndromes for each error type
    # X-error syndromes (detected by Z-type stabilizers, 6 bits)
    x_syndromes = {
        (0, 'X'): '000001', (1, 'X'): '000011', (2, 'X'): '000010',
        (3, 'X'): '000100', (4, 'X'): '001100', (5, 'X'): '001000',
        (6, 'X'): '010000', (7, 'X'): '110000', (8, 'X'): '100000'
    }

    # Populate X-error map
    for (qubit_idx, gate), x_syn_bits in x_syndromes.items():
        x_err_map[x_syn_bits] = (gate, qubit_idx)

    # Define Z error syndromes (detected by X-type stabilizers, 2 bits)
    # Map syndrome -> block index (0, 1, 2)
    z_syndromes = {
        '01': 0, # Z error in block 0 (qubits 0,1,2) -> s6=1, s7=0
        '11': 1, # Z error in block 1 (qubits 3,4,5) -> s6=1, s7=1
        '10': 2  # Z error in block 2 (qubits 6,7,8) -> s6=0, s7=1
    }

    # Populate Z-error map
    for z_syn_bits, block_idx in z_syndromes.items():
        z_err_map[z_syn_bits] = ('Z', block_idx * 3) # Use first qubit of block as loc

    return x_err_map, z_err_map

def init_state(qc, data_qubits, args):
    if args.initOne:
        if args.verb > 0: print("--- Initializing to logical state |1_L> ---")
        qc.x(data_qubits[0])
        qc.barrier()
    else:
        if args.verb > 0: print("--- Initializing to logical state |0_L> ---")

def evaluate_results(raw_syndrome, x_err_map, z_err_map, true_error_type, true_error_loc, verb=1):
    if verb > 0: print(f"\nMeasured Syndrome (raw): {raw_syndrome}")
    
    parts = raw_syndrome.split()
    if len(parts) == 2:
        sz_str, sx_str = parts
    else:
        # Fallback if registers merged differently
        if verb > 0: print("Warning: Unexpected syndrome format.")
        sz_str, sx_str = "00", "000000"

    x_corr = x_err_map.get(sx_str)
    z_corr = z_err_map.get(sz_str)
    
    # Combine corrections
    if x_corr is None and z_corr is None:
        decoded_type, decoded_loc = 'No error', None
    elif x_corr and z_corr is None:
        decoded_type, decoded_loc = x_corr
    elif z_corr and x_corr is None:
        decoded_type, decoded_loc = z_corr
    else:
        # Both X and Z errors detected
        x_type, x_loc = x_corr
        z_type, z_loc = z_corr 
        
        x_block = x_loc // 3
        z_block = z_loc // 3
        
        if x_block == z_block:
            decoded_type, decoded_loc = 'Y', x_loc
        else:
            decoded_type, decoded_loc = 'X+Z', [x_loc, z_loc]

    if verb > 0:
        print(f"Decoded Correction: Type='{decoded_type}', Location={decoded_loc}")
        print("\n--- Verifying Decoder Correctness ---")
        print(f"  Injected Error: ({true_error_type}, {true_error_loc})")
        print(f"  Decoded Error:  ({decoded_type}, {decoded_loc})")

    passed = False
    if true_error_type == 'No error':
        passed = (decoded_type == 'No error')
    elif true_error_type in ['X', 'Y']:
        passed = (true_error_type == decoded_type) and (true_error_loc == decoded_loc)
    elif true_error_type == 'Z':
        if decoded_type == 'Z' and decoded_loc is not None:
            true_block = true_error_loc // 3
            decoded_block = decoded_loc // 3
            passed = (true_block == decoded_block)
            if verb > 0: print(f"  (Verifying Z error in correct block: True Block={true_block}, Decoded Block={decoded_block})")
        else:
            passed = False

    if verb > 0:
        if passed:
            print("\n*** DECODER VERIFICATION PASSED ***")
        else:
            print("\n*** DECODER VERIFICATION FAILED ***")
            
    return passed

# --- 3. Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Shor's 9-Qubit Code Simulation")
    parser.add_argument('-x','--xq', type=int, default=1, help='Qubit (0-8) to apply an X error on.')
    parser.add_argument('-z','--zq', type=int, help='Qubit (0-8) to apply a Z error on.')
    parser.add_argument('-1','--initOne', action='store_true', help='Initialize the logical state to |1_L>.')
    parser.add_argument("-v", "--verb", type=int, help="Increase debug verbosity", default=1)
    args = parser.parse_args()

    true_error_type, true_error_loc = 'No error', None
    if args.xq is not None and args.zq is not None and args.xq == args.zq:
        true_error_type, true_error_loc = 'Y', args.xq
    elif args.xq is not None:
        true_error_type, true_error_loc = 'X', args.xq
    elif args.zq is not None:
        true_error_type, true_error_loc = 'Z', args.zq

    x_err_map, z_err_map = build_decoder_map()
    print('Mapping from classical syndrome to correction:'); 
    print(' X-error map (Z-stabilizers):'); pprint(x_err_map)
    print(' Z-error map (X-stabilizers):'); pprint(z_err_map)

    data_qubits = QuantumRegister(9, name='q')
    anc_qx = QuantumRegister(6, name='ax')
    anc_qz = QuantumRegister(2, name='az')
    synd_xbits = ClassicalRegister(6, name='sx')
    synd_zbits = ClassicalRegister(2, name='sz')
    qc = QuantumCircuit(data_qubits, anc_qx, anc_qz, synd_xbits, synd_zbits)

    init_state(qc, data_qubits, args)
    qc.append(get_encoder_circuit().to_instruction(), data_qubits)

    print(f"--- Injecting true error: Type='{true_error_type}', Location={true_error_loc} ---")
    if args.xq is not None: qc.x(args.xq)
    if args.zq is not None: qc.z(args.zq)

    qc.append(get_syndrome_circuit().to_instruction(), qc.qubits)
    qc.measure(anc_qx, synd_xbits)
    qc.measure(anc_qz, synd_zbits)
    if args.verb <= 1: print(qc)
    else: print(qc.decompose())

    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    result = backend.run(pm.run(qc), shots=1).result()
    counts = result.get_counts()
    
    # Handle multiple registers in counts: format "sz sx" (space separated)
    raw_syndrome = list(counts.keys())[0]
    
    evaluate_results(raw_syndrome, x_err_map, z_err_map, true_error_type, true_error_loc, args.verb)

if __name__ == "__main__":
    main()
