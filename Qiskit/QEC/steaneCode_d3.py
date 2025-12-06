#!/usr/bin/env python3
__author__ = "Gemini, based on Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
Code Summary
This script is a Qiskit simulation of the 7-qubit Steane code (distance d=3).
It implements a fully corrected Encoder that aligns with the Standard Hamming
Parity Matrix used by the Decoder.
'''

import argparse
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from collections import Counter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# --- 1. Global Code Parameters ---
DISTANCE = 3
NUM_QUBITS = 7
NUM_ANC_X = 3 
NUM_ANC_Z = 3 
assert DISTANCE == 3

# --- 2. The Hamming Code Decoder ---
def build_hamming_decoder_map():
    """
    Builds a lookup table for the [7,4,3] Hamming code.
    Maps a 3-bit syndrome string to the single qubit location (0-6).
    """
    decoder_map = {}
    decoder_map['000'] = []

    # Map based on Syndrome Indices S0, S1, S2
    # S0: 3,4,5,6 | S1: 1,2,5,6 | S2: 0,2,4,6
    # We build the map dynamically to ensure it matches the circuit logic exactly.
    for i in range(NUM_QUBITS):
        s0 = 1 if i in [3, 4, 5, 6] else 0
        s1 = 1 if i in [1, 2, 5, 6] else 0
        s2 = 1 if i in [0, 2, 4, 6] else 0
        
        # This order (S0 S1 S2) matches the concatenation in decode_steane_syndrome
        synd_str = f"{s0}{s1}{s2}"
        decoder_map[synd_str] = [i]

    return decoder_map

def decode_steane_syndrome(full_syndrome):
    hamming_decoder = build_hamming_decoder_map()
    
    # full_syndrome comes in as 'Z_syndrome X_syndrome'
    num_synd_bits = 3
    sz_full_str = full_syndrome[:num_synd_bits] # Contains results from Z-Stabilizers
    sx_full_str = full_syndrome[num_synd_bits:] # Contains results from X-Stabilizers

    # --- CORRECTION LOGIC ---
    # 1. Z-Stabilizers detect X-Errors (Bit flips)
    logical_x_errors = hamming_decoder.get(sz_full_str, "Unknown")
    
    # 2. X-Stabilizers detect Z-Errors (Phase flips)
    logical_z_errors = hamming_decoder.get(sx_full_str, "Unknown")
                
    return {'Z': logical_z_errors, 'X': logical_x_errors}


def get_encoder_circuit():
    """
    Returns a QuantumCircuit that encodes the logical state |0> for the Steane code.
    This specific circuit ensures the state satisfies the stabilizers:
    S0(3,4,5,6), S1(1,2,5,6), S2(0,2,4,6).
    """
    qr = QuantumRegister(NUM_QUBITS, name='q')
    qc = QuantumCircuit(qr, name='Encoder')
    
    # 1. Prepare superposition on the "generator" qubits (4, 5, 6)
    qc.h(qr[4])
    qc.h(qr[5])
    qc.h(qr[6])
    
    # 2. CNOT Cascade to enforce parity constraints
    # These connections are derived mathematically to satisfy H.x = 0
    
    # Q0 needs parity of 5,6 to satisfy S2 (0,2,4,6) & S1, S0
    qc.cx(qr[5], qr[0])
    qc.cx(qr[6], qr[0])

    # Q1 needs parity of 4,6
    qc.cx(qr[4], qr[1])
    qc.cx(qr[6], qr[1])
    
    # Q2 needs parity of 4,5
    qc.cx(qr[4], qr[2])
    qc.cx(qr[5], qr[2])

    # Q3 needs parity of 4,5,6
    qc.cx(qr[4], qr[3])
    qc.cx(qr[5], qr[3])
    qc.cx(qr[6], qr[3])
    
    qc.barrier()
    return qc

def get_syndrome_circuit():
    """Returns a circuit for measuring the stabilizers of the Steane code."""
    q = QuantumRegister(NUM_QUBITS, name='q')
    anc_qx = QuantumRegister(NUM_ANC_X, name='ax') # Z-stabilizers (detect X)
    anc_qz = QuantumRegister(NUM_ANC_Z, name='az') # X-stabilizers (detect Z)
    qc = QuantumCircuit(q, anc_qx, anc_qz, name='Syndrome')

    qc.barrier(label='Z-stab:')
    # --- Part 1: Measure Z-Stabilizers (Detect X Errors) ---
    # S0 (Index 0) -> 3,4,5,6
    qc.cx(q[3], anc_qx[0])
    qc.cx(q[4], anc_qx[0])
    qc.cx(q[5], anc_qx[0])
    qc.cx(q[6], anc_qx[0])
    
    # S1 (Index 1) -> 1,2,5,6
    qc.cx(q[1], anc_qx[1])
    qc.cx(q[2], anc_qx[1])
    qc.cx(q[5], anc_qx[1])
    qc.cx(q[6], anc_qx[1])

    # S2 (Index 2) -> 0,2,4,6
    qc.cx(q[0], anc_qx[2])
    qc.cx(q[2], anc_qx[2])
    qc.cx(q[4], anc_qx[2])
    qc.cx(q[6], anc_qx[2])
    
    qc.barrier(label='X-stab:')
    
    # --- Part 2: Measure X-Stabilizers (Detect Z Errors) ---
    qc.h(anc_qz)
    
    # S0 (Index 0) -> 3,4,5,6
    qc.cx(anc_qz[0], q[3])
    qc.cx(anc_qz[0], q[4])
    qc.cx(anc_qz[0], q[5])
    qc.cx(anc_qz[0], q[6])
    
    # S1 (Index 1) -> 1,2,5,6
    qc.cx(anc_qz[1], q[1])
    qc.cx(anc_qz[1], q[2])
    qc.cx(anc_qz[1], q[5])
    qc.cx(anc_qz[1], q[6])

    # S2 (Index 2) -> 0,2,4,6
    qc.cx(anc_qz[2], q[0])
    qc.cx(anc_qz[2], q[2])
    qc.cx(anc_qz[2], q[4])
    qc.cx(anc_qz[2], q[6])
    
    qc.h(anc_qz)
    qc.barrier()
    
    return qc


def evaluate_results(raw_syndrome_key, true_x_errors, true_z_errors, verb=1):
    if verb > 0: print(f"\nMeasured Syndrome (raw key): '{raw_syndrome_key}'")
    sx_str, sz_str = raw_syndrome_key.split()
    
    # Qiskit Bit Ordering Correction:
    # Qiskit returns 'S2 S1 S0' (Bit 2, Bit 1, Bit 0).
    # Our decoder map expects 'S0 S1 S2'.
    # We must reverse the strings to match the map keys.
    sz_corr = sz_str[::-1]
    sx_corr = sx_str[::-1]
    
    full_syndrome = sz_corr + sx_corr
    if verb > 0: print(f"Processed Full Syndrome (S0S1S2): Z-Stabs='{sz_corr}', X-Stabs='{sx_corr}'")
    
    decoded_corrections = decode_steane_syndrome(full_syndrome)
    decoded_x_locs = decoded_corrections['X'] 
    decoded_z_locs = decoded_corrections['Z'] 
    
    val_x = [] if decoded_x_locs == "Unknown" else decoded_x_locs
    val_z = [] if decoded_z_locs == "Unknown" else decoded_z_locs

    if verb > 0:
        print(f"\nDecoded Correction: X on qubits {decoded_x_locs}")
        print(f"Decoded Correction: Z on qubits {decoded_z_locs}")

        print("\n--- Verifying Decoder Correctness ---")
        print(f"  Injected X Errors on qubits: {true_x_errors}")
        print(f"  Decoded X Correction for qubits: {decoded_x_locs}")
        print(f"  Injected Z Errors on qubits: {true_z_errors}")
        print(f"  Decoded Z Correction for qubits: {decoded_z_locs}")
    
    passed_x = (sorted(true_x_errors) == sorted(val_x))
    passed_z = (sorted(true_z_errors) == sorted(val_z))

    if passed_x and passed_z:
        if verb > 0: print("\n*** DECODER VERIFICATION PASSED ***")
        return True
    else:
        if verb > 0:
            print("\n*** DECODER VERIFICATION FAILED ***")
            if not passed_x: print(" -> X-error decoding mismatch.")
            if not passed_z: print(" -> Z-error decoding mismatch.")
        return False


# --- 3. Main Execution Logic ---
def init_state(qc, data_qubits, args):
    if args.initOne:
        if args.verb > 0: print("--- Initializing to logical state |1_L> ---")
        # Logical X for Steane is transversal X (X on all qubits)
        qc.x(data_qubits) 
        qc.barrier(label='1-L   ')
    else:
        if args.verb > 0: print("--- Initializing to logical state |0_L> ---")

def main():
    parser = argparse.ArgumentParser(description=f"Steane {NUM_QUBITS}-Qubit (d={DISTANCE}) Code Simulation")
    parser.add_argument('-x','--xq', type=int, nargs='+', default=[], help=f'Qubit(s) (0-{NUM_QUBITS-1}) to apply an X error on.')
    parser.add_argument('-y','--yq', type=int, nargs='+', default=[], help=f'Qubit(s) (0-{NUM_QUBITS-1}) to apply a Y error on.')
    parser.add_argument('-z','--zq', type=int, nargs='+', default=[], help=f'Qubit(s) (0-{NUM_QUBITS-1}) to apply a Z error on.')
    parser.add_argument('-1','--initOne', action='store_true', help='Initialize the logical state to |1_L>.')
    parser.add_argument("-v", "--verb", type=int, help="Increase debug verbosity", default=1)
    args = parser.parse_args()
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
        
    raw_x = args.xq if args.xq else []
    raw_y = args.yq if args.yq else []
    raw_z = args.zq if args.zq else []

    true_x_errors = sorted(list(set(raw_x + raw_y)))
    true_z_errors = sorted(list(set(raw_z + raw_y)))
    
    print(f"--- Injecting true errors: X on {raw_x}, Y on {raw_y}, Z on {raw_z} ---")
    print(f"    (Effective physical errors: X on {true_x_errors}, Z on {true_z_errors})")

    data_qubits = QuantumRegister(NUM_QUBITS, name='q')
    anc_qx = QuantumRegister(NUM_ANC_X, name='ax') 
    anc_qz = QuantumRegister(NUM_ANC_Z, name='az') 
    synd_xbits = ClassicalRegister(NUM_ANC_X, name='sx')
    synd_zbits = ClassicalRegister(NUM_ANC_Z, name='sz')
    qc = QuantumCircuit(data_qubits, anc_qx, anc_qz, synd_zbits, synd_xbits)

    print("--- Building Encoder --- d=%d"%DISTANCE)
    qc.append(get_encoder_circuit().to_instruction(), data_qubits)

    init_state(qc, data_qubits, args)

    for loc in true_x_errors: qc.x(loc)
    for loc in true_z_errors: qc.z(loc)
    #if true_x_errors or true_z_errors: qc.barrier()

    print("--- Building Syndrome Measurement Circuit ---")
    qc.append(get_syndrome_circuit().to_instruction(), qc.qubits)
    
    # Measure stabilizers
    # anc_qx holds Z-stabilizer results (detects X errors) -> synd_zbits (Logically Z info)
    qc.measure(anc_qx, synd_zbits) 
    # anc_qz holds X-stabilizer results (detects Z errors) -> synd_xbits (Logically X info)
    qc.measure(anc_qz, synd_xbits)
    
    print("--- Final Circuit built --- nq=%d"%qc.num_qubits)
    print('M:  gates count:', qc.count_ops())

    if args.verb <=1: print(qc)
    if args.verb >1: print(qc.decompose())

    print("--- Running the simulation... ---")
    shots=1
    backend = AerSimulator()
    backend.set_max_qubits(qc.num_qubits)   
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qcT=pm.run(qc)
          
    result = backend.run(qcT, shots=shots).result()
    counts = result.get_counts()
    
    raw_syndrome_key = list(counts.keys())[0]
    evaluate_results(raw_syndrome_key, true_x_errors, true_z_errors, args.verb)

if __name__ == "__main__":
    main()
