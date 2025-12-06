#!/usr/bin/env python3
__author__ = "Jan Balewski, modified by Gemini"
__email__ = "janstar1122@gmail.com"

import argparse
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from itertools import combinations
from collections import Counter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# --- 1. Global Code Parameters ---
DISTANCE = 5
NUM_QUBITS = DISTANCE * DISTANCE
NUM_ANC_X = DISTANCE * (DISTANCE - 1)
NUM_ANC_Z = DISTANCE - 1
assert DISTANCE == 5  # it sometimes fails for d=3, I did not debug it


def decode_shor_syndrome(full_syndrome, d=DISTANCE):
    rep_code_decoder = build_repetition_code_decoder_map(d)
    
    num_sz_bits = d - 1
    sz_full_str = full_syndrome[:num_sz_bits]
    sx_full_str = full_syndrome[num_sz_bits:]

    sz_str_reversed = sz_full_str[::-1]
    sx_str_reversed = sx_full_str[::-1]
    
    logical_z_errors = rep_code_decoder.get(sz_str_reversed, "Unknown")
    
    x_errors = []
    num_sx_bits_per_block = d - 1
    for i in range(d):
        block_start = i * num_sx_bits_per_block
        block_end = block_start + num_sx_bits_per_block
        sx_block_str = sx_str_reversed[block_start:block_end]
        block_x_errors = rep_code_decoder.get(sx_block_str, "Unknown")
        
        if block_x_errors != "Unknown":
            for loc in block_x_errors:
                x_errors.append(i * d + loc)
                
    return {'Z': logical_z_errors, 'X': x_errors}


# --- 3. Main Execution Logic ---
def init_state(qc, data_qubits, args):
    if args.initOne:
        print("--- Initializing to logical state |1_L> ---")
        qc.x(data_qubits[0])
        qc.barrier()
    else:
        print("--- Initializing to logical state |0_L> ---")

def main():
    parser = argparse.ArgumentParser(description=f"Shor's {NUM_QUBITS}-Qubit (d={DISTANCE}) Code Simulation")
    parser.add_argument('-x','--xq', type=int, nargs='+', help=f'Qubit(s) (0-{NUM_QUBITS-1}) to apply an X error on.')
    parser.add_argument('-z','--zq', type=int, nargs='+', help=f'Qubit(s) (0-{NUM_QUBITS-1}) to apply a Z error on.')
    parser.add_argument('-o','--initOne', action='store_true', help='Initialize the logical state to |1_L>.')
    parser.add_argument("-v", "--verb", type=int, help="Increase debug verbosity", default=1)
    args = parser.parse_args()

    true_x_errors = sorted(list(set(args.xq))) if args.xq else []
    true_z_errors = sorted(list(set(args.zq))) if args.zq else []
    print(f"--- Injecting true errors: X on {true_x_errors}, Z on {true_z_errors} ---")

    data_qubits = QuantumRegister(NUM_QUBITS, name='q')
    anc_qx = QuantumRegister(NUM_ANC_X, name='ax')
    anc_qz = QuantumRegister(NUM_ANC_Z, name='az')
    synd_xbits = ClassicalRegister(NUM_ANC_X, name='sx')
    synd_zbits = ClassicalRegister(NUM_ANC_Z, name='sz')
    qc = QuantumCircuit(data_qubits, anc_qx, anc_qz, synd_zbits, synd_xbits)

    init_state(qc, data_qubits, args)

    print("--- Building Encoder --- d=%d"%DISTANCE)
    d = DISTANCE
    for i in range(1, d): qc.cx(data_qubits[0], data_qubits[i * d])
    qc.h([data_qubits[i * d] for i in range(d)])
    qc.barrier()
    for i in range(d):
        block_start = i * d
        for j in range(1, d): qc.cx(data_qubits[block_start], data_qubits[block_start + j])
    qc.barrier()

    for loc in true_x_errors: qc.x(loc)
    for loc in true_z_errors: qc.z(loc)
    if true_x_errors or true_z_errors: qc.barrier()

    print("--- Building Syndrome Measurement Circuit ---")
    for i in range(d):
        block_start = i * d
        anc_start = i * (d - 1)
        for j in range(d - 1):
            qc.cx(data_qubits[block_start + j], anc_qx[anc_start + j])
            qc.cx(data_qubits[block_start + j + 1], anc_qx[anc_start + j])
    qc.barrier()
    qc.h(anc_qz)
    qc.barrier()
    for i in range(d - 1):
        anc = anc_qz[i]
        for j in range(d): qc.cx(anc, data_qubits[i * d + j])
        for j in range(d): qc.cx(anc, data_qubits[(i + 1) * d + j])
    qc.barrier()
    qc.h(anc_qz)
    qc.barrier()
    
    qc.measure(anc_qz, synd_zbits)
    qc.measure(anc_qx, synd_xbits)
    
    print("--- Final Circuit built --- nq=%d"%qc.num_qubits)
    print('M:  gates count:', qc.count_ops())

    if args.verb > 1: print(qc)
    
    print("--- Running the simulation... ---")
    shots=1
    backend = AerSimulator()
    backend.set_max_qubits(qc.num_qubits)
    if 1:   # TranspilerError: 'HighLevelSynthesis is unable to synthesize "measure"'
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        qc=pm.run(qc)
        
        print('M: transp  gates count:', qc.count_ops())
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()
    print('counts:', counts)
    
  

if __name__ == "__main__":
    main()
