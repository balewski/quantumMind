#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Unit test for Shor's 25-qubit (d=5) code simulation (shoreCode_d5.py).
Runs multiple iterations with random errors to verify decoder performance.
'''

import argparse
import random
from collections import Counter
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Import functions and constants from shoreCode_d5
from shoreCode_d5 import (
    init_state,
    decode_shor_syndrome,
    evaluate_results,
    get_encoder_circuit,
    get_syndrome_circuit,
    DISTANCE,
    NUM_QUBITS,
    NUM_ANC_X,
    NUM_ANC_Z
)

def build_circuit_and_measure(qc, data_qubits, anc_qx, anc_qz, synd_xbits, synd_zbits):
    """
    Reconstructs the encoding and syndrome measurement parts of shoreCode_d5.
    """
    d = DISTANCE
    
    # --- Encoder ---
    # Phase 1: CNOTs from qubit 0 to heads of other blocks
    for i in range(1, d): 
        qc.cx(data_qubits[0], data_qubits[i * d])
    
    # Phase 2: Hadamard on heads of all blocks
    qc.h([data_qubits[i * d] for i in range(d)])
    qc.barrier()
    
    # Phase 3: CNOTs within each block (repetition code encoding)
    for i in range(d):
        block_start = i * d
        for j in range(1, d): 
            qc.cx(data_qubits[block_start], data_qubits[block_start + j])
    qc.barrier()

    # --- (Error injection happens here in main flow) ---
    
    # --- Syndrome Measurement ---
    # Part 1: Measure X-type errors (using Z-stabilizers on anc_qx)
    for i in range(d):
        block_start = i * d
        anc_start = i * (d - 1)
        for j in range(d - 1):
            # Stabilizers Z_j Z_{j+1} in each block
            qc.cx(data_qubits[block_start + j], anc_qx[anc_start + j])
            qc.cx(data_qubits[block_start + j + 1], anc_qx[anc_start + j])
    qc.barrier()
    
    # Part 2: Measure Z-type errors (using X-stabilizers on anc_qz)
    # Prepare ancillas in |+>
    qc.h(anc_qz)
    qc.barrier()
    
    for i in range(d - 1):
        anc = anc_qz[i]
        # Stabilizers X_{block_i} X_{block_{i+1}}
        # Transversal CNOTs from ancilla to all qubits in block i and block i+1
        for j in range(d): qc.cx(anc, data_qubits[i * d + j])
        for j in range(d): qc.cx(anc, data_qubits[(i + 1) * d + j])
    qc.barrier()
    
    # Measure ancillas in X basis
    qc.h(anc_qz)
    qc.barrier()
    
    # Measurements
    qc.measure(anc_qz, synd_zbits)
    qc.measure(anc_qx, synd_xbits)


def run_single_test(args, backend, pm):
    """
    Runs a single simulation of the Shor d=5 code with random error injection.
    """
    
    # 1. Setup Registers
    data_qubits = QuantumRegister(NUM_QUBITS, name='q')
    anc_qx = QuantumRegister(NUM_ANC_X, name='ax')
    anc_qz = QuantumRegister(NUM_ANC_Z, name='az')
    synd_xbits = ClassicalRegister(NUM_ANC_X, name='sx')
    synd_zbits = ClassicalRegister(NUM_ANC_Z, name='sz')
    qc = QuantumCircuit(data_qubits, anc_qx, anc_qz, synd_zbits, synd_xbits)

    # 2. Initialize State
    init_state(qc, data_qubits, args)
    
    # 3. Build Encoder
    qc.append(get_encoder_circuit().to_instruction(), data_qubits)

    # 4. Inject Random Errors
    injected_errors = {} # q_idx -> type
    true_x_locs = []
    true_z_locs = []
    
    if args.numErr > 0:
        target_qubits = random.sample(range(NUM_QUBITS), args.numErr)
        for q_idx in target_qubits:
            err_type = random.choice(['X', 'Z'])
            injected_errors[q_idx] = err_type
            
            # Track components for verification
            if err_type == 'X':
                true_x_locs.append(q_idx)
                qc.x(data_qubits[q_idx])
            elif err_type == 'Z':
                true_z_locs.append(q_idx)
                qc.z(data_qubits[q_idx])
    
    if args.verb > 0:
        print(f"Injected: {injected_errors}")

    qc.barrier()

    # 5. Syndrome Measurement
    qc.append(get_syndrome_circuit().to_instruction(), qc.qubits)
    
    qc.measure(anc_qz, synd_zbits)
    qc.measure(anc_qx, synd_xbits)

    if args.verb > 1:
        print(qc)

    # 6. Simulate
    # backend and pm are passed in to reuse resources and avoid memory leaks
    qcT = pm.run(qc)
    result = backend.run(qcT, shots=1).result()
    counts = result.get_counts()
    raw_syndrome = list(counts.keys())[0]
    
    # 7. Verify using shoreCode_d5's evaluate_results
    is_success = evaluate_results(raw_syndrome, sorted(true_x_locs), sorted(true_z_locs), args.verb)

    return is_success, {
        'injected': injected_errors,
        'decoded': "N/A (handled by evaluate_results)",
        'syndrome': raw_syndrome
    }

def main():
    parser = argparse.ArgumentParser(description="Unit test for Shor's 25-qubit (d=5) code")
    parser.add_argument('-e','--numErr', type=int, default=1, help='Number of errors injected per circuit')
    parser.add_argument('-n','--numRep', type=int, default=100, help='Number of repetitions')
    parser.add_argument('-1','--initOne', action='store_true', help='Initialize logical state to |1_L>')
    # Dummy args for init_state compatibility
    parser.add_argument('-i','--initState', type=float, nargs=2, default=None, help=argparse.SUPPRESS)
    parser.add_argument("-v", "--verb", type=int, help="Increase debug verbosity", default=0)
    
    args = parser.parse_args()
    
    print(f"Running evaluation: {args.numRep} reps, {args.numErr} errors/circuit, initOne={args.initOne}")
    
    success_count = 0
    failures = []
    
    # Verbosity control
    original_verb = args.verb
    
    # Setup backend once to avoid memory issues (force stabilizer method for Clifford circuits)
    backend = AerSimulator(method='stabilizer')
    # Calculate total qubits for pass manager setup
    total_qubits = NUM_QUBITS + NUM_ANC_X + NUM_ANC_Z
    # We need to trick set_max_qubits or just let PM handle it. 
    # set_max_qubits is a backend option but usually for dynamic circuits. 
    # For simple run, generate_preset_pass_manager is enough.
    # But user code had backend.set_max_qubits(qc.num_qubits).
    # AerSimulator doesn't inherently have a fixed limit unless set.
    # We'll assume standard usage is fine.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

    for i in range(args.numRep):
        # For first run (i=0), use user's verbosity (prints circuit if >=2)
        # For runs 0-4, ensure verbosity is at least 1 to print injections
        # For runs >=5, silence it (verb=0)
        
        if i < 5:
            args.verb = max(1, original_verb)
        else:
            args.verb = 0
            
        # Special case: If user wanted circuit (verb=2) but we are in runs 1-4, 
        # we don't want to print circuit again.
        if i > 0 and args.verb > 1:
            args.verb = 1

        is_success, info = run_single_test(args, backend, pm)
        if is_success:
            success_count += 1
        else:
            failures.append(info)
            
    args.verb = original_verb # Restore at end
            
    fraction = success_count / args.numRep
    print(f"\nSuccess rate: {success_count}/{args.numRep} ({fraction:.2%})")
    
    if failures:
        print("\nSample Failures (first 5):")
        for i, fail in enumerate(failures[:5]):
            print(f"--- Failure {i+1} ---")
            print(f"  Injected: {fail['injected']}")
            print(f"  Syndrome: {fail['syndrome']}")
            print(f"  Decoded:  {fail['decoded']}")

if __name__ == "__main__":
    main()

