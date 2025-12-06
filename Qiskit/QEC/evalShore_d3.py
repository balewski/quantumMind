#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Unit test for Shor's 9-qubit code simulation (shoreCode_d3.py).
Runs multiple iterations with random errors to verify decoder performance.
'''

import argparse
import random
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Import functions from shoreCode_d3
from shoreCode_d3 import (
    get_encoder_circuit,
    get_syndrome_circuit,
    build_decoder_map,
    init_state,
    evaluate_results
)

def run_single_test(args, x_err_map, z_err_map):
    """
    Runs a single simulation of the Shor code with random error injection.
    Returns (is_success, error_info_dict)
    """
    
    # 1. Setup Circuit
    data_qubits = QuantumRegister(9, name='q')
    anc_qx = QuantumRegister(6, name='ax')
    anc_qz = QuantumRegister(2, name='az')
    synd_xbits = ClassicalRegister(6, name='sx')
    synd_zbits = ClassicalRegister(2, name='sz')
    qc = QuantumCircuit(data_qubits, anc_qx, anc_qz, synd_xbits, synd_zbits)

    # 2. Initialize State
    # Mock args for init_state
    init_state(qc, data_qubits, args)
    
    # 3. Encode
    qc.append(get_encoder_circuit().to_instruction(), data_qubits)

    # 4. Inject Random Errors
    # args.numErr determines how many errors to inject
    # We pick 'numErr' distinct qubits and assign a random Pauli (X, Y, Z) to each
    injected_errors = {} # Map qubit_idx -> error_type ('X', 'Y', 'Z')
    
    if args.numErr > 0:
        target_qubits = random.sample(range(9), args.numErr)
        for q_idx in target_qubits:
            err_type = random.choice(['X', 'Z'])
            injected_errors[q_idx] = err_type
            if err_type == 'X':
                qc.x(data_qubits[q_idx])
            elif err_type == 'Z':
                qc.z(data_qubits[q_idx])
    
    if args.verb > 0:
        print(f"Injected: {injected_errors}")
    
    # 5. Measure Syndrome
    qc.append(get_syndrome_circuit().to_instruction(), qc.qubits)
    qc.measure(anc_qx, synd_xbits)
    qc.measure(anc_qz, synd_zbits)

    if args.verb > 1:
        print(qc)

    # 6. Simulate
    simulator = AerSimulator()
    pm = generate_preset_pass_manager(backend=simulator, optimization_level=1)
    result = simulator.run(pm.run(qc), shots=1).result()
    counts = result.get_counts()
    raw_syndrome = list(counts.keys())[0]
    
    # 7. Decode & Verify using shoreCode_d3's evaluate_results
    # Determine true error for verification
    if len(injected_errors) == 0:
        true_type, true_loc = 'No error', None
    elif len(injected_errors) == 1:
        loc = list(injected_errors.keys())[0]
        true_type = injected_errors[loc]
        true_loc = loc
    else:
        true_type, true_loc = 'Complex', None # Will cause verify to fail, as expected

    is_success = evaluate_results(raw_syndrome, x_err_map, z_err_map, true_type, true_loc, verb=args.verb)

    return is_success, {
        'injected': injected_errors,
        'decoded': "N/A (handled by evaluate_results)",
        'syndrome': raw_syndrome
    }

def main():
    parser = argparse.ArgumentParser(description="Unit test for Shor's 9-qubit code")
    parser.add_argument('-e','--numErr', type=int, default=1, help='Number of errors injected per circuit')
    parser.add_argument('-n','--numRep', type=int, default=100, help='Number of repetitions')
    parser.add_argument('-1','--initOne', action='store_true', help='Initialize logical state to |1_L>')
    # Dummy args for init_state compatibility
    parser.add_argument('-i','--initState', type=float, nargs=2, default=None, help=argparse.SUPPRESS)
    parser.add_argument("-v", "--verb", type=int, help="Increase debug verbosity", default=0)
    
    args = parser.parse_args()
    
    print(f"Running evaluation: {args.numRep} reps, {args.numErr} errors/circuit, initOne={args.initOne}")
    
    x_err_map, z_err_map = build_decoder_map()
    
    success_count = 0
    failures = []
    
    for i in range(args.numRep):
        if i < 1: args.verb = 2  # Print circuit once (+ injection)
        elif i < 2: args.verb = 1 # Print injection info only
        else: args.verb = 0
        is_success, info = run_single_test(args, x_err_map, z_err_map)
        if is_success:
            success_count += 1
        else:
            failures.append(info)
            
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

