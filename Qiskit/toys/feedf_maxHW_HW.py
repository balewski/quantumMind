#!/usr/bin/env python3
import argparse
from curses import pair_number
import itertools
from functools import reduce
from pprint import pprint

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.classical import expr
from qiskit_ibm_runtime import SamplerV2 as Sampler

def parse_args():
    parser = argparse.ArgumentParser(description="Generalized Majority Vote Simulation with Feed-Forward")
    parser.add_argument('-d', "--dist", type=int, default=5, help="Number of data qubits (distance d)")
    parser.add_argument('-n', "--shots", type=int, default=10, help="Number of simulation shots")
    parser.add_argument('-c', "--printCirc", action="store_true", help="Print circuit ")
    parser.add_argument('-b', '--backend', default='ideal', help="quantum backend")
    parser.add_argument('-s', '--seed', type=int, default=42, help="Seed for random generator")
    return parser.parse_args()

def build_boolean_majority_circuit(n_qubits):
    """
    Computes majority using purely boolean logic (AND/OR).
    Majority threshold = floor(n/2) + 1.
    If ANY subset of size 'threshold' has all 1s, the majority is 1.
    """
    threshold = (n_qubits // 2) + 1
    print(f"Building circuit for d={n_qubits}, Majority Threshold={threshold}")

    qr_a = QuantumRegister(n_qubits, 'a')
    qr_b = QuantumRegister(1, 'b')
    cr_a = ClassicalRegister(n_qubits, 'm_a')
    cr_b = ClassicalRegister(1, 'm_b')
    
    qc = QuantumCircuit(qr_a, qr_b, cr_a, cr_b)

    # 1. State Prep (Superposition)
    qc.h(qr_a)
    qc.barrier()

    # 2. Measure Data
    qc.measure(qr_a, cr_a)
    qc.barrier()

    # 3. Feed-Forward: Boolean Majority Logic
    subset_conditions = []
    
    # Generate all combinations of indices of size 'threshold'
    for indices in itertools.combinations(range(n_qubits), threshold):
        # Create AND chain for this specific subset of bits
        term = reduce(lambda x, y: expr.bit_and(x, y), [cr_a[i] for i in indices])
        subset_conditions.append(term)
        
    # OR all the subset conditions together
    if not subset_conditions:
        print("Warning: No majority conditions generated.")
        majority_expr = None 
    else:
        majority_expr = reduce(lambda x, y: expr.bit_or(x, y), subset_conditions)

    # 4. Apply Feed-Forward X
    if majority_expr is not None:
        with qc.if_test(majority_expr):
            qc.x(qr_b)

    qc.barrier()
    qc.measure(qr_b, cr_b)
    
    return qc

def verify_results(data_a, data_b, n_qubits, mxRow=12):
    bs_a = data_a.get_bitstrings()
    bs_b = data_b.get_bitstrings()
    shots = len(bs_a)
    assert len(bs_b) == shots
    
    all = {}  # nested dict: all[b_value][a_value] = counts

    print("\n--- Verification (Ordered by Measurement) ---")
    threshold = (n_qubits // 2) + 1
    correct = 0
    total = shots

    print(f"{'Meas (b a)':<20} | {'Maj(a)':<6} | {'Res(b)':<6} | {'Check'}")
    print("-" * 55)

    for i in range(shots):
        val_a = bs_a[i]
        val_b = bs_b[i]
        
        if val_b not in all:
            all[val_b] = {}
        if val_a not in all[val_b]:
            all[val_b][val_a] = 0
        all[val_b][val_a] += 1

        ones = val_a.count('1')
        expected_b = '1' if ones >= threshold else '0'
        if val_b == expected_b:
            correct += 1

        if i < mxRow:
            bitstring = f"{val_b} {val_a}"
            status = "OK" if val_b == expected_b else "FAIL"
            print(f"{bitstring:<20} | {expected_b:<6} | {val_b:<6} | {status}")

    print("-" * 55)
    acc = correct/total if total > 0 else 0
    print(f"Total Accuracy: {acc:.1%} (Shots: {total})")
    
    print("\nall[b-value]{a-value:counts}:")
    pprint(all)

    return {"acc": acc}

if __name__ == "__main__":
    args = parse_args()
    
    # 1. Build Circuit
    qc = build_boolean_majority_circuit(args.dist)
    if args.printCirc:   print(qc)
    
    # 2. Setup Backend & Sampler
    print(f"Running circuit with {args.shots} shots...")
    
    if args.backend == 'ideal':
        # Initialize AerSimulator with seed if provided
        backend = AerSimulator(seed_simulator=args.seed)
        qc_run = qc 
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit import transpile
        service = QiskitRuntimeService()
        print(f'\nUsing real HW backend: {args.backend} ...')
        backend = service.backend(args.backend)
        qc_run = transpile(qc, backend)
        if args.printCirc:   print(qc_run)
        
    print('M: running sampler ...')
    sampler = Sampler(mode=backend)
    
    # 3. Run Job
    # If seed is provided, we can also pass it to the run options for some backends,
    # but for AerSimulator, initializing it in the constructor is robust.
    job = sampler.run([qc_run], shots=args.shots)
    result = job.result()
    print('M: job done')
    
    # 4. Extract Data & Verify
    creg_a = result[0].data.m_a
    creg_b = result[0].data.m_b
    
    final_dict = verify_results(creg_a, creg_b, args.dist)

    print("\nSummary , backend:", backend.name)
    pprint(final_dict)
