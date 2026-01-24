#!/usr/bin/env python3
"""
This program benchmarks quantum qubits by running a repeated "X-gate, Measure, Reset" cycle for a specified depth. It constructs a Qiskit circuit where every qubit independently undergoes this sequence, recording results in separate registers. The script executes this circuit on either an ideal simulator or real IBM hardware to test state preparation and measurement fidelity. Post-processing calculates the average probability of correctly measuring the |1> state for each qubit. Finally, it prints a detailed report combining these probabilities with hardware calibration data (T1, T2, and readout errors).
"""
import argparse
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

def parse_args():
    parser = argparse.ArgumentParser(description="X-Measure-Reset Sequence Benchmark")
    parser.add_argument('--size', type=int, default=5, help="Number of qubits")
    parser.add_argument('--depth', type=int, default=8, help="Number of X-Meas-Reset cycles per qubit")
    parser.add_argument('-n', "--shots", type=int, default=1000, help="Number of simulation shots")
    parser.add_argument('-c', "--printCirc", action="store_true", help="Print circuit")
    parser.add_argument('-b', '--backend', default='ideal', help="quantum backend")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Seed for random generator")
    return parser.parse_args()

def build_x_meas_reset_circuit(n_qubits, depth):
    print(f"Building circuit: {n_qubits} qubits, {depth} cycles (X-Meas-Reset)")
    qr = QuantumRegister(n_qubits, 'q')
    crs = [ClassicalRegister(depth, name=f"c{i}") for i in range(n_qubits)]
    qc = QuantumCircuit(qr, *crs)

    for d in range(depth):
        for q in range(n_qubits):
            qc.reset(qr[q])  # even for initial qubits, to make result consistent
            qc.x(qr[q])
            qc.measure(qr[q], crs[q][d])            
        if d < depth-1: qc.barrier()
    return qc

def print_calibration_data(backend, phys_qubits):
    if backend is None or backend.name == 'aer_simulator': return
    props = backend.properties()
    if not props: return
    
    print("\n--- Calibration Data ---")
    print(f"{'PhysID':<6} | {'T1(us)':<8} | {'T2(us)':<8} | {'err s0->m1':<10} | {'err s1->m0':<10} | {'AvgErr':<8} | {'X_Gate_Err':<10}")
    print("-" * 80)
    
    for q in sorted(list(set(phys_qubits))):
        t1 = props.t1(q) * 1e6
        t2 = props.t2(q) * 1e6
        p01 = props.qubit_property(q, 'prob_meas1_prep0')[0]
        p10 = props.qubit_property(q, 'prob_meas0_prep1')[0]
        x_err = props.gate_error('x', [q])
        
        print(f"{q:<6} | {t1:<8.1f} | {t2:<8.1f} | {p01:<10.4f} | {p10:<10.4f} | {(p01+p10)/2:<8.4f} | {x_err:<10.2e}")
    print("-" * 80)
    print(f"{'PhysID':<6} | {'T1(us)':<8} | {'T2(us)':<8} | {'err s0->m1':<10} | {'err s1->m0':<10} | {'AvgErr':<8} | {'X_Gate_Err':<10}")

def analyze_results(result_data, n_qubits,  phys_layout):
    """
    Computes probability of measuring state 0 and Binomial error using raw uint8 data.
    """
    print("\n" + "="*60)
    print("ANALYSIS: Probability of State |0> (Target: 0.0)")
    print("="*60)
    print(f"{'LogQ':<6} | {'PhysQ':<6} | {'Prob(0)':<10} | {'std prob':<12}")
    print("-" * 45)

    for i in range(n_qubits):
        # Access the raw BitArray directly
        bit_array = getattr(result_data, f"c{i}")
        
        # Unpack bits (shots x depth)
        raw_bits = np.unpackbits(bit_array.array, axis=1, bitorder='little')
        
        #  Get depth directly from the BitArray object ---
        depth = bit_array.num_bits
        # Valid slice
        valid_bits = raw_bits[:, :depth]
        if i==0: print('BAS',bit_array.array.shape,'depth',depth,valid_bits.shape)
        # Statistics
        total_samples = valid_bits.size
        count_1 = np.count_nonzero(valid_bits)
        count_0 = total_samples - count_1
        
        prob_0 = count_0 / total_samples
        
        # Binomial Error Calculation
        if 0 < prob_0 < 1:
            err_0 = np.sqrt(prob_0 * (1 - prob_0) / total_samples)
        else:
            err_0 = 1.0 / total_samples
        
        phys_id = phys_layout[i]
        print(f"{i:<6} | {phys_id:<6} | {prob_0:.4f}     | {err_0:.4f}")

if __name__ == "__main__":
    args = parse_args()

    # 1. Build
    qc = build_x_meas_reset_circuit(args.size, args.depth)
    if args.printCirc: print(qc)
    
    # 2. Backend Setup
    if args.backend == 'ideal':
        print(f"\nRunning simulation (Ideal Aer) with {args.shots} shots...")
        backend = AerSimulator(seed_simulator=args.seed)
        qc_run = qc 
        phys_layout = list(range(args.size))
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit import transpile
        print(f'\nUsing real HW backend: {args.backend} ...')
        service = QiskitRuntimeService()
        backend = service.backend(args.backend)
        qc_run = transpile(qc, backend)
        phys_layout = qc_run.layout.final_index_layout(filter_ancillas=True)
        print(f"Transpilation complete. Physical Layout: {phys_layout}")
        
    # 3. Calibration & Execution
    print_calibration_data(backend, phys_layout[:args.size])
    
    print(f'\nRunning sampler on {backend.name}  nq={args.size} depth={args.depth}...')
    sampler = Sampler(mode=backend)
    job = sampler.run([qc_run], shots=args.shots)
    result = job.result()
    print('Job done. shots:',args.shots)
    
    # 4. Analysis
    analyze_results(result[0].data, args.size, phys_layout)
