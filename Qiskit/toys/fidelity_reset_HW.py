#!/usr/bin/env python3
"""
This program benchmarks quantum qubits by running a repeated "X-gate, Measure, Reset" cycle for a specified depth. It constructs a Qiskit circuit where every qubit independently undergoes this sequence, recording results in separate registers. The script executes this circuit on either an ideal simulator or real IBM hardware to test state preparation and measurement fidelity. Post-processing calculates the average probability of correctly measuring the |1> state for each qubit. Finally, it prints a detailed report combining these probabilities with hardware calibration data (T1, T2, and readout errors).
"""
import argparse
import math
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
    """
    Constructs a circuit with n_qubits.
    For each qubit, repeats (X -> Measure -> Reset) 'depth' times.
    Records bits in separate registers for each qubit.
    """
    print(f"Building circuit: {n_qubits} qubits, {depth} cycles (X-Meas-Reset)")
    
    qr = QuantumRegister(n_qubits, 'q')
    # Create a separate ClassicalRegister for each qubit to hold its 'depth' measurements
    crs = [ClassicalRegister(depth, name=f"c{i}") for i in range(n_qubits)]
    
    qc = QuantumCircuit(qr, *crs)

    # Perform the sequence
    for d in range(depth):
        for q in range(n_qubits):
            if d>0 : qc.reset(qr[q])
            qc.x(qr[q])
            # Measure into the specific bit index d of the specific register for qubit q
            qc.measure(qr[q], crs[q][d])            
        if d< depth-1: qc.barrier()
        
    return qc

def extract_physical_layout(qc):
    """Extracts physical IDs from a transpiled circuit."""
    try:   
        physQubitLayout = qc.layout.final_index_layout(filter_ancillas=True)
        # If layout is None or empty, fallback
        if physQubitLayout is None:
             physQubitLayout = list(range(qc.num_qubits))
    except AttributeError:
        # If qc doesn't have layout attribute (not transpiled)
        physQubitLayout = list(range(qc.num_qubits))
    except Exception:
        physQubitLayout = list(range(qc.num_qubits))
    
    return physQubitLayout

def print_calibration_data(backend, phys_qubits):
    """
    Extracts and prints T1, T2, detailed Readout Errors, and Gate Errors.
    """
    if backend is None or backend.name == 'aer_simulator':
        return

    try:
        props = backend.properties()
        if props is None:
            print("\n--- Calibration Data (Not available for this backend) ---")
            return
            
        print("\n--- Calibration Data (From Backend DB) ---")
        print(f"Backend: {backend.name}  (Last Update: {props.last_update_date})")
        # Updated Header with separate readout error columns
        print(f"{'PhysID':<6} | {'T1(us)':<8} | {'T2(us)':<8} | {'s0->m1':<8} | {'s1->m0':<8} | {'AvgErr':<8} | {'X_Gate_Err':<10}")
        print("-" * 80)
        
        unique_qubits = sorted(list(set(phys_qubits)))
        
        for q in unique_qubits:
            # T1 / T2
            try:
                t1 = props.t1(q) * 1e6
                t2 = props.t2(q) * 1e6
            except:
                t1, t2 = -1.0, -1.0
            
            # Readout Errors
            try:
                p_0m1 = props.qubit_property(q, 'prob_meas1_prep0')[0] # Error: Prepared 0, Measured 1
                p_1m0 = props.qubit_property(q, 'prob_meas0_prep1')[0] # Error: Prepared 1, Measured 0
                ave_meas_err = (p_0m1 + p_1m0) / 2.0
            except:
                p_0m1, p_1m0, ave_meas_err = -1.0, -1.0, -1.0
            
            # Gate Error
            try:
                x_err = props.gate_error('x', [q])
            except:
                x_err = -1.0

            # Formatting
            s_t1 = f"{t1:.1f}" if t1 >= 0 else "N/A"
            s_t2 = f"{t2:.1f}" if t2 >= 0 else "N/A"
            
            s_0m1 = f"{p_0m1:.4f}" if p_0m1 >= 0 else "N/A"
            s_1m0 = f"{p_1m0:.4f}" if p_1m0 >= 0 else "N/A"
            s_avg = f"{ave_meas_err:.4f}" if ave_meas_err >= 0 else "N/A"
            
            s_xerr = f"{x_err:.2e}" if x_err >= 0 else "N/A"
            
            print(f"{q:<6} | {s_t1:<8} | {s_t2:<8} | {s_0m1:<8} | {s_1m0:<8} | {s_avg:<8} | {s_xerr:<10}")
            
    except Exception as e:
        print(f"\n[Warning] Could not extract calibration data: {e}")
        
def analyze_results(result_data, n_qubits, depth, total_shots, phys_layout):
    """
    Computes average probability of measuring state 1.
    """
    print("\n" + "="*80)
    print("ANALYSIS: Probability of State |0> (Target prob: 0.0)")
    print("="*80)
    
    print(f"{'LogQ':<6} | {'PhysQ':<6} | {'Prob(0)':<10} | {'Bits(1)/Total'}")
    print("-" * 50)
    
    total_ones_all = 0
    total_bits_all = 0
    
    for i in range(n_qubits):
        reg_name = f"c{i}"
        # SamplerV2 returns bitstrings in the data object under the register name
        try:
            bitstrings = getattr(result_data, reg_name).get_bitstrings()
        except AttributeError:
            print(f"Error: Could not find register data for {reg_name}")
            continue

        qubit_ones = 0
        qubit_bits = 0
        
        for bs in bitstrings:
            # Note: bs length is 'depth'
            qubit_ones += bs.count('0')
            qubit_bits += len(bs)
            
        prob = qubit_ones / qubit_bits if qubit_bits > 0 else 0.0
        
        phys_id = phys_layout[i] if i < len(phys_layout) else "?"
        print(f"{i:<6} | {phys_id:<6} | {prob:.4f}    | {qubit_ones}/{qubit_bits}")
        
        total_ones_all += qubit_ones
        total_bits_all += qubit_bits

    print("-" * 50)
    global_avg = total_ones_all / total_bits_all if total_bits_all > 0 else 0.0
    print(f"Global Average Prob(1): {global_avg:.5f}")


if __name__ == "__main__":
    
    args = parse_args()
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    # 1. Build
    qc = build_x_meas_reset_circuit(args.size, args.depth)
    
    if args.printCirc:
        print(qc)
    
    # 2. Backend Setup
    if args.backend == 'ideal':
        print(f"\nRunning simulation (Ideal Aer) with {args.shots} shots...")
        backend = AerSimulator(seed_simulator=args.seed)
        qc_run = qc 
        phys_layout = list(range(args.size))
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit import transpile
        service = QiskitRuntimeService()
        print(f'\nUsing real HW backend: {args.backend} ...')
        backend = service.backend(args.backend)
        qc_run = transpile(qc, backend)
        phys_layout = extract_physical_layout(qc_run)
        print(f"Transpilation complete. Physical Layout: {phys_layout}")
        
    # 3. Print Calibration (using physical qubits from layout)
    # We only care about the first n_qubits mapped in the layout
    active_phys_qubits = phys_layout[:args.size]
    print_calibration_data(backend, active_phys_qubits)
        
    # 4. Run Sampler
    print('\nRunning sampler on %s ...'%(backend.name))
    sampler = Sampler(mode=backend)
    job = sampler.run([qc_run], shots=args.shots)
    result = job.result()
    print('Job done.')
    
    # 5. Analysis
    # result[0].data contains the registers for the first (and only) pub
    analyze_results(result[0].data, args.size, args.depth, args.shots, phys_layout)
