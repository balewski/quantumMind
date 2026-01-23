#!/usr/bin/env python3
import argparse
import itertools
import math
from functools import reduce

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.classical import expr
from qiskit_ibm_runtime import SamplerV2 as Sampler

def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Majority Vote with Binomial Error Bars")
    parser.add_argument('--size', type=int, default=5, help="Number of qubit pairs (register size)")
    parser.add_argument('--dist', type=int, default=5, help="Number of measurement repetitions (must be ODD)")
    parser.add_argument('--reset', action='store_true', help="Use native qiskit reset (unconditional) instead of FF-logic reset")
    parser.add_argument('-n', "--shots", type=int, default=10, help="Number of simulation shots")
    parser.add_argument('-c', "--printCirc", action="store_true", help="Print circuit ")
    parser.add_argument('-b', '--backend', default='ideal', help="quantum backend")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Seed for random generator")
    
    args = parser.parse_args()
    
    if args.dist % 2 == 0:
        parser.error(f"Argument --dist must be odd, but got {args.dist}.")
        
    return args

def build_temporal_majority_circuit(n_qubits, d_repetitions, use_native_reset):
    threshold = (d_repetitions // 2) + 1
    print(f"Building circuit: size={n_qubits} pairs, dist={d_repetitions} meas/qubit, Threshold={threshold}, NativeReset={use_native_reset}")

    qr_a = QuantumRegister(n_qubits, 'a')
    qr_b = QuantumRegister(n_qubits, 'b')
    
    # 1. History registers named 'ca' (Corrected Assignment/Classical A)
    cr_a_history_list = [ClassicalRegister(d_repetitions, name=f"ca{i}") for i in range(n_qubits)]
    
    # 2. Final verification register for 'a' (to check reset)
    cr_final_a = ClassicalRegister(n_qubits, name='m_a')
    
    # 3. Final register for 'b'
    cr_b = ClassicalRegister(n_qubits, name='m_b')
    
    # Combine all
    qc = QuantumCircuit(qr_a, qr_b, *cr_a_history_list, cr_final_a, cr_b)

    # --- Initialization ---
    qc.h(qr_a)     
    qc.barrier()

    # --- Repeated Measurement (Temporal Loop) ---
    for rep in range(d_repetitions):
        for i in range(n_qubits):
            # Measure into specific history bit
            qc.measure(qr_a[i], cr_a_history_list[i][rep])
        qc.barrier()

    # --- Feed-Forward: Compute Majority & Reset ---
    for i in range(n_qubits):
        history_reg = cr_a_history_list[i]
        subset_conditions = []
        
        # Boolean Majority Logic
        for indices in itertools.combinations(range(d_repetitions), threshold):
            bits = [history_reg[idx] for idx in indices]
            # Fix for d=1: Lift single bits to Expr
            if len(bits) == 1:
                term = expr.lift(bits[0])
            else:
                term = reduce(lambda x, y: expr.bit_and(x, y), bits)
            subset_conditions.append(term)
            
        if not subset_conditions:
            majority_expr = None
        else:
            if len(subset_conditions) == 1:
                majority_expr = subset_conditions[0]
            else:
                majority_expr = reduce(lambda x, y: expr.bit_or(x, y), subset_conditions)

        # Apply Correction
        if majority_expr is not None:
            with qc.if_test(majority_expr):
                qc.x(qr_b[i]) # Set b to 1
                if not use_native_reset:
                    qc.x(qr_a[i]) # Manual Reset a to 0
        
        # Native Reset (Unconditional)
        if use_native_reset:
            qc.reset(qr_a[i])

    qc.barrier()
    
    # --- Final Measurements ---
    qc.measure(qr_a, cr_final_a)
    qc.measure(qr_b, cr_b)
    
    return qc

def count_changes(bitstring):
    """Counts number of bit flips in a string."""
    changes = 0
    for k in range(len(bitstring) - 1):
        if bitstring[k] != bitstring[k+1]:
            changes += 1
    return changes

def extract_physical_layout(qc):
    """Extracts physical IDs."""
    try:  
        physQubitLayout = qc._layout.final_index_layout(filter_ancillas=True)
        nqTot = len(physQubitLayout)
    except:
        nqTot = qc.num_qubits
        physQubitLayout = [i for i in range(nqTot)]
    return physQubitLayout

def print_calibration_data(backend, phys_qubits):
    """
    Extracts and prints T1, T2, Readout Errors, and Gate Errors.
    """
    if backend is None or backend.name == 'aer_simulator':
        return

    try:
        props = backend.properties()
        if props is None:
            print("\n--- D) Calibration Data (Not available for this backend) ---")
            return
            
        print("\n--- D) Calibration Data (From Backend DB) ---")
        print(f"Backend: {backend.name}  (Last Update: {props.last_update_date})")
        print(f"{'PhysID':<6} | {'T1(us)':<8} | {'T2(us)':<8} | {'AveMeasErr':<10} | {'Set0Meas1':<9} | {'Set1Meas0':<9} | {'X_Gate_Err':<10}")
        print("-" * 85)
        
        unique_qubits = sorted(list(set(phys_qubits)))
        
        for q in unique_qubits:
            try:
                t1 = props.t1(q) * 1e6
                t2 = props.t2(q) * 1e6
            except:
                t1, t2 = -1.0, -1.0
                
            try:
                p_0m1 = props.qubit_property(q, 'prob_meas1_prep0')[0]
                p_1m0 = props.qubit_property(q, 'prob_meas0_prep1')[0]
                ave_meas_err = (p_0m1 + p_1m0) / 2.0
            except:
                p_0m1, p_1m0, ave_meas_err = -1.0, -1.0, -1.0
            
            try:
                x_err = props.gate_error('x', [q])
            except:
                x_err = -1.0

            s_t1 = f"{t1:.1f}" if t1 >= 0 else "N/A"
            s_t2 = f"{t2:.1f}" if t2 >= 0 else "N/A"
            s_avg = f"{ave_meas_err:.4f}" if ave_meas_err >= 0 else "N/A"
            s_0m1 = f"{p_0m1:.4f}" if p_0m1 >= 0 else "N/A"
            s_1m0 = f"{p_1m0:.4f}" if p_1m0 >= 0 else "N/A"
            s_xerr = f"{x_err:.4e}" if x_err >= 0 else "N/A"
            
            print(f"{q:<6} | {s_t1:<8} | {s_t2:<8} | {s_avg:<10} | {s_0m1:<9} | {s_1m0:<9} | {s_xerr:<10}")
            
    except Exception as e:
        print(f"\n[Warning] Could not extract calibration data: {e}")

def verify_temporal_results(result_data, n_qubits, d_repetitions, qc_used, backend=None):
    """
    Analyzes results including Reset Efficiency and Calibration Data.
    Adds Binomial Error Bars to failure probabilities.
    """
    bs_b = result_data.m_b.get_bitstrings()       
    bs_final_a = result_data.m_a.get_bitstrings() 
    
    histories = []
    for i in range(n_qubits):
        reg_name = f"ca{i}" 
        histories.append(getattr(result_data, reg_name).get_bitstrings())

    # Get Physical Layout
    phys_layout = extract_physical_layout(qc_used)
    phys_a = phys_layout[0:n_qubits]
    phys_b = phys_layout[n_qubits:2*n_qubits]

    print("\n" + "="*80)
    print("VERIFICATION & ANALYSIS")
    print("="*80)
    
    total_shots = len(bs_b)
    threshold = (d_repetitions // 2) + 1
    total_samples = total_shots * n_qubits
    
    ff_logic_errors = 0
    ff_errors_per_qubit = {i: 0 for i in range(n_qubits)}
    reset_failures_per_qubit = {i: 0 for i in range(n_qubits)}
    qubit_stats = []
    for i in range(n_qubits):
        qubit_stats.append({k: 0 for k in range(d_repetitions)})

    print(f"{'Shot':<5} | {'L_Idx':<5} | {'Hist(ca)':<12} | {'Maj':<3} | {'Meas B':<6} | {'Final A':<7} | {'Status'}")
    print("-" * 80)

    printed_samples = 0
    
    for shot_idx in range(total_shots):
        for q_idx in range(n_qubits):
            a_history_str = histories[q_idx][shot_idx]
            final_a_str = bs_final_a[shot_idx]
            meas_final_a = int(final_a_str[-(q_idx+1)])
            b_string = bs_b[shot_idx]
            meas_b = int(b_string[-(q_idx+1)])

            # 1. Measurement Consistency
            n_changes = count_changes(a_history_str)
            qubit_stats[q_idx][n_changes] += 1
            
            # 2. Feed-Forward
            ones_count = a_history_str.count('1')
            maj_val = 1 if ones_count >= threshold else 0
            
            is_ff_error = (meas_b != maj_val)
            if is_ff_error:
                ff_logic_errors += 1
                ff_errors_per_qubit[q_idx] += 1
            
            # 3. Reset
            if meas_final_a == 1:
                reset_failures_per_qubit[q_idx] += 1

            if printed_samples < 15 or is_ff_error:
                if printed_samples < 25: 
                    status = "ERROR" if is_ff_error else "OK"
                    if meas_final_a == 1: status += " (RstFail)"
                    print(f"{shot_idx:<5} | {q_idx:<5} | {a_history_str:<12} | {maj_val:<3} | {meas_b:<6} | {meas_final_a:<7} | {status}")
                    printed_samples += 1

    print("-" * 80)
    
    # Helper for Binomial Error: sqrt( (Nfail * Npass) / Ntotal^3 )
    def calc_rate_err(n_fail, total):
        if total == 0: return 0.0, 0.0
        n_pass = total - n_fail
        rate = n_fail / total
        err = math.sqrt((n_fail * n_pass) / (total**3))
        return rate, err

    # --- SUMMARY A ---
    print("\n--- A) Feed-Forward Error Summary (Per Pair) ---")
    print(f"{'Log Pair':<8} | {'Phys Pair (A->B)':<18} | {'Errors':<7} | {'Rate':<8}  {'+/- sig':<8}")
    print("-" * 65)
    for q_idx in range(n_qubits):
        p_a = f"{phys_a[q_idx]:03d}" if q_idx < len(phys_a) else "???"
        p_b = f"{phys_b[q_idx]:03d}" if q_idx < len(phys_b) else "???"
        errs = ff_errors_per_qubit[q_idx]
        
        rate, err = calc_rate_err(errs, total_shots)
        print(f"{q_idx:<8} | {p_a} -> {p_b:<9} | {errs:<7} | {rate:.2%} +/- {err:.2%}")
        
    print("-" * 65)
    tot_rate, tot_err = calc_rate_err(ff_logic_errors, total_samples)
    print(f"Total FF Errors: {ff_logic_errors} (Fidelity: {1.0-tot_rate:.2%} +/- {tot_err:.2%})")

    # --- SUMMARY B ---
    print("\n--- B) Measurement Repetition Consistency ---")
    max_disp = min(d_repetitions, 10) 
    header_counts = " | ".join([f"{k:>7}" for k in range(max_disp)])
    print(f"{'PhysID':<8} | {'LogID':<5} | {header_counts} | {'Rate(>0)':<8}  {'+/- sig':<8}")
    print("-" * (17 + 10*max_disp + 22))
    
    for q_idx in range(n_qubits):
        phys_id = f"{phys_a[q_idx]:03d}"
        stats = qubit_stats[q_idx]
        counts_str = " | ".join([f"{stats[k]:>7}" for k in range(max_disp)])
        
        unstable = sum(stats[k] for k in range(1, d_repetitions))
        rate, err = calc_rate_err(unstable, total_shots)
        
        print(f"{phys_id:<8} | {q_idx:<5} | {counts_str} | {rate:.2%} +/- {err:.2%}")

    # --- SUMMARY C ---
    print("\n--- C) A-Qubit Reset Fidelity (Target: 0) ---")
    print(f"{'PhysID':<8} | {'LogID':<5} | {'Failures':<10} | {'Rate':<8}  {'+/- sig':<8}")
    print("-" * 52)
    total_reset_fails = 0
    for q_idx in range(n_qubits):
        phys_id = f"{phys_a[q_idx]:03d}"
        fails = reset_failures_per_qubit[q_idx]
        total_reset_fails += fails
        
        rate, err = calc_rate_err(fails, total_shots)
        print(f"{phys_id:<8} | {q_idx:<5} | {fails:<10} | {rate:.2%} +/- {err:.2%}")
        
    print("-" * 52)
    tot_fail_rate, tot_fail_err = calc_rate_err(total_reset_fails, total_samples)
    print(f"Total Reset Failures: {total_reset_fails} (Success: {1.0-tot_fail_rate:.2%} +/- {tot_fail_err:.2%})")
    
    # --- SUMMARY E (Calibration Data) ---
    all_phys_qubits = phys_a + phys_b
    print_calibration_data(backend, all_phys_qubits)

    # --- SUMMARY E ---
    print("\n--- E) Physical Qubit Map ---")
    print(f"Logical A: {phys_a}")
    print(f"Logical B: {phys_b}")


if __name__ == "__main__":
    args = parse_args()
    
    # 1. Build
    qc = build_temporal_majority_circuit(args.size, args.dist, args.reset)
    if args.printCirc: print(qc)
    
    # 2. Backend
    print(f"Running circuit with {args.shots} shots...")
    if args.backend == 'ideal':
        backend = AerSimulator(seed_simulator=args.seed)
        qc_run = qc 
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit import transpile
        service = QiskitRuntimeService()
        print(f'\nUsing real HW backend: {args.backend} ...')
        backend = service.backend(args.backend)
        qc_run = transpile(qc, backend)
        
    print('M: running sampler ...')
    sampler = Sampler(mode=backend)
    
    # 3. Run
    job = sampler.run([qc_run], shots=args.shots)
    result = job.result()
    print('M: job done')
    
    # 4. Verify
    verify_temporal_results(result[0].data, args.size, args.dist, qc_run, backend)
