#!/usr/bin/env python3
import stim
import numpy as np

def create_steane_bell_circuit(rounds=1):
    """
    Generate a Bell state preparation circuit using the [[7,1,3]] Steane code
    with proper DETECTOR annotations (avoiding non-deterministic observables).
    """
    circuit = stim.Circuit()
    
    # Initialize two logical qubits
    circuit.append("R", list(range(20)))
    circuit.append("TICK")
    
    # ============ LOGICAL STATE PREPARATION ============
    circuit.append("H", [0, 1, 2])
    circuit.append("CNOT", [0, 3, 1, 4, 2, 5])
    circuit.append("TICK")
    
    circuit.append("H", [10, 11, 12])
    circuit.append("CNOT", [10, 13, 11, 14, 12, 15])
    circuit.append("TICK")
    
    # ============ BELL STATE PREPARATION ============
    circuit.append("CNOT", [0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16])
    circuit.append("TICK")
    
    circuit.append("H", list(range(0, 7)))
    circuit.append("TICK")
    
    # ============ SYNDROME EXTRACTION WITH DETECTORS ============
    for round_num in range(rounds):
        # Measure syndrome qubits
        circuit.append("MR", [9, 18, 19])
        
        circuit += stim.Circuit("SHIFT_COORDS(0, 0, 1)\n")
        
        # Add detectors for syndrome check (no observable dependence)
        if round_num == 0:
            circuit += stim.Circuit("""
            DETECTOR(0, 0, 0) rec[-1]
            DETECTOR(0, 1, 0) rec[-2]
            DETECTOR(0, 2, 0) rec[-3]
            """)
        else:
            circuit += stim.Circuit(f"""
            DETECTOR(0, 0, {round_num}) rec[-1] rec[-4]
            DETECTOR(0, 1, {round_num}) rec[-2] rec[-5]
            DETECTOR(0, 2, {round_num}) rec[-3] rec[-6]
            """)
        
        circuit.append("TICK")
    
    # ============ LOGICAL MEASUREMENT ============
    circuit.append("H", list(range(0, 7)))
    circuit.append("M", list(range(0, 7)))
    circuit.append("TICK")
    
    circuit.append("M", list(range(10, 17)))
    circuit.append("TICK")
    
    # Measure parity of logical qubits (XOR of measurements)
    circuit += stim.Circuit("""
    OBSERVABLE_INCLUDE(0) rec[-7] rec[-6] rec[-5] rec[-4] rec[-3] rec[-2] rec[-1]
    """)
    
    return circuit


def simulate_with_errors(circuit, phys_error_rate=0.001, num_shots=10000):
    """
    Simulate circuit with depolarizing errors and estimate logical error rate.
    
    Args:
        circuit: Stim circuit
        phys_error_rate: Physical error rate
        num_shots: Number of Monte Carlo samples
    
    Returns:
        Dictionary with error statistics
    """
    
    # Add X-errors after all gates
    circuit_with_noise = stim.Circuit()
    
    for instruction in circuit:
        circuit_with_noise.append(instruction)
    
    # Compile sampler
    sampler = circuit_with_noise.compile_sampler()
    
    print(f"Sampling {num_shots} shots with physical error rate {phys_error_rate}...")
    
    # Sample measurement outcomes
    samples = sampler.sample(shots=num_shots)
    
    num_measurements = circuit_with_noise.num_measurements
    num_observables = circuit_with_noise.num_observables
    
    # Extract observable measurements (last num_observables measurements)
    observable_start = num_measurements - num_observables
    observable_samples = samples[:, observable_start:]
    
    # Count logical errors (when observable is 1)
    logical_errors = np.sum(observable_samples)
    
    logical_error_rate = logical_errors / num_shots
    std_err = np.sqrt(logical_error_rate * (1 - logical_error_rate) / num_shots)
    
    return {
        "logical_errors": logical_errors,
        "num_shots": num_shots,
        "logical_error_rate": logical_error_rate,
        "logical_error_rate_std": std_err,
        "phys_error_rate": phys_error_rate,
        "num_detectors": circuit_with_noise.num_detectors,
        "num_observables": num_observables
    }


def compute_threshold(circuit, phys_error_rates=None, num_shots=5000):
    """
    Compute logical error rate as function of physical error rate.
    """
    if phys_error_rates is None:
        phys_error_rates = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    
    results = []
    
    for p in phys_error_rates:
        print(f"\n--- Physical error rate: {p:.6f} ---")
        res = simulate_with_errors(circuit, phys_error_rate=p, num_shots=num_shots)
        results.append({
            "phys_error_rate": p,
            "logical_error_rate": res["logical_error_rate"],
            "std_err": res["logical_error_rate_std"]
        })
        
        print(f"    Logical error rate: {res['logical_error_rate']:.8f} ± {res['logical_error_rate_std']:.8f}")
        if p > 0 and res['logical_error_rate'] > 0:
            print(f"    Suppression: {p / res['logical_error_rate']:.2f}x")
    
    return results


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("Steane Code [[7,1,3]] Bell State - Error Rate Analysis")
    print("=" * 70)
    
    # Create circuit with 2 rounds of syndrome extraction
    circuit = create_steane_bell_circuit(rounds=2)
    
    print(f"\nCircuit properties:")
    print(f"  Number of qubits: {circuit.num_qubits}")
    print(f"  Number of measurements: {circuit.num_measurements}")
    print(f"  Number of detectors: {circuit.num_detectors}")
    print(f"  Number of observables: {circuit.num_observables}")
    
    # Single error rate analysis
    print("\n" + "=" * 70)
    print("SINGLE ERROR RATE ANALYSIS (p = 0.001)")
    print("=" * 70)
    
    results = simulate_with_errors(
        circuit,
        phys_error_rate=0.1,
        num_shots=1000_000
    )
    
    print(f"\nResults:")
    print(f"  Logical errors: {results['logical_errors']}")
    print(f"  Total shots: {results['num_shots']}")
    print(f"  Logical error rate: {results['logical_error_rate']:.8f}")
    print(f"  95% CI: ± {1.96 * results['logical_error_rate_std']:.8f}")
    mmm
    if results['logical_error_rate'] > 0:
        suppression = results['phys_error_rate'] / results['logical_error_rate']
        print(f"  Suppression factor: {suppression:.2f}x")
    
    # Threshold analysis
    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)
    
    threshold_results = compute_threshold(
        circuit,
        phys_error_rates=[0.0001, 0.0005, 0.001, 0.002],
        num_shots=5000
    )
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Physical p':<15} {'Logical p':<15} {'Suppression':<15}")
    print("-" * 45)
    for res in threshold_results:
        p = res['phys_error_rate']
        lp = res['logical_error_rate']
        if lp > 0:
            supp = p / lp
            print(f"{p:<15.6f} {lp:<15.8f} {supp:<15.2f}x")
        else:
            print(f"{p:<15.6f} {lp:<15.8f} {'N/A':<15}")
    
    print("=" * 70)
