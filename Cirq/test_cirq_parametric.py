import cirq
import numpy as np
import sympy
from typing import List
import argparse

def create_parameterized_circuit(n_layers: int = 3):
    """Create a parameterized circuit with RY gates on q0 interleaved with CZ gates."""
    # Create 2 qubits
    q0, q1 = cirq.LineQubit.range(2)
    
    # Create circuit
    circuit = cirq.Circuit()
    
    # Create parameter symbols for vectorized phi
    phi_params = sympy.symbols(f'phi_0:{n_layers}')
    
    # Build the circuit with interleaved RY and CZ gates
    for i in range(n_layers):
        # Apply parameterized RY gate to q0
        circuit.append(cirq.ry(phi_params[i]).on(q0))
        
        # Apply CZ gate between q0 and q1
        circuit.append(cirq.CZ(q0, q1))
    
    # Add measurements
    circuit.append(cirq.measure(q0, q1, key='result'))
    
    return circuit, phi_params

def bind_random_parameters(circuit, phi_params: List[sympy.Symbol], seed: int = None):
    """Bind random values to the parameterized angles phi."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random angles between 0 and 2π
    random_angles = np.random.uniform(0, 2 * np.pi, len(phi_params))
    
    # Create parameter resolver
    param_resolver = {phi_params[i]: random_angles[i] for i in range(len(phi_params))}
    
    # Resolve the parameters in the circuit
    resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
    
    return resolved_circuit, random_angles

def run_simulation(circuit, n_shots: int = 1024):
    """Run the circuit simulation with ideal simulator."""
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=n_shots)
    return result

def analyze_results(result, n_shots: int, random_angles: np.ndarray):
    """Analyze and print the measurement results."""
    # Get histogram of results
    counts = result.histogram(key='result')
    
    # Calculate probabilities
    probabilities = {}
    for state in range(4):  # 2 qubits = 4 possible states
        count = counts.get(state, 0)
        probabilities[state] = count / n_shots
    
    print("\nMeasurement Results:")
    print("-" * 50)
    print(f"{'State':<10} {'Count':<10} {'Probability':<12}")
    print("-" * 50)
    
    for state in sorted(probabilities.keys()):
        state_str = f"|{state:02b}⟩"
        count = counts.get(state, 0)
        prob = probabilities[state]
        if prob > 0.001:  # Only print states with probability > 0.1%
            print(f"{state_str:<10} {count:<10} {prob:.6f}")
    
    print("\nBound Random Angles (phi vector):")
    print("-" * 50)
    for i, angle in enumerate(random_angles):
        print(f"phi[{i}] = {angle:.6f} rad ({np.degrees(angle):.2f}°)")

def get_statevector(circuit, phi_params: List[sympy.Symbol], random_angles: np.ndarray):
    """Get the statevector before measurement."""
    # Create circuit without measurement
    circuit_no_measure = cirq.Circuit()
    q0, q1 = cirq.LineQubit.range(2)
    
    # Rebuild circuit without measurement
    param_resolver = {phi_params[i]: random_angles[i] for i in range(len(phi_params))}
    
    for moment in circuit[:-1]:  # Skip the measurement moment
        circuit_no_measure.append(moment)
    
    resolved_circuit = cirq.resolve_parameters(circuit_no_measure, param_resolver)
    
    # Simulate to get statevector
    simulator = cirq.Simulator()
    result = simulator.simulate(resolved_circuit)
    statevector = result.final_state_vector
    
    print("\nStatevector (before measurement):")
    print("-" * 50)
    for i, amp in enumerate(statevector):
        if abs(amp) > 0.001:
            state_str = f"|{i:02b}⟩"
            print(f"{state_str}: {amp:.6f}")
    
    return statevector

def main():
    parser = argparse.ArgumentParser(description='Parameterized circuit with RY and CZ gates')
    parser.add_argument('--layers', type=int, default=3,
                       help='Number of RY-CZ layers (default: 3)')
    parser.add_argument('--shots', type=int, default=1024,
                       help='Number of measurement shots (default: 1024)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for parameter generation (default: 42)')
    parser.add_argument('--show-statevector', action='store_true',
                       help='Show the statevector before measurement')
    
    args = parser.parse_args()
    
    print(f"Creating parameterized circuit with {args.layers} layers")
    print(f"Using ideal simulator with all-to-all connectivity")
    print(f"Random seed: {args.seed}")
    
    # Create parameterized circuit
    circuit, phi_params = create_parameterized_circuit(args.layers)
    
    print("\nParameterized Circuit (symbolic):")
    print(circuit)
    
    # Bind random parameters
    resolved_circuit, random_angles = bind_random_parameters(circuit, phi_params, args.seed)
    
    print("\nResolved Circuit (with bound parameters):")
    print(resolved_circuit)
    
    # Get statevector if requested
    if args.show_statevector:
        statevector = get_statevector(circuit, phi_params, random_angles)
    
    # Run simulation
    print(f"\nRunning simulation with {args.shots} shots...")
    result = run_simulation(resolved_circuit, args.shots)
    
    # Analyze results
    analyze_results(result, args.shots, random_angles)

if __name__ == "__main__":
    main()
