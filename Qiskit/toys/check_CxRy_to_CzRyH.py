#!/usr/bin/env python3
"""
Compare and verify the equivalence of two quantum circuit implementations:
1. CX-RY-CX sequence
2. H-CZ-RY-CZ-H sequence

This script demonstrates that these implementations are equivalent up to a global phase.
"""

import numpy as np
from typing import Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# Configuration
PRECISION = 2
TOLERANCE = 1e-8
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=PRECISION)

def are_unitaries_equal_up_to_phase(U1: np.ndarray, U2: np.ndarray) -> bool:
    """
    Check if two unitary matrices are equal up to a global phase.

    Args:
        U1: First unitary matrix
        U2: Second unitary matrix

    Returns:
        bool: True if matrices are equal up to a global phase, False otherwise
    """
    max_index = np.unravel_index(np.argmax(np.abs(U1)), U1.shape)
    
    if np.abs(U1[max_index]) < TOLERANCE or np.abs(U2[max_index]) < TOLERANCE:
        return np.allclose(U1, U2)
    
    phase = U1[max_index] / U2[max_index]
    print('phase=%.2f'%(phase))
    return np.allclose(U1, U2 * phase, atol=TOLERANCE)

def create_circuit_cx(th1: float, th2: float) -> QuantumCircuit:
    """
    Create a quantum circuit using CX gates.

    Args:
        th1: First rotation angle (unused in current implementation)
        th2: Second rotation angle

    Returns:
        QuantumCircuit: Circuit with CX-RY-CX sequence
    """
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.ry(th2, 1)
    qc.cx(0, 1)    
    return qc

def create_circuit_cz(th1: float, th2: float) -> QuantumCircuit:
    """
    Create a quantum circuit using CZ gates.

    Args:
        th1: First rotation angle (unused in current implementation)
        th2: Second rotation angle

    Returns:
        QuantumCircuit: Circuit with H-CZ-RY-CZ-H sequence
    """
    qc = QuantumCircuit(2)
    qc.h(1)
    qc.cz(0, 1)
    qc.ry(-th2, 1) 
    qc.cz(0, 1)
    qc.h(1)
    return qc

def compare_circuits(th1: float, th2: float) -> bool:
    """
    Compare the unitary matrices of two circuit implementations.

    Args:
        th1: First rotation angle
        th2: Second rotation angle

    Returns:
        bool: True if circuits are equivalent up to global phase
    """
    circuit_cx = create_circuit_cx(th1, th2)
    circuit_cz = create_circuit_cz(th1, th2)

    unitary_cx = Operator(circuit_cx).data
    unitary_cz = Operator(circuit_cz).data

    print("CX circuit unitary:\n", unitary_cx)
    print("CZ circuit unitary:\n", unitary_cz)
    
    print(circuit_cx)
    print(circuit_cz)

    are_equal = are_unitaries_equal_up_to_phase(unitary_cx, unitary_cz)
    print('Circuits equal up to global phase=%s'%(str(are_equal)))
    
    return are_equal

def main():
    """Run the circuit comparison tests with random angles."""
    np.random.seed(43)  # for reproducibility
    num_tests = 4
    
    for i in range(num_tests):
        th1 = np.random.uniform(0, 2*np.pi)
        th2 = np.random.uniform(0, 2*np.pi)
        
        print('\nTest %d: Ry(theta1=%.3f) and Ry(theta2=%.3f)'%(i+1, th1, th2))
        if compare_circuits(th1, th2):
            print("✓ Circuits are equivalent!")
        else:
            print("✗ Circuit equivalence test failed!")
            return
        print("-" * 50)
    
    print('✓ All %d test pairs passed!'%(num_tests))

if __name__ == "__main__":
    main()
