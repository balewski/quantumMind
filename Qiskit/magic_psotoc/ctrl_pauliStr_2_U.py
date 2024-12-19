#!/usr/bin/env python3
__author__ = "Jan Balewski, ChatGPT"
__email__ = "janstar1122@gmail.com"

import random
import numpy as np
from qiskit import QuantumCircuit# , Aer, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

#...!...!....................
def print_complex_nice_matrix(array,label=None, tol=1e-10):
    # Set print options for better readability
    np.set_printoptions(precision=2, suppress=True)

    # Custom format for complex numbers with consistent width (10 characters, including signs and decimals)
    def complex_formatter_2d(z):
        real_part = f"{z.real:7.2f}" if abs(z.real) > tol else "   0   "
        imag_part = f"{z.imag:+7.2f}j" if abs(z.imag) > tol else "        "
        return f"{real_part}{imag_part}"

    # Apply formatting to the array
    formatted_array = np.vectorize(complex_formatter_2d)(array)


    # density_matrix.get_dimensions()
    if label!=None:
        if isinstance(array, np.ndarray):
            print('\n%s:'%label, array.shape)
        else:
             print('\n%s:'%label)

    # Print array row by row
    for row in formatted_array:
        print('  '.join(row))
        
#...!...!....................
def random_pauli_string(n):
    """Generate a random Pauli string of length n."""
    paulis = ['I', 'X', 'Y', 'Z']
    pauli_string = [random.choice(paulis) for _ in range(n)]
    return pauli_string

#...!...!....................
def controlled_pauli_circuit(pauli_string):
    """
    Create a QuantumCircuit that:
    - Initializes qubit 0 in the |+⟩ state.
    - Applies controlled-Pauli gates from qubit 0 to qubits 1 to n.
    """
    n = len(pauli_string)
    total_qubits = n + 1  # Including the control qubit
    qc = QuantumCircuit(total_qubits)
    
    # Initialize qubit 0 in the |+⟩ state
    qc.h(0)
    
    for i, p in enumerate(pauli_string):
        target_qubit = i + 1  # Since qubit 0 is the control qubit
        if p == 'I':
            pass  # Identity operator; no action needed
        elif p == 'X':
            qc.cx(0, target_qubit)
        elif p == 'Y':
            qc.cy(0, target_qubit)
        elif p == 'Z':
            qc.cz(0, target_qubit)
        else:
            raise ValueError(f"Invalid Pauli operator: {p}")
    return qc

#=================================
#  M A I N
#=================================
if __name__ == "__main__":

    # Set the number of qubits (excluding the control qubit)
    n = 2

    # Generate a random Pauli string
    pauli_str = random_pauli_string(n)
    print("Random Pauli string:", pauli_str)

    # Create the quantum circuit
    qc = controlled_pauli_circuit(pauli_str)
    qc.save_unitary()
    # Visualize the circuit
    print("\nQuantum Circuit:")
    print(qc.draw(output='text'))

    # Optional: Obtain the unitary matrix
    backend = AerSimulator()#method='unitary')
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    qcT = pm.run(qc)
    print('transpiled for',backend)
    print(qcT.draw('text', idle_wires=False))

    job = backend.run(qcT)
    result = job.result()
    U = result.get_unitary(qcT)

    print("\nUnitary Matrix:")

    print_complex_nice_matrix(U,'ideal')
