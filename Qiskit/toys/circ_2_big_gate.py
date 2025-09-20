#!/usr/bin/env python3
__author__ = "Jan Balewski, ChatGPT"
__email__ = "janstar1122@gmail.com"

"""
Create custom quantum gates (T-gates) and compose them into larger circuit structures
"""



import random
import numpy as np
from qiskit import QuantumCircuit# , Aer, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

#...!...!....................
def magicUk_specific(n, qubit_list):
    """
    Create a QuantumCircuit with n qubits where T-gates are applied
    to qubits specified in qubit_list.

    Parameters:
    n (int): Total number of qubits.
    qubit_list (list): List of qubits to apply the T-gate on.

    Returns:
    QuantumCircuit: The quantum circuit implementing the unitary.
    """
    if not all(0 <= q < n for q in qubit_list):
        raise ValueError("All qubits in qubit_list must be within the range 0 to n-1.")

    qc = QuantumCircuit(n)
    for qubit in qubit_list:
        qc.t(qubit)
    return qc


#=================================
#  M A I N
#=================================
if __name__ == "__main__":

    n = 5
    qubit_list = [0, 1, 4]

    qc = magicUk_specific(n, qubit_list)
    print(qc.draw(output='text'))

    # Obtain the unitary matrix and assign it to 'Uk'
    Uk_mat = Operator(qc).data

    # Convert the circuit to a gate
    Uk_op = qc.to_gate(label='Uk')
    Uk_op_inv=Uk_op.inverse()
    Uk_op_inv.label='Uk_inv'

    
    # Initialize a new quantum circuit with n qubits
    qc2 = QuantumCircuit(n)

    # Apply the unitary gate three times
    for _ in range(3):
        qc2.append(Uk_op, qargs=range(n))
    qc2.append(Uk_op_inv, qargs=range(n))

    # Visualize the new circuit
    print("New Quantum Circuit:")
    print(qc2.draw(output='text'))
