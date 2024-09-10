#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
user defined unitary as sparse matrix
- correct unitarity
- save as qasm
- run BQskit on QASM
- read as qiskit and print

Problem:  order of indices ???
'''

import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import qr

# Step 1: Define the sparse matrix
data = np.array([1, 1, 0.71, 0.71j,  0.71j, 0.71,  1, 1, 1, 1 ], dtype=complex)
rows = [0, 1, 2, 2,  3, 3, 4, 5, 6, 7]
cols = [0, 1, 4, 5,  4, 5, 2, 3, 6, 7]

# Create the sparse matrix in COO format
size = 8  # Assuming 8x8 matrix (based on indices)
sparse_matrix = coo_matrix((data, (rows, cols)), shape=(size, size))

# Convert to dense format for processing
dense_matrix = sparse_matrix.toarray()

# Step 2: Apply QR decomposition to make the matrix unitary
Q, R = qr(dense_matrix)

# Q is now a unitary matrix

# Step 3: Print the unitary matrix
print("Corrected Unitary Matrix:")
print(Q.round(3))

# Step 4: Verify unitarity: Q^dagger * Q should be identity
identity_check = np.allclose(np.dot(Q.conj().T, Q), np.eye(size))
print("\nIs the matrix unitary? ", identity_check)

# Step 3: Check unitarity: Q^dagger * Q should be identity
unitarity_check = np.dot(Q.conj().T, Q)
identity_matrix = np.eye(size)

# Step 4: Calculate the largest difference between the expected identity matrix and Q^dagger * Q
largest_difference = np.max(np.abs(unitarity_check - identity_matrix))
print('max diff:',largest_difference)


from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# Step 1: Initialize a Quantum Circuit with 3 qubits
qc = QuantumCircuit(3)

# Step 2: Apply the unitary matrix Q to the quantum circuit
unitary_matrix = Operator(Q)  # Convert the matrix Q to a Qiskit Operator
qc.unitary(unitary_matrix, [0, 1, 2], label='Unitary')

# Step 3: Display the quantum circuit
print(qc.draw('text'))

from qiskit import qasm2
qasmF='aa1.qasm'
# Convert the quantum circuit to an OpenQASM string
qasm_str = qasm2.dump(qc,qasmF)

print('M: saved qc to:',qasmF)
from bqskit import compile, Circuit

# Load a circuit from QASM
qcBQ = Circuit.from_file(qasmF)

qcSyn = compile(qcBQ)
print(qcSyn)
qasmF2='aa2.qasm'
# Save output as QASM
qcSyn.save(qasmF2)

print('M: saved qcSyn to:',qasmF2)

qc2=qasm2.load(qasmF2)
print(qc2)
