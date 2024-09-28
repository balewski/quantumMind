import numpy as np
from scipy.stats import unitary_group

def is_unitary(matrix):
    """Check if a matrix is unitary: U * U^dagger = I"""
    return np.allclose(np.eye(matrix.shape[0]), matrix @ np.conj(matrix.T))

def convert_big_to_little_endian(nq, unitary_big):
    """Convert a unitary matrix from big-endian to little-endian for nq qubits"""
    dim = 2 ** nq  # Dimension of the unitary matrix
    indices = np.arange(dim)
    
    # Calculate the permutation to swap qubit indices for big-endian to little-endian conversion
    permuted_indices = np.array([int(f"{i:0{nq}b}"[::-1], 2) for i in indices])
    
    # Apply the permutation to the rows and columns
    P = np.eye(dim)[permuted_indices]  # Permutation matrix
    print('P:',P)
    unitary_little = P @ unitary_big @ P.T
    
    return unitary_little

# Number of qubits
nq = 3

# Generate a random unitary matrix of dimension 2^nq
dim = 2 ** nq
unitary_big = unitary_group.rvs(dim)  # Generates a random unitary matrix

# Verify that the generated matrix is unitary
if is_unitary(unitary_big):
    print("The generated matrix is unitary.")
else:
    print("The generated matrix is NOT unitary.")

# Convert the unitary matrix from big-endian to little-endian convention
unitary_little = convert_big_to_little_endian(nq, unitary_big)

# Verify that the converted matrix is still unitary
if is_unitary(unitary_little):
    print("The converted matrix is still unitary.")
else:
    print("The converted matrix is NOT unitary.")

# Output the matrices
print("\nOriginal matrix in big-endian convention:")
print(unitary_big)
print("\nConverted matrix in little-endian convention:")
print(unitary_little)
