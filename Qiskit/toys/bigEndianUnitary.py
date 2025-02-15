#!/usr/bin/env python3
__author__ = "Jan Balewski, ChatGPT"
__email__ = "janstar1122@gmail.com"

import numpy as np
from scipy.stats import unitary_group

#...!...!....................
def is_unitary(matrix):
    """Check if a matrix is unitary: U * U^dagger = I"""
    return np.allclose(np.eye(matrix.shape[0]), matrix @ np.conj(matrix.T))

#...!...!....................
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
def print_complex_nice_vector(vector,label=None, tol=1e-10):
    # Set print options for better readability
    np.set_printoptions(precision=2, suppress=True)

    # Custom format for complex numbers with consistent width (10 characters, including signs and decimals)
    def complex_formatter_1d(z):
        real_part = f"{z.real:7.2f}" if abs(z.real) > tol else "   0    "
        imag_part = f"{z.imag:+7.2f}j" if abs(z.imag) > tol else "        "
        return f"{real_part}{imag_part}"

    # Apply formatting to the vector
    formatted_vector = np.vectorize(complex_formatter_1d)(vector)

    if label!=None:    print('\n%s:'%label, vector.shape)
    # Print the vector as a column with row indices
    for idx, val in enumerate(formatted_vector):
        print(f"{idx:2d}: {val}")
        
#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    args=commandline_parser()

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

    unitary_big[1,1]=0.1j
    print_complex_nice(unitary_big)
    print("\nConverted matrix in little-endian convention:")
    print_complex_nice(unitary_little)
