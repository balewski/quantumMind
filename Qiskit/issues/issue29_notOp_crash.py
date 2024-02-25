#!/usr/bin/env python3
'''  transpiler crashes  on hw3N=expr.bit_not(hw3)

'''

from qiskit import   QuantumCircuit
from pprint import pprint

from qiskit.circuit.classical import expr

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile

def feedF_circuit_hamming_weight(n=4, k=2):
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n)
    qc = QuantumCircuit(qr, cr)

    # Apply Hadamard gates to the first n-1 qubits to create superpositions
    for i in range(n - 1):
        qc.h(i)
    qc.barrier()

    # Measure the first n-1 qubits
    for i in range(n - 1):
        qc.measure(i, i)
    qc.barrier()

    # Create intermediate variables for all partial bit values
    b01 = expr.bit_and(cr[0], cr[1])  # Bitwise AND of cr[0] and cr[1]
    b02 = expr.bit_and(cr[0], cr[2])  # Bitwise AND of cr[0] and cr[2]
    b12 = expr.bit_and(cr[1], cr[2])  # Bitwise AND of cr[1] and cr[2]
    hw3= expr.bit_or(b01, cr[2])
    hw3N=expr.bit_not(hw3)  <== CULPRIT
    
    
    # Intermediate ORs
    b01_or_b02 = expr.bit_or(b01, b02)  # Bitwise OR of b01 and b02
    final_or = expr.bit_or(b01_or_b02, b12)  # Final OR for the condition

    # Create intermediate variables for XOR operations to exclude all 1s case
    b01_xor_b02 = expr.bit_xor(cr[0], cr[1])  # Bitwise XOR of cr[0] and cr[1]
    all_xor = expr.bit_xor(b01_xor_b02, cr[2])  # Bitwise XOR of the result with cr[2]

    # Combine OR and XOR results to form the condition for HW=2 excluding all 1s
    hw2=expr.bit_and(final_or, hw3N)
    
    # Use if_test for conditional execution based on the Hamming weight
    with qc.if_test(hw2):
        qc.x(qr[3])  # Apply X gate to the last qubit if Hamming weight is exactly 2
    
    # Measure the last qubit
    qc.measure(n - 1, n - 1)

    return qc



# -------Create a Quantum Circuit 
qc=feedF_circuit_hamming_weight()

print(qc)

from qiskit_aer import AerSimulator
backend = AerSimulator()

job =  backend.run(qc,shots=1000)
result = job.result()
counts=result.get_counts(0)
print('M: counts:', counts)

print('M:ok')

