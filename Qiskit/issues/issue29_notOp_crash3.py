#!/usr/bin/env python3
'''  transpiler crashes  on hw3N=expr.bit_not(hw3)
  This code crashes on both  local simu & HW, FIXED

Aziz: Qiskit is putting out if containing the bit_not operator on a single bit as if (~c[0]) for the backend side to read, but the backend side seems to not like the ~ on a single bit. We recommend you to use hw3N=expr.logic_not(hw3) since for the case of single bit they are often quite similar. It is important to keep in mind that qiskit expressions are different than hardware capabilities.
In Aer, I asked if they can fix this issue but in the meantime you can use hw3N=expr.logic_not(hw3)or casting:
from qiskit.circuit.classical import expr, types
hw3N = expr.cast(expr.bit_not(expr.cast(hw3, types.Uint(1))), types.Bool())



'''

from qiskit import   QuantumCircuit
from pprint import pprint

from qiskit.circuit.classical import expr

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile

#...!...!....................
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
    hw3= expr.bit_and(b01, cr[2])
    #hw3N=expr.bit_not(hw3) # <== CULPRIT
    hw3N=expr.logic_not(hw3)
    
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
shots=4000

if 1:  # do HW
    print('M: access QiskitRuntimeService()...')
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    backN="ibm_cusco"
    backend = service.get_backend(backN)
    if "if_else" not in backend.target:
        from qiskit.circuit import IfElseOp
        backend.target.add_instruction(IfElseOp, name="if_else")
    print('M: transpiling for ...',backend)
    qcT = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42)
    print('M: executing ...')
    job=backend.run(qcT,shots=shots, dynamic=True)
else:
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    job =  backend.run(qc,shots=shots)
    
result = job.result()
counts=result.get_counts(0)
print('M: counts:', counts)

print('M:ok',backend)

