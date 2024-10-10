#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
example of state vector computation using Qiskit 1.2
https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html


'''
from qiskit import  QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit_aer import AerSimulator
from pprint import pprint
from bigEndianUnitary import print_complex_nice_vector

# iSWAP matrix operator
iswap_op = Operator([[1, 0, 0, 0],
                     [0, 0, 1j, 0],
                     [0, 1j, 0, 0],
                     [0, 0, 0, 1]])
# Using operators in circuits
nq=2;nb=2
qc = QuantumCircuit(2)

# Add gates
qc.sdg(1)
qc.h(1)
qc.sdg(0)
qc.unitary(iswap_op, [0, 1], label='iswap')
qc.sdg(0)
qc.save_statevector()  # can be inserted in any place in the circuit
qc.sdg(0)

qc.unitary(iswap_op, [0, 1], label='iswap')
qc.s(1)


print(qc)

# Statevector simulation method
backend1 = AerSimulator(method="statevector")
print('job started,  nq=%d  at %s ...'%(qc.num_qubits,backend1.name))

job = backend1.run(qc)
result = job.result() 
#1pprint(result)
#1print('state vect:',result.data())
vec1=result.data()['statevector']
print_complex_nice_vector(vec1,label='out vec')
print('probabilities', result.get_counts())
