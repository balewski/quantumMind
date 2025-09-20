#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Error mitigation examples with state vector and density matrix methods (partial implementation)

Example of state vector computation using Qiskit 1.2
https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html
"""
from qiskit import  QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit_aer import AerSimulator
from pprint import pprint
from bigEndianUnitary import print_complex_nice_matrix, print_complex_nice_vector, convert_big_to_little_endian
import numpy as np


cnotM=np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,0,1],
                [0,0,1,0]]) ; name='myCNOT'

nq=2
opSmall=convert_big_to_little_endian(nq,cnotM)
myOp = Operator(opSmall)

# Using operators in circuits
print('before (bigEndian):'); print_complex_nice_matrix(cnotM)
print('after:(smallEndian)'); print_complex_nice_matrix(opSmall)


qc = QuantumCircuit(2)

# Add gates
#qc.rx(0.2,0)
qc.x(0)
qc.x(1)
qc.unitary(myOp, [0, 1], label=name)
qc.save_statevector()  # can be inserted in any place in the circuit

print(qc)

backend = AerSimulator(method="statevector")
print('job started,  nq=%d  at %s ...'%(qc.num_qubits,backend.name))

# Statevector simulation method
sim_statevector = AerSimulator(method='statevector')
job = sim_statevector.run(qc)
result = job.result() 
#1pprint(result)
print('state vect:') #,result.data())
outVec=result.data()['statevector']
print_complex_nice_vector(outVec)
print('probabilities', result.get_counts())

