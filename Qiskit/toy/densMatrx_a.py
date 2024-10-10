#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html

import qiskit.quantum_info as qi
from qiskit import  QuantumCircuit
from qiskit_aer import AerSimulator  # needed for qc.set_...
from bigEndianUnitary import print_complex_nice_matrix

num_qubits = 2
rho = qi.random_density_matrix(2 ** num_qubits, seed=100)
qc = QuantumCircuit(num_qubits)
qc.set_density_matrix(rho)
qc.h(0)
qc.save_density_matrix()

print_complex_nice_matrix(rho,'input density matrix')

backend1 = AerSimulator(method="density_matrix")
print('job started,  nq=%d  at %s ...'%(qc.num_qubits,backend1.name))
print(qc)
job = backend1.run(qc)
result = job.result() 

rho2=result.data()['density_matrix']
print_complex_nice_matrix(rho2,'output density matrix')

print('\nsame using sympy.Matrix')
from sympy import Matrix
print(Matrix(rho2))
