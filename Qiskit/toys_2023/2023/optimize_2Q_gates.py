#!/usr/bin/env python3

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

# Create single-qubit gate circuit
qc1 = QuantumCircuit(2)
qc1.x(0)
qc1.s(0)
qc1.h(1)
qc1.barrier([0, 1])
qc1.z(0)
qc1.t(0)
qc1.h(1)
qc1.sdg(1)
qc1.cx(0,1)
qc1.cx(0,1)
qc1.h(1)

print('qc1=')
print(qc1)

optLev=1
print('\n Transpile=(Optimize and fuse gates) in qc1, optLev=',optLev)
basis_gates=['u3','cx']

qc2 = transpile(qc1, basis_gates=basis_gates, optimization_level=optLev)
print(qc2)

print('\n only Decompose() qc1')
qc3=qc1.decompose()
print(qc3)

