#!/usr/bin/env python3

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

# Create single-qubit gate circuit
qc1 = QuantumCircuit(1)
qc1.x(0)
qc1.barrier([0])
qc1.s(0)
qc1.y(0)
qc1.barrier([0])
qc1.u(0.4,0.3,.2,qubit=0)
qc1.h(0)
qc1.barrier([0])
qc1.x(0)
qc1.sdg(0)
qc1.barrier([0])
qc1.x(0)
qc1.h(0)
qc1.x(0)
qc1.barrier([0])
qc1.x(0)
qc1.t(0)
qc1.sx(0)
qc1.sdg(0)


print('user defined circuit')
print(qc1)

optLev=1
print('\n Transpile=(Optimize and fuse gates), optLev=',optLev)
#basis_gates=['u1','u2','u3','x']
basis_gates=['u1','u2','u3']
#basis_gates=['u','u2','p']


qc2 = transpile(qc1, basis_gates=basis_gates, optimization_level=optLev)
print(qc2)

basis2_gates=['p','sx']

qc3 = transpile(qc2, basis_gates=basis2_gates, optimization_level=optLev)
print(qc3)


print('\n Alternative, only Decompose() to Us')
qc3=qc1.decompose()
print(qc3)

