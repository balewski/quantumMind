#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


from qiskit import  transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit import qpy
inpF='grover_june13b.qpy'
with open(inpF, 'rb') as fd:
    qc=qpy.load(fd)[0]
print(qc)

backend = AerSimulator()
qcT1 = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42)
print(qcT1)

qcT2 = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42,basis_gates=['cz','u'])
#print(qcT2)
opsOD=qcT2.count_ops()  # ordered dict
print('ops ideal:',opsOD)

qpuN='ibm_aachen'
service = QiskitRuntimeService()
backend = service.backend(qpuN)
print('use true HW backend =', backend.name)
qcT3 = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42)
opsOD=qcT3.count_ops()  # ordered dict
print('ops %s:'%backend.name,opsOD)

