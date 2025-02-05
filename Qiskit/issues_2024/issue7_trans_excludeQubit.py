#!/usr/bin/env python3

from qiskit.test.mock import FakeLima
from qiskit.providers.aer import AerSimulator
import qiskit as qk
from qiskit.converters import circuit_to_dag

device_backend = FakeLima()
backend=AerSimulator.from_backend(device_backend)

print('backend=',backend,seed)
qc1=qk.QuantumCircuit.from_qasm_file('circ28.qasm')
print(qc1)
seed=111
iniQ=[3,4,5] #initial_layout'

qc1t=qk.transpile(qc1, backend=backend, optimization_level=3, seed_transpiler=111)


print('M:ok')

