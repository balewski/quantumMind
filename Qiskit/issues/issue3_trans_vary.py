#!/usr/bin/env python3

from qiskit.test.mock import FakeLima
from qiskit.providers.aer import AerSimulator
import qiskit as qk
from qiskit.converters import circuit_to_dag

device_backend = FakeLima()
backend=AerSimulator.from_backend(device_backend)
seed=111
print('backend=',backend,seed)


qc1=qk.QuantumCircuit.from_qasm_file('circ28.qasm')
print(qc1)
qc1t=qk.transpile(qc1, backend=backend, optimization_level=3, seed_transpiler=seed)


qc2=qk.QuantumCircuit.from_qasm_file('circ29.qasm')

print(qc2)
qc2t=qk.transpile(qc2, backend=backend, optimization_level=3, seed_transpiler=seed
)

sum1=str(qc1t.count_ops())
sum2=str(qc2t.count_ops())

print('qc1 transp depth=',qc1t.depth(),sum1)
print('qc2 transp depth=',qc2t.depth(),sum2)
assert qc1t.depth()==qc2t.depth()  #test
assert sum1==sum2 # test 2 count gates and all other details
print('M:ok')

