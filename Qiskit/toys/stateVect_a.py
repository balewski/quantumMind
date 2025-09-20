#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Advanced state vector simulation with iSWAP operators and custom circuit construction
"""

# https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html

import qiskit.quantum_info as qi
from qiskit import  QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator  # needed for qc.set_statevector(psi)
from numpy import sqrt

u = qi.Statevector([1 / sqrt(2), 1 / sqrt(2)])
v = qi.Statevector([(1 + 2.0j) / 3, -2 / 3])
w = qi.Statevector([1 / 3, 2 / 3])

print("State vectors u, v, and w have been defined.")
print(v.draw("text"))


num_qubits = 2
psi = qi.random_statevector(2 ** num_qubits, seed=100)

# Set initial state to generated statevector
qc = QuantumCircuit(num_qubits)
qc.set_statevector(psi)
qc.save_state()

# Transpile for simulator
backend = AerSimulator(method='statevector')

pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
qcT = pm.run(qc)
print('transpiled for',backend)
print(qcT.draw('text', idle_wires=False))

# Run and get saved data
result = backend.run(qcT).result()

print('state vect:',result.data())
print('probabilities', result.get_counts())
