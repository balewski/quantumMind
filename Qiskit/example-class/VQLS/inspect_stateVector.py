#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
explore state vector operations using qiskit
'''
from qiskit.quantum_info import Statevector


# make 2-qubit state |-> (x) |0> 
psi = Statevector.from_label('-0')   
print('2q psi:',psi)
print(psi.to_dict())

probs = psi.probabilities()
print(' Probabilities for measuring both qubits',probs)
print('as dict',psi.probabilities_dict())

print('trace',psi.trace())


# Probabilities for measuring only qubit-0
probs_qubit_0 = psi.probabilities([0])
print('Qubit-0 probs: {}'.format(probs_qubit_0))
print('as dict',psi.probabilities_dict([0]))

# Probabilities for measuring only qubit-1
probs_qubit_1 = psi.probabilities([1])
print('Qubit-1 probs: {}'.format(probs_qubit_1))

# Probabilities for measuring both qubits
probs = psi.probabilities([0, 1])
print('probs 2q: {}'.format(probs))

# Probabilities for measuring both qubits
# but swapping qubits 0 and 1 in output
probs_swapped = psi.probabilities([1, 0])
print('Swapped probs: {}'.format(probs_swapped))


