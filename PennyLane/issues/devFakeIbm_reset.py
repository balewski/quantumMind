#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''


'''


import pennylane as qml
from pennylane import numpy as np
num_qubit=3 ; shots=11000

if 0: 
    from qiskit.providers.aer import AerSimulator
    from qiskit.providers.fake_provider  import FakeHanoi
    fake_hanoi_backend = FakeHanoi()
    aer_simulator = AerSimulator.from_backend(fake_hanoi_backend)
    dev = qml.device('qiskit.aer', wires=num_qubit, backend=aer_simulator, shots=shots)
else:
    dev = qml.device('default.qubit', wires=num_qubit,shots=shots)

print('dev=',dev)
@qml.qnode(dev)
def circuit(x):
    ang=np.arccos(x) ;    qml.RY(ang,0)
    m = qml.measure(0,reset=True)
    qml.cond(m, qml.PauliX)(0)
    return qml.expval(qml.PauliX(0) )

x=0.5
print(qml.draw(circuit, decimals=2)(x), '\n')
y = circuit(x)
print('input X=%.3f   Y=%.3f   shots=%d   '%(x,y,shots))

