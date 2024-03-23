#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
minimal demonstrator   reset consusmes memory during training

'''


import pennylane as qml
from pennylane import numpy as np
num_qubit=2 ; shots=1000
dev = qml.device('default.qubit', wires=num_qubit,shots=shots)
from time import time

nReset=30

@qml.qnode(dev)
def circuit(x):
    ang=np.arccos(x)
    qml.RX(ang, wires=0)
    for j in range(nReset):
        m = qml.measure(0,reset=True)
        qml.cond(m, qml.PauliX)(0)
    return qml.expval(qml.PauliZ(0) )

# Initialize parameters
x=-0.33
print(qml.draw(circuit, decimals=2)(x), '\n')
T0=time()
print(' run circ with %d resets ...'%nReset)
y = circuit(x)
elaT=time()-T0
print('input X=%.2f   Y=%.2f   shots=%d  elaT=%.1f sec  nReset=%d'%(x,y,shots,elaT, nReset))

