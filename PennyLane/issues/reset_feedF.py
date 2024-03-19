#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
minimal demonstrator  reset & feedF not work togerther

'''


import pennylane as qml
from pennylane import numpy as np
num_qubit=2 ; shots=1000; iqt=0
dev = qml.device('default.qubit', wires=num_qubit,shots=shots)

@qml.qnode(dev)
def circuit(x):
    ang=np.arccos(x) ;    qml.RX(ang,0)
    m = qml.measure(0,reset=True)
    qml.cond(m, qml.PauliX)(iqt)
    return qml.expval(qml.PauliZ(iqt) )

x=0.16
print(qml.draw(circuit, decimals=2)(x), '\n')
y = circuit(x)
print('input X=%.2f   Y=%.2f   shots=%d  targetQubit=%d '%(x,y,shots,iqt))

