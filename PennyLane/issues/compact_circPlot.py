#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
plot circuit in compact form

'''


import pennylane as qml
from pennylane import numpy as np
num_qubit=6 ; shots=1000
dev = qml.device('default.qubit', wires=num_qubit,shots=shots)

@qml.qnode(dev)
def f3(X):
    ninp=X.shape[0]
    ang=np.arccos(X)
    qout=ninp
    mL=[None for i in range(ninp)]
    #.... set and measure input qubits
    for i in range(ninp):  
        qml.RX(ang[i],i)
        mL[i]=qml.measure(i)   
    hw=sum(mL)   #... compute hamming weights
    qml.cond(hw == 1, qml.PauliX)(qout)  # select 1 value of HW
    return qml.probs(ninp)



nSamp=50
X=np.random.uniform(-0.99,0.99,size=(num_qubit-1,nSamp))

print(qml.draw(f3)(X[:,0]))
probs=f3(X)
print(X[:,0], np.mean(X[:,0])); print(probs[0])


