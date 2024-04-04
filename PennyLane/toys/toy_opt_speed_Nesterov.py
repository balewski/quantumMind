#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
compares speed for 
dev: state based, circuit execution:  vectorized  --> 0.05 sec/step
dev: state based, circuit execution:  list compreh.  --> 3.4 sec/step  
dev: shot-based, circuit execution:  list compreh. --> 14 sec/step 
dev: shot-based, circuit execution:  vectorized  -> crash, as expected

'''
import numpy as cnp
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from time import time

n_sampl = 300  ; n_feature=2; n_qubits=3; layers=1; epochs=10

dev = qml.device('default.qubit', wires=n_qubits)  
#dev = qml.device('default.qubit', wires=n_qubits, shots=1000) 

#.... input data
X = cnp.random.uniform(-1, 1, size=(n_sampl, n_feature))
Y = cnp.where(X[:, 0] * X[:, 1] > 0, 1, -1) # Compute labels
#... trainable params
params = 0.2 * np.random.random(size=(layers, n_qubits,3))

@qml.qnode(dev)  
def circuit(params,x):
    a=cnp.arccos(x)  # encoding of the input to [0,2pi]
    qml.RY(a[0], wires=0)
    qml.RY(a[1], wires=1)
    for layer in range(layers): # EfficientSU2 ansatz
        qml.Barrier()
        for qubit in range(n_qubits):
            qml.RX(params[layer, qubit, 0], wires=qubit)
            qml.RY(params[layer, qubit, 1], wires=qubit)
            qml.RZ(params[layer, qubit, 2], wires=qubit)        
        for qubit in range(n_qubits):
            qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
    return qml.expval(qml.PauliZ(2))

print(qml.draw(circuit, decimals=2)(params,X[0]), '\n')

#... classical ML utility func
def cost_function( params, X, Y):  # vectorized code
    predL= [ circuit(params, x1) for x1 in X ]  # list comprehension, very slow
    #predL = circuit(params,  X.T)  # vectorized execution
    cost = np.mean((Y - qml.math.stack(predL)) ** 2)
    return cost

#... run optimizer
opt = NesterovMomentumOptimizer(0.10, momentum=0.90)
#opt=qml.SPSAOptimizer(maxiter=epochs)  # works w/ shots & vectorized circuit excution


for it in range(epochs):
    T0=time()
    params = opt.step(lambda p: cost_function(p, X, Y), params)
    durT=time()-T0
    print('epoch=%d,   %.2f sec/step'%(it,durT));    continue
    cost= cost_function(params, X, Y)  # doubles time
    print('epoch=%d, cost=%.3f   %.2f sec/step'%(it,cost,durT))
