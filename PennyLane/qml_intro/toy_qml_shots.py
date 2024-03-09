#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
minimal demonstrator on how to optimze shot-based device QML  problem
Input data are non-lineray separable
Goal1: classify 2 categories
Goal2: use shot-based device instead of the state-vector
'''

import numpy as cnp
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

n_sampl = 300  ; n_feature=2; n_qubits=3; layers=1; epochs=60

dev = qml.device('default.qubit', wires=n_qubits)
#dev = qml.device('default.qubit', wires=n_qubits,shots=5000)

#.... input data
X = cnp.random.uniform(-1, 1, size=(n_sampl, n_feature))
Y = cnp.where(X[:, 0] * X[:, 1] > 0, 1, -1) # Compute labels
#... trainable params
params = 0.2 * np.random.random(size=(layers, n_qubits,3))

@qml.qnode(dev)  
def circuit(params,x):
    a=np.arccos(x)  # encoding of the input to [0,2pi]
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
    pred = circuit(params,  X.T)
    cost = np.mean((Y - qml.math.stack(pred)) ** 2)
    return cost

def accuracy_metric(params,X, Y):  # for binary classification
    pred = circuit(params,  X.T)
    pred_classes = [1 if p > 0 else -1 for p in pred]
    correct = np.mean(np.array(pred_classes) == Y)
    return correct

#... run optimizer
opt = NesterovMomentumOptimizer(0.03, momentum=0.95)
for it in range(epochs):
     params = opt.step(lambda p: cost_function(p, X, Y), params) # optimizer line
     cost= cost_function(params, X, Y)
     acc= accuracy_metric(params,X, Y) 
     if it%10==0: print('epoch=%d, cost=%.3f  acc=%.3f'%(it,cost,acc))



