#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
use ansatz with uneven matrix of parameters 
'''

import numpy as cnp
import pennylane as qml
from pennylane import numpy as np

n_sampl = 300  ; n_feature=2; n_qubits=3; layers=1; epochs=60

dev = qml.device('default.qubit', wires=n_qubits)
#dev = qml.device('default.qubit', wires=n_qubits,shots=5000)

#.... input data
X = cnp.random.uniform(-1, 1, size=(n_sampl, n_feature))
#... Compute labels ....
if 1:  # cakeM4, trains very well
    Y = cnp.where(X[:, 0] * X[:, 1] > 0, 1, -1) 
else:  # circle, trains very poorly
    Y =  np.where( np.linalg.norm(X, axis=1) < 0.8, 1, -1)

#... trainable params
def init_params(layers,n_qubits):
    assert n_qubits==3
    out=[]
    fact=0.2
    for ly in range(layers):
        p1=fact*np.random.random(size=(1), requires_grad=True)
        p2=fact*np.random.random(size=(2), requires_grad=True)
        p3=fact*np.random.random(size=(3), requires_grad=True)
        out.append((p1,p2,p3))
    return tuple(out)
    
params=init_params(layers,n_qubits)
print('ini shape:',len(params),len(params[0]))
#print('ini dump0:',params[0])
#print('ini dump1:',params[1])


@qml.qnode(dev)  
def circuit(params,x):
    a=np.arccos(x)  # encoding of the input to [0,2pi]
    qml.RY(a[0], wires=0)
    qml.RY(a[1], wires=1)
    readQ=2  # 2 is good, 0 gives very poor accuracy
    for layer in range(layers): # EfficientSU2 ansatz
        qml.Barrier()
        parV=params[layer]
        assert len(parV)==n_qubits
        for qubit in range(n_qubits):
            parL=parV[qubit]
            qml.RX(parL[0], wires=qubit)
            if qubit<1: continue
            qml.RY(parL[1], wires=qubit)
            if qubit<2: continue
            qml.RZ(parL[2], wires=qubit)        
        for qubit in range(n_qubits-1):
            qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
    return qml.expval(qml.PauliZ(readQ))

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
opt = qml.NesterovMomentumOptimizer(0.03, momentum=0.95)
#opt=qml.SPSAOptimizer(maxiter=epochs)  # works w/ shots

for it in range(epochs):
     params = opt.step(lambda p: cost_function(p, X, Y), params) # optimizer line
     cost= cost_function(params, X, Y)
     acc= accuracy_metric(params,X, Y) 
     if it%10==0: print('epoch=%d, cost=%.3f  acc=%.3f'%(it,cost,acc))



