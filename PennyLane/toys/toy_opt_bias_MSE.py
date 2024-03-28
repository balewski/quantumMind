#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
minimal demonstrator on how to fit threshold of EV to binary classify data
use batched training
Important: labels must be in convention  +1/-1  for square_loss to work
'''
import pennylane as qml
from pennylane import numpy as np
import numpy as cnp

n_samples = 500  ;  n_qubits=1;
xLo=-0.2
xThr=(1+xLo)/2.  # <=== GROUND TRUTH to be fitted

XD = cnp.random.uniform(xLo, 1, size=(n_samples,))
YD = cnp.where(XD > xThr, 1, 0)  # +1/0 labeling
ya=sum(YD==1); yb=sum(YD==0); print('BalanceB:%.2f'%(ya/(yb+yb)))
print("Shape of  X:", XD.shape,' xThr=%.2f'%xThr)
YD = YD * 2 - 1   # +1/-1 labeling

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit(params, x):
    a=np.arccos(x)  # encoding of the input to [0,2pi]
    qml.RY(a, wires=0)
    #qml.RY(params[0], wires=0)
    return qml.expval(qml.PauliZ(0)) 


#------
# test circuit assembly
params = 0.3 * np.random.randn(1, requires_grad=True)
x0 = [0.34]

print("Params:", params.shape)
print("X0: ", x0)

print(qml.draw(circuit, decimals=3)(params,x0))


def circ_to_label(params,  x):  # not used
    pred=circuit(params, x)
    lab_pred=np.where(pred > params[0], 1, -1)  # +1/-1 labeling
    return lab_pred


# . . . . . . . .
# add ML  computational elements

# bias is added classically
def circ_and_bias(params,  x):  # output is real number
    return circuit(params, x) - params[0]

def cost(params,  x, y):   # vecorized
    pred = circ_and_bias(params, x.T)
    mse_loss= np.mean((y -pred) ** 2)   
    return mse_loss

def accuracy(labels, pred):  # vectorized
    match= labels== pred
    acc=sum(match)/match.shape[0]
    return acc



#- - - - - - - - - - - - - - - - - - - 
#  TRAIN
opt = qml.NesterovMomentumOptimizer(0.5)
batch_size = 50
steps =70

for it in range(steps):

    # Update the params by one optimizer step, using only a limited batch of data
    idxL = np.random.randint(0, len(XD), (batch_size,))
    X_batch = XD[idxL]
    Y_batch = YD[idxL]
    params = opt.step(cost, params, x=X_batch, y=Y_batch)

    # Compute accuracy
    lab_pred = [np.sign(circ_and_bias(params,  x)) for x in XD]

    current_cost = cost(params,  XD, YD)
    acc = accuracy(YD, lab_pred)
    if it%1==0 : print("Iter: %4d | Cost: %.4f | Accuracy: %.4f" % (it + 1, current_cost, acc))
    if acc>0.98 : break

print("Params:", params.shape)
print("opt parms: %.3f  true Thr=%.3f  nSampl=%d"%( params[0],xThr,n_samples) )
