#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Variational Classifier: 2 features --> binary class
The case of non-linearly separable 
Input 2D input data which are non-linearly separable

Based on 
INPUT: 2 real-valued vectors Based on https://pennylane.ai/qml/demos/tutorial_variational_classifier/

"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

n_qubits=3

#...... pick one device
#dev = qml.device('default.qubit', wires=n_qubits)  # works
dev = qml.device('default.qubit', wires=n_qubits, shots=5000)  # FixMe

######################################################################
print('input 2D circle binary data and pre-process') 
XY = np.load("../notebooks/data/circ2d_bin.npy")
# Separate the data and labels
X = XY[:,:-1]
Y = XY[:,-1]
print('X sh:', X.shape, Y.shape)

print(f"First X sample (original)  : {X[0]}, shape:",X.shape)

print('Labels sample',Y[::10])
print(' Warning: data were not shuffled yet')

# Split data on train/test subsets
np.random.seed(0)
num_data = Y.shape[0]
num_train = int(0.7 * num_data)
index = np.random.permutation(range(num_data))
X_train = X[index[:num_train]]
Y_train = Y[index[:num_train]]
X_val = X[index[num_train:]]
Y_val = Y[index[num_train:]]

# We need these later for plotting
X_train = X[index[:num_train]]
X_val = X[index[num_train:]]

print('train labels sample',Y_train[:10])
print('train X sample',X_train[:3])
print('val labels sample',Y_val)


    
######################################################################
#  Encoding

def state_preparation(x):
    a=np.arccos(x)
    qml.RY(a[0], wires=0)
    qml.RY(a[1], wires=1)

@qml.qnode(dev)
def test(angles):
    state_preparation(angles)
    return qml.expval(qml.PauliZ(0))

print(qml.draw(test, decimals=2)(X[0]), '\n')

######################################################################
# Define the EfficientSU2 ansatz

def efficient_su2_ansatz(params):
    """
    EfficientSU2 ansatz for 3 qubits.
    Parameters should have a shape of (layers, n_qubits, 3),
    where layers is the number of ansatz layers,
    n_qubits is the number of qubits (3 in this case),
    and 3 represents the rotation angles for RX, RY, and RZ gates.
    """
    # Number of layers
    layers = params.shape[0]

    for layer in range(layers): # Apply rotational gates to each qubit
        qml.Barrier()
        for qubit in range(n_qubits):
            qml.RX(params[layer, qubit, 0], wires=qubit)
            qml.RY(params[layer, qubit, 1], wires=qubit)
            qml.RZ(params[layer, qubit, 2], wires=qubit)
        
        # Apply CNOTs for entanglement, forming a ring of qubits
        for qubit in range(n_qubits):
            qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])


# Define a QNode that uses the ansatz
@qml.qnode(dev)
def circuit(params,x):
    state_preparation(x)  
    efficient_su2_ansatz(params)
    return qml.expval(qml.PauliZ(2))

#........ test full circuit 
# Example usage
layers = 2
num_u3_ang= 3  # const, relates to U3 parametrization
weights_init = 0.2 * np.random.random(size=(layers, n_qubits, num_u3_ang))

print('QML dims  qubits=%d  EfficientSU2 ansatz, layers=%d'%(n_qubits,layers))
print('weights sh:',weights_init.shape,'\n full circ:')

print(qml.draw(circuit, decimals=2)(weights_init,X[0]), '\n')

######################################################################
# remaining classical functionality required for ML training
#  loss-function
def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

def cost(weights,  X, Y):
    global function_calls
    function_calls += X.shape[0]
    # Transpose the batch of input data in order to match the expected dims
    pred = circuit(weights,  X.T) # circuit is vectorized
    return square_loss(Y, pred)
    
#  ACCURACY, for monitoring only
def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc




######################################################################
# Optimization  https://openqaoa.entropicalabs.com/optimizers/pennylane-optimizers/
# needs gradient 
opt = NesterovMomentumOptimizer(0.2)

print('\n train the variational classifier...')
shots=5000
steps = 80
batch_size = 10
weights = weights_init

if 1:  # initial accuracy
    pred_val = np.sign(circuit(weights,  X_val.T))
    acc_val = accuracy(Y_val, pred_val)

function_calls = 0
for it in range(steps):
    # Update the weights by one optimizer step, use just one batch-size of data selected at random, so 1 step is NOT 1 epoch
    batch_index = np.random.randint(0, num_train, (batch_size,))
    X_batch = X_train[batch_index]
    Y_batch = Y_train[batch_index]
    
    weights = opt.step(lambda p: cost(p, X_batch, Y_batch), weights)
    # Compute predictions on train and validation set
    pred_val = np.sign(circuit(weights,  X_val.T))
    acc_val = accuracy(Y_val, pred_val)
    pred_train = np.sign(circuit(weights,  X_train.T))

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, pred_train)

    if (it + 1) % 10 == 0 or it<10:
        _cost = cost(weights,  X_train,Y_train)
        print(
            f"Iter: {it + 1:5d} | train Cost: {_cost:0.4f} | "
            f"Acc train: {acc_train:0.4f} | Acc validation: {acc_val:0.4f}"
        )
        
######################################################################
print('\n INFER on val-data')
#  define a function to make a predictions over multiple data points.
pred_val = np.sign(circuit(weights,  X_val.T))
acc_val = accuracy(Y_val, pred_val)
print('\npred_val:%s  acc_val=%.3f'%(pred_val,acc_val))
print('targets:',Y_val,type(Y_val))
res=pred_val - Y_val
print('L2:',res**2)
loss = square_loss(Y_val, pred_val)
print('loss:',loss,'ndata:',pred_val.shape)

print('end-weight:',weights)
print("Total number of cost function calls:", function_calls)

print('end')
