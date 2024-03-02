#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Variational Classifier: iris 4 features --> binary class

Based on 
INPUT: 2 real-valued vectors Based on https://pennylane.ai/qml/demos/tutorial_variational_classifier/

"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

n_wire=2
dev = qml.device("default.qubit", wires=n_wire)  # state-vector

# dev = qml.device("default.qubit", wires=n_wire,  shots=10000)


######################################################################
''' Encoding
we will work with data from the positive subspace, so that we can ignore signs

The circuit is coded according to the scheme in Möttönen, et al. (2004), or—as presented for positive vectors only—in Schuld and Petruccione (2018). We also decomposed controlled Y-axis rotations into more basic gates, following Nielsen and Chuang (2010). UCR encoding
'''

def get_angles(x): # maps 4 real>0 --> 5 Ry angles 
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

def state_preparation(a):
    qml.RY(a[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)


#........ test encoding circuit 
x = np.array([0.53896774, 0.79503606, 0.27826503, 0.4], requires_grad=False)
ang = get_angles(x)

@qml.qnode(dev)
def test(angles):
    state_preparation(angles)
    return qml.state()

print('\nM: test encoding via UCR')
state = test(ang)
print("x               : ", np.round(x, 4))
print("angles          : ", np.round(ang, 4))
print("amplitude vector: ", np.round(np.real(state), 4))

print(qml.draw(test, decimals=2)(ang), '\n')

######################################################################
'''  Define QML layers, as   2-qubit variational layer

define the variational quantum circuit as this state preparation routine, followed by a repetition of the layer structure.
'''
def layer(layer_weights):
    for wire in range(2):
        qml.Rot(*layer_weights[wire], wires=wire)
    qml.CNOT(wires=[0, 1])

@qml.qnode(dev)
def circuit(weights, x):  # decision made by meas Q0
    state_preparation(x)
    for layer_weights in weights:
        layer(layer_weights)

    return qml.expval(qml.PauliZ(0))


#........ test full circuit 

num_qubits = 2
num_layers = 2
num_u3_ang= 3  # const, relates to U3 parametrization

weights_init = 0.01 * np.random.randn(num_layers, num_qubits, num_u3_ang, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)
print('QML dims  qubits=%d  ansatz layers=%d'%(num_qubits,num_layers))
print('weights sh:',weights_init.shape,' bias sh:',bias_init.shape,'\n full circ:')

print(qml.draw(circuit, decimals=2)(weights_init,ang), '\n')


######################################################################
''' remaining classical functionality required for ML training 

'''

# bias is added classically
def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

#  loss-function
def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

def cost(weights, bias, X, Y):
    # Transpose the batch of input data in order to make the indexing
    # in state_preparation work
    predictions = variational_classifier(weights, bias, X.T)
    return square_loss(Y, predictions)
    
#  ACCURACY, for monitoring only
def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

######################################################################
#- - - - - - - -
print('input Iris data and pre-process') 


data = np.loadtxt("../notebooks/data/iris_classes1and2_scaled.txt")
X = data[:, 0:2]
print(f"First X sample (original)  : {X[0]}, shape:",X.shape)

# pad the vectors to size 2^2=4 with constant values
padding = np.ones((len(X), 2)) * 0.1
X_pad = np.c_[X, padding]
print(f"First X sample (padded)    : {X_pad[0]}")

# normalize each input
normalization = np.sqrt(np.sum(X_pad**2, -1))
X_norm = (X_pad.T / normalization).T
print(f"First X sample (normalized): {X_norm[0]}, shape:",X_norm.shape)

# the angles for state preparation are the features
features = np.array([get_angles(x) for x in X_norm], requires_grad=False)
print(f"First features sample      : {features[0]}  <-- input to UCR encoding")

Y = data[:, -1]  # labels
print('Labels sample',Y[::10])
print(' Warning: data were not shuffled')

# Split data on train/test subsets
np.random.seed(0)
num_data = len(Y)
num_train = int(0.80 * num_data)
index = np.random.permutation(range(num_data))
feats_train = features[index[:num_train]]
Y_train = Y[index[:num_train]]
feats_val = features[index[num_train:]]
Y_val = Y[index[num_train:]]

# We need these later for plotting
X_train = X[index[:num_train]]
X_val = X[index[num_train:]]

print('train labels sample',Y_train[:10])
print('train feature sample',feats_train[:3])

######################################################################
''' Optimization
we minimize the cost, using the imported optimizer.

'''

opt = NesterovMomentumOptimizer(0.01)
batch_size = 8
steps = 70

print('\n train the variational classifier...')
weights = weights_init
bias = bias_init
for it in range(steps):
    # Update the weights by one optimizer step, use just one batch-size of data selected at random, so 1 step is NOT 1 epoch
    batch_index = np.random.randint(0, num_train, (batch_size,))
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    weights, bias, _, _ = opt.step(cost, weights, bias, feats_train_batch, Y_train_batch)

    # Compute predictions on train and validation set
    predictions_train = np.sign(variational_classifier(weights, bias, feats_train.T))
    predictions_val = np.sign(variational_classifier(weights, bias, feats_val.T))

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    if (it + 1) % 5 == 0:
        _cost = cost(weights, bias, features, Y)
        print(
            f"Iter: {it + 1:5d} | Cost: {_cost:0.7f} | "
            f"Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}"
        )

    if it==0:
        print('feats_train_batch sh:',feats_train_batch.shape)
        print('predictions_train sh:',predictions_train.shape,' predictions_val sh:',predictions_val.shape,' acc_train sh:', acc_train.shape)

        
######################################################################
print('\n INFER on val-data')
#  define a function to make a predictions over multiple data points.

preds_val = np.sign(variational_classifier(weights, bias, feats_val.T))

print('\npred_val:',preds_val,type(preds_val))
print('targets:',Y_val,type(Y_val))
res=preds_val - Y_val
print('L2:',res**2)
loss = square_loss(Y_val, preds_val)
print('loss:',loss,'ndata:',preds_val.shape)
