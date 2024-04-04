#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Based on : https://pennylane.ai/qml/demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax/

compares speed for  ??
dev: state based, circuit execution:  vectorized  --> 0.05 sec/step
dev: state based, circuit execution:  list compreh.  --> 3.4 sec/step  
dev: shot-based, circuit execution:  list compreh. --> 14 sec/step 
dev: shot-based, circuit execution:  vectorized  -> crash, as expected

'''
import numpy as cnp
import pennylane as qml
import jax
from jax import numpy as jnp
import optax
from time import time

n_sampl = 300
n_feature=2; n_qubits=3; layers=1; steps=10

#dev = qml.device('default.qubit', wires=n_qubits)  # works
dev = qml.device('default.qubit', wires=n_qubits, shots=1000)  # ValueError: probabilities do not sum to 1

#.... input data
Xu= cnp.random.uniform(-1, 1, size=(n_sampl, n_feature) )
Xa=cnp.arccos(Xu)
X = jnp.array(Xa )
Y = jnp.where(Xu[:, 0] * Xu[:, 1] > 0, 1, -1) # Compute labels
#... trainable params
params = 0.2 * jnp.array( cnp.random.random(size=(layers, n_qubits,3)) )

@qml.qnode(dev)  
def circuit(params,x):
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
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
@jax.jit
def loss_fn( params, X, Y):  # vectorized code
    predL = circuit(params,  X.T)  # vectorized execution
    #cost = jnp.mean((Y - qml.math.stack(predL)) ** 2)
    cost = jnp.mean((Y - predL) ** 2)
    return cost

print('M: verify code sanity, X:',X.shape)
T0=time()
val=loss_fn(params, X,Y)
durT=time()-T0
print('elaT=%.1f sec, one loss_fn:%s'%(durT,val))

print('grad:',jax.grad(loss_fn)(params,X,Y))

#... run optimizer
opt = optax.adam(learning_rate=0.3)
opt_state = opt.init(params)

def update_step(opt, params, opt_state, data, targets):
    loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

for i in range(100):
    params, opt_state, loss_val = update_step(opt, params, opt_state, X,Y)
 
    if i % 5 == 0:
        print(f"Step: {i} Loss: {loss_val}")

