#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

r"""

Based on https://pennylane.ai/qml/demos/tutorial_noisy_circuits/

Noisy circuits
 We'll also explore how to employ channel gradients to optimize noise parameters in a circuit.


"""

import pennylane as qml
from jax import numpy as np
import jax
import jaxopt

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

dev = qml.device('default.mixed', wires=2)

# converts [-inf, inf] --> [0,1]
def sigmoid(x):
    return 1/(1+np.exp(-x))

@qml.qnode(dev)
def circuit(x):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.AmplitudeDamping(sigmoid(x), wires=0)  # p = sigmoid(x) is reduction of Bloch sphere
    qml.AmplitudeDamping(sigmoid(x), wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

######################################################################
# We optimize the circuit with respect to a simple cost function that attains its minimum when
# the output of the QNode is equal to the experimental value:

def cost(x, target):
    return (circuit(x) - target)**2

######################################################################
# All that remains is to optimize the parameter. We use a straightforward gradient descent
# method.

target_ev = 0.8781  # observed expectation value from circuit
steps = 300
params = np.array(0.021)

print(qml.draw(circuit)(params)) 

opt = jaxopt.GradientDescent(cost, stepsize=0.4, acceleration = False)
opt_state = opt.init_state(params)
for i in range(steps):
    params, opt_state = opt.update(params, opt_state,target=target_ev)
    if (i + 1) % 30 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params,target=target_ev)))
    
print("Optimized param: %.2f"%params)
print(f"Optimized noise parameter p = {sigmoid(params.take(0)):.4f}")
print(f"QNode output after optimization = {circuit(params):.4f}")
print(f"Targte output value = {target_ev}")
print(qml.draw(circuit)(params)) 
print(' opt_state:',opt_state)
