#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
The program is structured to optimize a quantum circuit with the goal of minimizing the expectation value of a Hamiltonian, represented by <Z>, for a single qubit. The cost function will be designed to measure the squared difference between the circuit's output and a target expectation value. 
'''

import pennylane as qml
from pennylane import numpy as np

n_wires=1
shots=1000
# pick one device
if 0:     # Initialize the FakeHanoi simulator
    from qiskit.providers.aer import AerSimulator
    from qiskit.providers.fake_provider  import FakeHanoi
    fake_hanoi_backend = FakeHanoi()
    aer_simulator = AerSimulator.from_backend(fake_hanoi_backend)
    dev = qml.device('qiskit.aer', wires=n_wires, backend=aer_simulator, shots=shots)
if 0:  # Set up the state vector sim  device
    dev = qml.device('default.qubit', wires=n_wires)
if 1: # Set up a shot-based sim device.
    dev = qml.device('default.qubit', wires=n_wires, shots=shots)

# Define the quantum circuit (ansatz)
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# Define the objective function (the cost function) with a target expectation value

# Function call counter
function_calls = 0

def cost(params, target_value):
    global function_calls
    function_calls += 1
    return (circuit(params) - target_value) ** 2

# Initialize parameters
params = np.array([0.011, 0.012], requires_grad=True)

                  
# pick optimizer, both work with shot-based simu
#optimizer = qml.GradientDescentOptimizer(stepsize=0.4)
#optimizer = qml.NesterovMomentumOptimizer(0.4)
#optimizer = qml.RMSPropOptimizer(stepsize=0.1, decay=0.9)
optimizer=qml.SPSAOptimizer(maxiter=50)  
# Set the target expectation value
target_value = -0.3

# Optimization loop
num_steps = 50
for i in range(num_steps):
  
    params = optimizer.step(lambda p: cost(p, target_value), params)
    prev_cost = cost(params, target_value)
    
    if i%5==0:
        print("Step %d: Cost = %.4f" % (i+1, float(prev_cost)))

# Output the optimized parameters and the final cost
print("Optimized rotation angles: [%.3f, %.3f]" % (params[0], params[1]))
print("Final cost : %.3f" % float(cost(params, target_value)))
print("Final circ output : %.3f  target: %.3f" % (float(circuit(params)),target_value))
print("Total number of cost function calls:", function_calls)
print('qml setup:',dev,optimizer)

