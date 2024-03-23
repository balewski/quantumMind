#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
minimal demonstrator   multipl resets crash SPSAOptimizer() in ver 0.35

'''


import pennylane as qml
from pennylane import numpy as np
num_qubit=1 ; shots=1000
dev = qml.device('default.qubit', wires=num_qubit,shots=shots)
from time import time

nReset=2

@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    for j in range(nReset):
        m = qml.measure(0,reset=True)
        qml.cond(m, qml.PauliX)(0)
    return qml.expval(qml.PauliZ(0) )

# Initialize parameters
params = np.array([0.011, 0.012], requires_grad=True)

print(qml.draw(circuit, decimals=2)(params), '\n')
T0=time()
print(' run circ with %d resets ...'%nReset)
y = circuit(params)
elaT=time()-T0
print('input X=%s  Y=%.2f   shots=%d  elaT=%.1f sec  nReset=%d'%(params,y,shots,elaT, nReset))

#....  training  code
def cost(params, target_value):
    global function_calls
    return (circuit(params) - target_value) ** 2


num_steps=3
optimizer=qml.SPSAOptimizer(maxiter=num_steps)


# Set the target expectation value
target_value = -0.3
T0=time()

print('\n  Optimization loop ...')
for i in range(num_steps):

    params = optimizer.step(lambda p: cost(p, target_value), params)
    prev_cost = cost(params, target_value)
    elaT=time()-T0    
    print("Step %d: Cost = %.4f  elaT=%.1f sec" % (i+1, float(prev_cost),elaT))

# Output the optimized parameters and the final cost
print("Optimized rotation angles: [%.3f, %.3f]" % (params[0], params[1]))
print("Final cost : %.3f" % float(cost(params, target_value)))
print("Final circ output : %.3f  target: %.3f" % (float(circuit(params)),target_value))

print('qml setup:',dev,optimizer)
