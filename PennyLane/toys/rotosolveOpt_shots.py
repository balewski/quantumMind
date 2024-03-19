#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
Use case of RotosolveOptimizer working w/ shots
parameter array is 'uneven'
I do not understand wht this circuit does, but optimizer is happy
based on https://docs.pennylane.ai/en/stable/code/api/pennylane.RotosolveOptimizer.html 
'''


import pennylane as qml
from pennylane import numpy as np

n_sampl = 300  ; n_feature=2; n_qubits=3; layers=1; epochs=60

#dev = qml.device('default.qubit', wires=n_qubits)
dev = qml.device('default.qubit', wires=n_qubits,shots=5000)


# measuring the expectation value of the tensor product of PauliZ operators on all qubits.
@qml.qnode(dev)
def circuit(rot_param, layer_par, crot_param, rot_weights=None, crot_weights=None):
    #print('ff',rot_param, layer_par, crot_param)
    for i, par in enumerate(rot_param * rot_weights):
        qml.RX(par, wires=i)
    for w in dev.wires:
        qml.RX(layer_par, wires=w)
    for i, par in enumerate(crot_param*crot_weights):
        qml.CRY(par, wires=[i, (i+1)%3])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))


''' It takes three parameters:
* rot_param controls three Pauli rotations with three parameters, multiplied with rot_weights,
* layer_par feeds into a layer of rotations with a single parameter, and
* crot_param feeds three parameters, multiplied with crot_weights, into three controlled Pauli rotations.
'''

init_param = (
    np.array([0.3, 0.2, 0.67], requires_grad=True),
    np.array(1.1, requires_grad=True),
    np.array([-0.2, 0.1, -2.5], requires_grad=True),
)
rot_weights = np.ones(3)  # not optimized, because no 'requires_grad=True'
crot_weights = np.ones(3)  # not optimized

# needed by Rotosolve:
nums_frequency = {
    "rot_param": {(0,): 1, (1,): 1, (2,): 1},
    "layer_par": {(): 3},
    "crot_param": {(0,): 2, (1,): 2, (2,): 2},
}

rot_weights = np.array([0.4, 0.8, 1.2], requires_grad=False)
crot_weights = np.array([0.5, 1.0, 1.5], requires_grad=False)

print('pp1:',init_param)
print('pp2:',*init_param)

print(qml.draw(circuit, decimals=2)(*init_param, rot_weights=rot_weights, crot_weights=crot_weights), '\n')

# Execute the circuit and get the measurement results
results = circuit(*init_param,  rot_weights=rot_weights, crot_weights=crot_weights)
print('results:',results)

#... classical ML utility func
def cost_function( rot_param, layer_par, crot_param, **args):  # vectorized code
    pred = circuit(rot_param, layer_par, crot_param,  **args)
    return pred

opt_kwargs = {"num_steps": 4}
opt = qml.optimize.RotosolveOptimizer(substep_optimizer="brute", substep_kwargs=opt_kwargs)
num_steps = 10

param = init_param
cost_rotosolve = []
for step in range(num_steps):
    param, cost, sub_cost = opt.step_and_cost(
        cost_function,
        *param,
        nums_frequency=nums_frequency,
        full_output=True,
        rot_weights=rot_weights,
        crot_weights=crot_weights,
    )
    print(f"Cost before step: {cost}")
    print(f"Minimization substeps: {np.round(sub_cost, 6)}")
    cost_rotosolve.extend(sub_cost)




