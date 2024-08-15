#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Imports parametrized circuit from Qikist and runs it on fake IBM backend using Qiskit Aer inside Penny lane

'''

# based on https://discuss.pennylane.ai/t/how-to-train-a-circuit-imported-from-qiskit/1832
import os
import pennylane as qml
from pennylane import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_aer.noise import NoiseModel

useFake=1

if useFake :
    from qiskit_ibm_runtime import QiskitRuntimeService
    ibm_token = os.getenv('QISKIT_IBM_TOKEN')
    service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)

    backend = service.backend('ibm_kyoto')
    print('use Qiskit fake backend %s version:%d'%(backend,backend.version) )
    noise_model = NoiseModel.from_backend(backend)

qc = QuantumCircuit(2)
theta = Parameter('θ')

state = [1/2,1/2,1/2,1/2]

qc.initialize(state, qc.qubits)
qc.rx(theta, [0])
qc.cx(0, 1)

my_circuit = qml.from_qiskit(qc)


if useFake:
    dev = qml.device(
        "qiskit.aer",
        wires=2,
        shots=1000,
        noise_model=noise_model,
        backend="aer_simulator_statevector",
        seed_simulator=1,
        seed_transpiler=1,
    )
else:
    print('use PennyLane device')
    dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def circuit(x):
    my_circuit(params={theta: x},wires=(1, 0))
    return qml.expval(qml.PauliZ(0))

theta_train = np.array(0.4, requires_grad=True)

print('\nM: init circ;')
print(qml.draw(circuit, decimals=3)(theta_train), '\n')

opt = qml.GradientDescentOptimizer()

for i in range(20):
    theta_train = opt.step(circuit, theta_train)
    if i % 5 == 0:
        print('Cost',i, circuit(theta_train))

print('M: final circ theta=%.2f'%theta_train)
print(qml.draw(circuit, decimals=3)(theta_train), '\n')
