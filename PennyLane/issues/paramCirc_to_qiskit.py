#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
minimal demonstrator on how to export parametrized PennyLane circuit to Qiskit

'''


from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import pennylane as qml
from pennylane import numpy as np
from pennylane_qiskit import load

def qk_circ():
    params = [Parameter(f'φ{i}') for i in range(2)]
    qc = QuantumCircuit(2)
    qc.rx(params[0], 0) 
    qc.rz(params[1], 0)
    qc.cx(0,1)
    qc.measure_all()
    return qc

qk_qc0=qk_circ()
print(type(qk_qc0))
print('true qk qc'); print(qk_qc0.draw())


# Define the PennyLane circuit
dev = qml.device('default.qubit', wires=2, shots=10)

# Define the QNode
@qml.qnode(dev)
def circuit(params):
    # Apply RX and RZ rotations to qubit 0 based on the given parameters
    qml.RX(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    qml.CNOT(wires=[0, 1])
    return [qml.sample(qml.PauliZ(i)) for i in range(2)]

params = np.array([0.7, 0.2])


print(qml.draw(circuit, decimals=2)(params), '\n')

qk_qc=load(circuit)
print(type(qk_qc))

# Bind parameter values and execute the circuit
pars4qiskit = {params[0]: 1.0, params[1]: 2.0}  
print(qk_qc(pars4qiskit))

