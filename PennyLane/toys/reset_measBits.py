#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# check reset

import pennylane as qml

@qml.qnode(qml.device('default.qubit', wires=2, shots=10))  # 2 qubits, 10 shots
def circuit():
    qml.Hadamard(wires=0)
    m0 = qml.measure(0) #, reset=True)
    qml.cond(m0, qml.PauliX)(1)
    # Measure both qubits
    return [qml.sample(qml.PauliZ(i)) for i in range(2)]

print(qml.draw(circuit, decimals=2)(), '\n')

# Execute the circuit and get the measurement results
measurement_results = circuit()
# Post-process the measurement results to pair up the measurements for each shot
paired_measurements = list(zip(measurement_results[0].numpy(), measurement_results[1].numpy()))

print(" 2-bit measurements:", paired_measurements)
