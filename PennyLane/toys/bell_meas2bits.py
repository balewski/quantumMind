#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


import pennylane as qml

def create_pennylane_circuit(params):
    @qml.qnode(qml.device('default.qubit', wires=2, shots=10))  # 2 qubits, 10 shots
    def circuit():
        # Apply RX rotations to both qubits based on the given parameters
        qml.RX(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        # Apply CNOT gate for entanglement
        qml.CNOT(wires=[0, 1])
        # Measure both qubits
        return [qml.sample(qml.PauliZ(i)) for i in range(2)]

    return circuit

# Example usage with 4 parameters for the rotations
params = [0.7, 0.2]

circuit = create_pennylane_circuit(params)
print(qml.draw(circuit, decimals=2)(), '\n')

# Execute the circuit and get the measurement results
measurement_results = circuit()

# Post-process the measurement results to pair up the measurements for each shot
paired_measurements = list(zip(measurement_results[0], measurement_results[1]))

print("List of pairs of 2-bit measurements:", paired_measurements)
