#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# demonstrate  qc.bind_parameters(param_values)

import qiskit
print('Qiskit ver',qiskit.__version__)  # 1.0.1
from qiskit_aer import AerSimulator
from qiskit import  QuantumCircuit, transpile
from qiskit.circuit import Parameter

# Define the parameters
theta = Parameter('θ')
phi = Parameter('φ')

# Create a parameterized quantum circuit
nq=5
qc = QuantumCircuit(nq)
qc.rx(theta, 0)
qc.ry(phi, 0)
for iq in range(1,nq): qc.cx(0,iq)

# Print the original parameterized circuit
print("Original parameterized circuit:")
print(qc)

# Define the parameter values
param_values = {theta: 1.57, phi: 3.14}  # Example values: π/2 and π

# Create a new circuit with parameters bound to the specified values
bound_qc = qc.assign_parameters(param_values)

# Print the bound circuit
print("\nCircuit after assigning parameters:")
print(bound_qc)

# Use Aer's qasm_simulator
backend = AerSimulator()

if 0:
    from qiskit.providers.fake_provider  import FakeHanoi
    fake_backend = FakeHanoi()
    backend = AerSimulator.from_backend(fake_backend)

# Since we need a measurement to simulate and get counts, add measurement to the bound circuit
bound_qc.measure_all()

# Transpile the circuit for the simulator
qcT = transpile(bound_qc, backend)
qcT=bound_qc
print(qcT.draw(output='text',idle_wires=False))
#print('tt',type(qcT._layout))
#print('ll',qcT._layout.final_index_layout(filter_ancillas=True))

# Execute the circuit
job =   backend.run(qcT, shots=1024)

# Get the results
result = job.result()

# Get the counts (measurement outcomes) from the result
counts = result.get_counts(bound_qc)

# Grab the results from the job.
result = job.result()
counts = result.get_counts(0)
print(counts)
