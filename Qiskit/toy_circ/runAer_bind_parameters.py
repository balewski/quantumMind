#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

Execute Qasm-defined circuit on local simulator

required INPUT:
--circName: ghz_5qm

'''
#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# demonstrate  qc.bind_parameters(param_values)

import qiskit
print('Qiskit ver',qiskit.__version__)  # 1.0.1

from qiskit import  QuantumCircuit, transpile
from qiskit.circuit import Parameter

# Define the parameters
theta = Parameter('θ')
phi = Parameter('φ')

# Create a parameterized quantum circuit
qc = QuantumCircuit(1)
qc.rx(theta, 0)
qc.ry(phi, 0)

# Print the original parameterized circuit
print("Original parameterized circuit:")
print(qc)

# Define the parameter values
param_values = {theta: 1.57, phi: 3.14}  # Example values: π/2 and π

# Create a new circuit with parameters bound to the specified values
bound_qc = qc.bind_parameters(param_values)

# Print the bound circuit
print("\nCircuit after binding parameters:")
print(bound_qc)

# Use Aer's qasm_simulator
backend = AerSimulator()

# Since we need a measurement to simulate and get counts, add measurement to the bound circuit
bound_qc.measure_all()

# Transpile the circuit for the simulator
transpiled_qc = transpile(bound_qc, backend)

# Execute the circuit
job =   backend.run(transpiled_qc, shots=1024)

# Get the results
result = job.result()

# Get the counts (measurement outcomes) from the result
counts = result.get_counts(bound_qc)

# Grab the results from the job.
result = job.result()
counts = result.get_counts(0)
print(counts)
