#!/usr/bin/env python3
''' problem: ??
See also:  https://github.com/qiskit-community/qiskit-braket-provider/blob/main/docs/tutorials/0_tutorial_qiskit-braket-provider_overview.ipynb

'''
from pprint import pprint
from time import time
from qiskit import QuantumCircuit

from braket.aws import AwsDevice
device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1")

from braket.circuits import Circuit

# Create an equivalent Braket circuit
qc_braket = Circuit()

# Add the same gates: H on qubit 0, and CNOT with control qubit 0 and target qubit 1
qc_braket.h(0).cnot(0, 1)

# Display the Braket circuit
print("Amazon Braket Circuit:")
print(qc_braket)

# = = = = = = = = = = = =
#  M A I N 
# = = = = = = = = = = = =

shots=10

#provider = IonQProvider()   # Remember to set env IONQ_API_KEY='....'
#print(provider.backends())  # Show all backends
#backend = provider.get_backend("ionq_simulator")
job = device.run(qc_braket, shots=shots)
print(job)

jobMD=job.metadata()

print(jobMD)
