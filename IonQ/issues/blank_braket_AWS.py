#!/usr/bin/env python3
''' problem: ??
See also:  https://github.com/qiskit-community/qiskit-braket-provider/blob/main/docs/tutorials/0_tutorial_qiskit-braket-provider_overview.ipynb

'''
from pprint import pprint
from time import time
from qiskit import QuantumCircuit

from braket.aws import AwsDevice
#device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1")
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

print('M: device',device)

from braket.circuits import Circuit

# Create an equivalent Braket circuit
qc_br = Circuit()

# Add the same gates: H on qubit 0, and CNOT with control qubit 0 and target qubit 1
qc_br.h(0).cnot(0, 1)
# Add measurement to both qubits
qc_br.probability(target=[0,1])
#https://docs.aws.amazon.com/braket/latest/developerguide/braket-result-types.html

print("Amazon Braket Circuit:")
print(qc_br)

print()
for instr in qc_br.instructions:
    print(instr)

# If you want to explicitly print which qubits are measured:
print('\nMeasured qubits:',qc_br._qubit_observable_mapping)

# = = = = = = = = = = = =
#  M A I N 
# = = = = = = = = = = = =

shots=10

job = device.run(qc_br, shots=shots)
print('\nJOB:',job)

jobMD=job.metadata()

print('\META:',jobMD)

jobRes=job.result()


print('M: comp circ',jobRes.get_compiled_circuit())

print('M: counts',jobRes.measurement_counts)
  
