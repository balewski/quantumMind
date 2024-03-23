#!/usr/bin/env python3
''' 
measure selected qubits

'''
from pprint import pprint
from time import time
from braket.aws import AwsDevice

device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
print('M: device',device)

from braket.circuits import Circuit

qc_br = Circuit()
qc_br.h(0).cnot(0, 1).x(2).x(2)
# Add measurement to both qubits
qc_br.probability(target=[2,1])

print("Amazon Braket Circuit:")
print(qc_br)

print()
for instr in qc_br.instructions:
    print(instr)

# If you want to explicitly print which qubits are measured:
print('\nMeasured qubits:',qc_br._qubit_observable_mapping)


job = device.run(qc_br, shots=10)

jobRes=job.result()


print('M: comp circ',jobRes.get_compiled_circuit())

print('M: counts',jobRes.measurement_counts)
  
