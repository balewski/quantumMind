#!/usr/bin/env python3
''' 
Bell state 
'''
from pprint import pprint
from time import time

from braket.aws import AwsDevice
device = AwsDevice("arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")
#device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

print('M: device',device)

from braket.circuits import Circuit

# Create an equivalent Braket circuit
qc_br = Circuit()
#qc_br.h(0).cnot(0, 1)  # Bell
qc_br.x(0).i(1)  #  Identity

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

shots=200

job = device.run(qc_br, shots=shots)
print('\nJOB:',job)

jobMD=job.metadata()

print('\META:',jobMD)

jobRes=job.result()


print('M: comp circ',jobRes.get_compiled_circuit())

print('M: counts',jobRes.measurement_counts)
  
'''  OUTPUT
M: comp circ OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[2];
sx q[4];
sx q[5];
ecr q[4], q[5];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[5];
sx q[4];
sx q[5];
measure q[5] -> c[0];
measure q[4] -> c[1];

M: counts Counter({'11': 89, '00': 74, '10': 23, '01': 14})
S/N=0.82  for 1 ECR-gate


= = = = = 
measure q[0] -> c[0];
measure q[1] -> c[1];
M: counts Counter({'00': 181, '01': 16, '10': 3})
S/N=0.90 

= = == = 
x q[0];
x q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
M: counts Counter({'11': 121, '10': 50, '01': 18, '00': 11})
S/N=0.60

= = = = = 
x q[0];
measure q[1] -> c[1];
measure q[0] -> c[0];
M: counts Counter({'10': 156, '11': 21, '00': 20, '01': 3})
S/N=0.6

'''
