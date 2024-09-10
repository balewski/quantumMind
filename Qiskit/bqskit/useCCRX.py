#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
ErmalClemens:
 I found the gate representation for this unitary: It's essentially a double controlled rotation by RX with angle phi=Arctan[sqrt( gamma dt/ (1-gamma dt) ) ].

Problem1:  extra factor of 2?
if gamma dt=0.5 --> phi=ArcTan(1) =pi/4

Problem2:  order of indices ???
'''

import numpy as np

from qiskit import QuantumCircuit,QuantumRegister, transpile
from qiskit_aer import  Aer, AerSimulator

from qiskit.circuit.library.standard_gates import RXGate
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt

theta = Parameter('θ')

qr=QuantumRegister(3)
circ=QuantumCircuit(qr)
CCRX=RXGate(theta).control(2)
circ.append(CCRX,qr)

print(circ)
simulator = Aer.get_backend('aer_simulator')

#circ = transpile(circ, simulator); print(circ)

circ = transpile(circ, simulator, basis_gates=['u','cx','ry'])
print(circ)

thVal=np.pi/2

qc=circ.assign_parameters({theta: thVal})

qc.save_unitary()

# Run and get unitary
result = simulator.run(qc).result()
unitary = result.get_unitary(qc)
print("Circuit thVal=%.3f unitary:\n"%thVal, np.asarray(unitary).round(3))
