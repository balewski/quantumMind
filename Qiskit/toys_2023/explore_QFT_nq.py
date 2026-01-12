#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Implements  n-qubit QFT,
original code from
https://github.com/Qiskit/qiskit-terra/blob/master/examples/python/qft.py

Here you can get new token:
 https://quantumexperience.ng.bluemix.net/qx/account/advanced
'''

import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag


backendSim=Aer.get_backend("qasm_simulator")
import math

#............................
#............................
def input_state(circ, n):
    """n-qubit input state for QFT that produces output 1."""
    for j in range(n):
        circ.h(j)
        circ.u1(-math.pi/float(2**(j)), j)

#............................
#............................
def qft_op(circ, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi/float(2**(j-k)), j, k)
        circ.h(j)

#............................
#............................
def layers_2_gateAddress(qdag):
  print('\n Address uniquly each gate based on layer and qubits')
  dagLay=qdag.layers()
  for k,lay in enumerate(dagLay):
    gateL=[]
    for op in lay['graph'].op_nodes():
      adrL=[str(k),op.name] +[str(qr.index) for qr in  op.qargs]
      adr='.'.join(adrL)
      gateL.append(adr)
    print('%2d layer: '%k,', '.join(gateL))


#=================================
#=================================
#  M A I N 
#=================================
#=================================
nqubit=3
nclbit=nqubit
print('Exercise %d-qubit  QFT '%(nqubit))

qc = QuantumCircuit(nqubit, nclbit, name="qft%d"%nqubit)
input_state(qc, nqubit)
qc.barrier()
qft_op(qc,  nqubit)
qc.barrier()
# qc.x(nqubit-1) # added to make measurement palindromic
for j in range( nqubit):
    qc.measure(j, j)

print(qc)

# Compile and run the Quantum circuit on a simulator backend
job = execute(qc, backendSim,shots=1024)
countsD=job.result().get_counts(qc)
print('QFT  counts:',countsD)

circF='out/circQFT_%dQ.qasm'%nqubit
fd=open(circF,'w')
fd.write(qc.qasm())
fd.close()


optLev=1
print('\n Transpile=(Optimize and fuse gates) in qc1, optLev=',optLev)
qc2 = transpile(qc, basis_gates=['u1','u2','u3','cx'], optimization_level=optLev)
print(qc2)

qdag = circuit_to_dag(qc2)
layers_2_gateAddress(qdag)
print('circ saved to',circF)
