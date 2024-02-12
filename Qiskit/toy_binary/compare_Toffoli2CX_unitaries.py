#!/usr/bin/env python3
__author__ = "Jan Balewski + ChatGPT"
__email__ = "janstar1122@gmail.com"

from qiskit import Aer, QuantumCircuit, transpile
import numpy as np
from pprint import pprint
from Util_JaccardIndex import circ_depth_aziz

#...!...!....................
def ccx_proxy(self,i,j,k):
    self.x(i)
    self.x(j)
    
    for m in range(2):
      self.ry(np.pi/4,k)
      self.cx(j,k)

      self.ry(np.pi/4,k)
      self.cx(i,k)

    self.x(i)
    self.x(j)
    self.z(k)  # fix common phase
    

# Define your two quantum circuits
qc1 = QuantumCircuit(3)
qc1.ccx(0,1,2)
print(qc1)

qc2 = QuantumCircuit(3)
ccx_proxy(qc2,0,1,2)
print(qc2)

backend=Aer.get_backend("qasm_simulator")
basis_gates = ['p','u3','cx']
qc3 = transpile(qc1, backend=backend,basis_gates =basis_gates,  optimization_level=3)
print(qc3)

# Transpile the circuits
simulator = Aer.get_backend('unitary_simulator')

qc1_us = transpile(qc1, simulator)
qc2_us= transpile(qc2, simulator)
qc3_us= transpile(qc3, simulator)

# Simulate the circuits to get their unitary representations
result1 = simulator.run(qc1_us).result()
U1 = result1.get_unitary()

result2 = simulator.run(qc2_us).result()
U2 = result2.get_unitary()

result3 = simulator.run(qc3_us).result()
U3 = result3.get_unitary()

# Compare the unitary matrices
equal12 = np.allclose(U1,U2, atol=1e-10)
equal13 = np.allclose(U1,U3, atol=1e-10)
equal23 = np.allclose(U2,U3, atol=1e-10)

# Format and print the unitary matrices
def niceU(U):
    return np.array2string(np.where(np.round(U, 3) == 0.+0.j, None, np.round(U, 3)), separator=', ')

# Format and print the unitary matrices
fU1 = niceU(U1)
fU2 = niceU(U2)
fU3 = niceU(U3)


print("Unitary for circuit 1:\n",fU1)

if equal12:
    print("\nPASS_12:  the same unitary ")
else:
    print("\nFAIL: different unitary 2")
    #print("Unitary for circuit 2:\n",fU2)
    fD=niceU(U2-U1)
    print("diff U2-U1:\n",fD)

if equal13:
    print("\nPASS_13:  the same unitary")
else:
    print("\nFAIL: different unitary 3")
    #print("Unitary for circuit 3:\n",fU3)
    fD=niceU(U3-U1)
    print("diff U3-U1:\n",fD)

if equal23:
    print("\nPASS_23:  the same unitary")

print('M: compare depths')
circ_depth_aziz(qc1,'Toffoli')
circ_depth_aziz(qc2,'CCX_proxy')
circ_depth_aziz(qc3,'transp Toffoli')
