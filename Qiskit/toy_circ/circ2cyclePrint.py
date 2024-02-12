#!/usr/bin/env python3
# https://qiskit.org/documentation/api/qiskit.dagcircuit.DAGCircuit.html

from qiskit import QuantumCircuit
from pprint import pprint
from qiskit.converters import circuit_to_dag, dag_to_circuit
import numpy as np

#...!...!..................
def print_dag_layers(qdag):
  dagLay=qdag.layers()
  print('Show %d layers as lists of gates, format: gateName qubits[] params[]; '%qdag.properties()['depth'])
  for k,lay in enumerate(dagLay):
    print('\n%d cycle: '%k,end='')
    for op in lay['graph'].op_nodes():
      qL=[qub._index for qub in  op.qargs]
      parV=[ float('%.6f'%v)  for v in op._op._params]      
      print('  ',op.name, 'q'+str(qL), parV,';', end='')

  print('\nend-of-cycles') 

#...!...!..................
def circA():
  qc = QuantumCircuit(3,3)
  qc.sx(0)
  qc.s(1)
  qc.h(0)
  qc.cx(0,1)
  qc.p(np.pi/3,qubit=2)
  qc.cx(1,2)
  qc.sx(0)
  qc.cx(0,1)
  qc.barrier([0, 1,2])
  for i in range(3): qc.measure(i,i)
  return qc


#...!...!..................
def circB1():
  qc = QuantumCircuit(3)
  qc.p(0.2,qubit=1)
  #qc.barrier([0, 1,2])
  for i in range(2):
    for iq in range(3):
      qc.sx(iq)
      qc.p(0.2,qubit=iq)
  qc.cx(0,1)
  for iq in range(3):
      qc.sx(iq)  
  return qc

#...!...!..................
def circC():
  qc = QuantumCircuit(2)
  qc.p(0.33,qubit=1)
  for iq in range(2): qc.sx(iq)  
  for i in range(3): qc.p(0.1*i,qubit=0)
  for iq in range(2): qc.sx(iq)
  qc.sx(1)
  qc.p(0.22,qubit=0)
  qc.cx(0,1)
  qc.p(0.44,qubit=1)
  for iq in range(2): qc.sx(iq)
  return qc

#=================================
#=================================
#  M A I N 
#=================================
#=================================

# Create single-qubit gate circuit
qc = circC()
print('INPUT qc=')
print(qc)

qdag = circuit_to_dag(qc)
#qdag.draw()  # pop-up persistent ImagViewer  (not from Matplotlib)

print('\nList DAG  properties:')
pprint(qdag.properties())

print('\nDecomposition of circuit DAG into cycles')
print_dag_layers(qdag)
