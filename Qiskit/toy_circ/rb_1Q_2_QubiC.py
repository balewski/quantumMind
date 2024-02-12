#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Randomized Benchmarking
based on 
https://community.qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html

Can run on a simulator and exports circuits for QubiC 
as list of list of gates decribed as follows:
- {'name': 'u3', 'param': [1.1, 2.2, 3.3], 'qubit': [0]}
- {'name': 'cx', 'param': [], 'qubit': [1, 2]}


'''

import time,os,sys
from pprint import pprint
import ruamel.yaml as yaml

import qiskit.ignis.verification.randomized_benchmarking as rb
from qiskit import transpile
from qiskit.converters import circuit_to_dag

import qiskit
qiskitVer=qiskit.__qiskit_version__
print('qiskit ver=',qiskitVer)
assert qiskitVer['qiskit'] >= '0.23.2'


def layers_2_dict(qdag):
  print('\n Order gates chronologically  (per layer) as list of lists')
  outLL=[]
  dagLay=qdag.layers()
  for k,lay in enumerate(dagLay):
    gateL=[]
    for op in lay['graph'].op_nodes():
      one={'name':op.name}
      one['qubit']= [qr._index  for qr in  op.qargs]
      one['param']=op._op._params
      gateL.append(one)
    outLL.append(gateL)
  #pprint(outLL)
  print(yaml.dump(outLL))


#=================================
#=================================
#  M A I N
#=================================
#=================================

'''
****** Step 1: Generate RB sequences *********
The RB sequences consist of random Clifford elements chosen uniformly from 
the Clifford group on  1-qubit, including a computed reversal element, 
that should return the qubits to the initial state.

'''

#Generate RB circuits (1Q RB)
rb_opts = {}
#Number of Cliffords in the sequence
#rb_opts['length_vector'] = [1, 10, 20, 50, 75, 100, 125, 150, 175, 200]
rb_opts['length_vector'] = [1, 2, 4]
#rb_opts['length_vector'] = [1, 10,  80,  150, 175]

rb_opts['nseeds'] = 1 #Number of random sequences
rb_opts['rb_pattern'] = [[0]] #Default pattern : list of qubits to test 
rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)

# 
print('Dump all %d circuit corresponding to the 1st RB sequence'%len(rb_circs[0]))
for circ in  rb_circs[0]:
    print(circ)
#print(rb_circs[0][0])

#.... Transpile circuits to U1,U2,U3 gates
basis_gates = ['u1','u2','u3']
basis2_gates = ['p','sx']

trans_circsLL = []
for rb_seed,rb_circL in enumerate(rb_circs):
    print('Compiling seed %d, circLen=%d'%(rb_seed,len(rb_circL)))
    circL = qiskit.compiler.transpile(rb_circL, basis_gates=basis_gates)
    trans_circsLL.append(circL)

#..... inspect compled circs
sd=0  # use 1st seed
for ic,circ in enumerate(trans_circsLL[sd]):
    print('---------- \n -------')
    circ_org=rb_circs[sd][ic]
    print('circ=',ic); print(circ_org); print(circ)
    circ2 = qiskit.compiler.transpile(circ, basis_gates=basis2_gates)
    print('4 QubiC'); print(circ2)
    #.... convert each circuit to DAG and export as dict
    qdag = circuit_to_dag(circ)
    print('\nList DAG  properties:')
    #pprint(qdag.properties())
    print(qdag.properties()['operations'])
    #1layers_2_dict(qdag)
