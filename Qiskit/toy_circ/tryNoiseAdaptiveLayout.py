#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Aziz:
The optimisation_level parameter uses several passes to transform and map the circuit to a real backend. So for your first question, when it's equal to 3, the NoiseAdaptiveLayout is used to analyse or select which set of qubits has the least cost in terms of gate errors and readout errors.

you need to use the Unroller pass before applying the NoiseAdaptativeLayout. 

'''

# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
import numpy as np
from qiskit_ibm_provider import IBMProvider

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-b','--backName',default='ibm_hanoi',
                        help="backend for transpiler" )

    args = parser.parse_args()

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()

print('M:IBMProvider()...')
provider = IBMProvider()

backend = provider.get_backend(args.backName)
print('\nmy backend=',backend)
print(backend.status())
print('nqubit=',backend.configuration().n_qubits)
hw_config=backend.configuration().to_dict()
#print('\nconfiguration :',hw_config.keys())
print('backend=',backend)
for x in [ 'max_experiments',  'max_shots', 'n_qubits' ]:
    print( x,hw_config[x])

from qiskit import QuantumCircuit

#load your grover_circuit first using qasm
qasmF='../example-class/noiseStudy/qcdata/qft_4Q.qasm'
qasmF='../example-class/noiseStudy/qcdata/grover_3Qas11.qasm'
circuit=QuantumCircuit().from_qasm_file(qasmF)
print(circuit)
qc=circuit

from qiskit.transpiler.passes import*
from qiskit.transpiler.passmanager import PassManager

#decompose the circuit onto basis gates
unroll=Unroller(backend.configuration().basis_gates)

#Use the NoiseAdaptiveLayout to find the best qubits with least noise
noiseadapt=NoiseAdaptiveLayout(backend.properties())

#Add the passes
pm = PassManager()
pm.append(unroll)
pm.append(noiseadapt)

trans_circ=pm.run(qc)

#print the best layout or selected qubits
print(pm.property_set['layout'])

