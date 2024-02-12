#!/usr/bin/env python3
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
from pprint import pprint
import qiskit
print('qiskit ver=',qiskit.__qiskit_version__)

circPath='../example-class/noiseStudy/qcdata/'
circF=circPath+'ghz_3qm.qasm'
circ=QuantumCircuit.from_qasm_file( circF )
print(circ)
dag = circuit_to_dag(circ)
dagProp=dag.properties()
pprint(dagProp)
dagF='my_dag.png'
dag.draw(filename=dagF)
print('\n feh or  display ',dagF)

'''
circuit depth :  the largest amount of instructions being executed on the same wire (qubit)
 - QuantumCircuit: not include compiler or simulator directives, such as 'barrier'
 - DAG : include barrier

 DAG factors: how many sub-circuits (in this case subgraphs) can be made from this DAG?
'''
