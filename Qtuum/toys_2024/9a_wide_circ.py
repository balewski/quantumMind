#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
explore how QTUUM transpiles circuit with 6 concurent CX

I'd like to see :
a) how many cycles were required for my circuit, is it 2 or more?
c) the assignment of qubit pairs to 5 interaction points on H1-1 for each ZZMax gate.


'''
from pprint import pprint
from access_qtuum import activate_qtuum_api
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

from pytket import Circuit
from pytket.passes import RebaseTket
from pytket.circuit import OpType
from qiskit.converters import circuit_to_dag

#...!...!..................
def create_wide_circ(nqa, N=1):  # TKET circ
    M=N+1
    print('define %d-q wide circ'%(M*nqa))
    qc = Circuit(M*nqa, M*nqa)
    for i in range(nqa):
        for j in range(N):
            qc.CX(M*i,M*i+1+j)
    return qc


#...!...!....................
def circ_depth_qiskit(qc,text='myCirc'):   # from Aziz
    len1=qc.depth(filter_function=lambda x: x.operation.name == 'cx')
    len2=qc.depth(filter_function=lambda x: x.operation.num_qubits == 2 )
    len3=qc.depth(filter_function=lambda x: x.operation.num_qubits > 2 )
    print('%s qiskit depth: cx-cycle=%d   2c_cycle=%d 3c+_cycle=%d '%(text,len1,len2,len3))


#...!...!....................
def circ_depth_tket(qc,text='myCirc'):   # tket
    len1=qcT.depth_by_type({OpType.ZZPhase, OpType.CX})
    len2=qc.depth()

    print('%s tket depth: 2c_cycle=%d any_cycle=%d '%(text,len1,len2))


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    machine = 'H1-1E'
    api=activate_qtuum_api()
    backend = QuantinuumBackend(device_name=machine, api_handler=api)
    print('machine=',machine)
    print("status:", backend.device_state(device_name=machine, api_handler=api))
    
    nq=2
    qc=create_wide_circ(nq,3) # TKET circ
    print(tk_to_qiskit(qc))

    
    print('commands for same circ transpiled for',machine)
    qcT=backend.get_compiled_circuit(qc)

    print('M: transpiled')
    qcT2=tk_to_qiskit(qcT);      print(qcT2)
    circ_depth_qiskit(qcT2,text='circ_trans')
    circ_depth_tket(qcT,text='circ_trans')

    RebaseTket().apply(qc) # to revert from QTUUM naitve gates
    qdag = circuit_to_dag(tk_to_qiskit(qc))
    print('\nList orginal DAG  properties:')
    pprint(qdag.properties())

    qdag = circuit_to_dag(tk_to_qiskit(qcT))
    print('\nList transpiled DAG  properties:')
    pprint(qdag.properties())

    if 0:
        print('\ndump transp circ:');
        for x in qcT.get_commands() :  print(x)
    
    
