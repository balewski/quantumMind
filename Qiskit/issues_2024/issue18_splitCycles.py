#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
from qiskit import QuantumCircuit,Aer,transpile
from pprint import pprint
from qiskit.converters import circuit_to_dag

#...!...!....................
def circ_depth_v1(qc,text='myCirc'):   # from Aziz 
    len1=qc.depth(filter_function=lambda x: x.operation.name == 'cx')
    len2=qc.depth(filter_function=lambda x: x.operation.num_qubits == 2 )
    len3=qc.depth(filter_function=lambda x: x.operation.num_qubits ==3 )
    len4=qc.depth(filter_function=lambda x: x.operation.num_qubits > 3 )
    print('%s depth: cx-cycle=%d  2c_cycle=%d 3c_cycle=%d 4+_cycle=%d '%(text,len1,len2,len3,len4))

    
#...!...!....................
def circuit_layers_info_v3(circ,verb=2):
    if verb>0:print('Analyze cycles of circ  with %d qubits and  depth %d'%(circ.num_qubits,circ.depth()))
    qdag = circuit_to_dag(circ)

    dagLay=qdag.layers()
    ncxl=0 # num cycles with CNOT
    ncx=0 # total CNOT

    meas_qids=[] # I want only measured qubits in order of clbits
    serLays=[] # serialized layers in text format
    for k,lay in enumerate(dagLay):
        gateL=[]
        cTag=' '
        for op in lay['graph'].op_nodes():
            adrL=['%d:%s'%(k,op.name)] +[str(qr._index) for qr in  op.qargs]
            adr='.'.join(adrL)
            gateL.append(adr)
            if 'cx'==op.name: ncx+=1

            if 'measure'==op.name: meas_qids.append(op.qargs[0]._index)
        txt=', '.join(gateL)
        if 'cx' in txt:
            ncxl+=1; cTag='*'
        if verb>1: print('%2d layer end: '%k,cTag,txt)
    if verb>0: print('  %d cycles w/ CNOTs, total %d CNOT'%(ncxl,ncx))

qasmF='pucr4a4d2s.qasm'
qc=QuantumCircuit().from_qasm_file(qasmF)

print(qc)
circuit_layers_info_v3(qc)


backend=Aer.get_backend("qasm_simulator")
optLev=3
print(' transpile L=%d for the ideal  backend ...'%optLev)
basis_gates = ['p','u','cx']
qcT = transpile(qc, backend=backend,basis_gates =basis_gates,  optimization_level=optLev )
print(qcT)
circuit_layers_info_v3(qcT)

# you can get the CX depth of a circuit
circ_depth_v1(qc,text='circ_orig')
circ_depth_v1(qcT,text='circ_trans')
