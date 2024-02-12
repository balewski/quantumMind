#!/usr/bin/env python3
'''
converts qasm circuit to quil 
based on DAG
only limitted set of gates is encoded
'''

import qiskit as qk
from qiskit.converters import circuit_to_dag, dag_to_circuit
from pprint import pprint
from pyquil import Program, get_qc
from pyquil.quilbase import Declare
from pyquil.gates import CNOT, Z, MEASURE
from functools import reduce


#...!...!....................
def layers_2_dict(qdag):
    print('\n Order gates chronologically  (per layer) as list of lists')
    outLL=[]
    dagLay=qdag.layers()
    for k,lay in enumerate(dagLay):      
        gateL=[]
        for op in lay['graph'].op_nodes():
            one={'name':op.name}
            one['qubit']= [qr._index  for qr in  op.qargs]
            one['param']=op.op._params
            gateL.append(one)
        outLL.append(gateL)
    pprint(outLL)

#...!...!....................
def dag_2_quil(qdag):
    print('\n Order gates  (per layer) ')
    outL=['DDD']
    mxClBit=-1
    dagLay=qdag.layers()
    for k,lay in enumerate(dagLay):
        for op in lay['graph'].op_nodes():
            qL=[str(qr._index)  for qr in  op.qargs]  # qubits as str
            cL=[cr._index  for cr in  op.cargs]  # clbits  as int
            pL=['%6f'% x for x in op.op._params]
            nq=len(qL)
            npa=len(pL)
            print('xx',k,op.name,qL,pL,nq,cL)
            if op.name=='barrier':
                continue # tmp
                txt='FENCE '+' '.join(qL)
            elif op.name=='measure':
                assert len(qL)==len(cL)==1
                txt='MEASURE %s ro[%d]'%(qL[0],cL[0])
                if mxClBit <cL[0] : mxClBit=cL[0]
            else:
                txt=op.name.upper()
                if txt=='CX': txt='CNOT'
                if npa>0 : txt+='('+','.join(pL)+') '
                txt+=' '+' '.join(qL)
            print(txt)
                       
            outL.append(txt)
    outL[0]=' DECLARE ro BIT[%d]'%(mxClBit+1)
    pprint(outL)
    outStr='\n'.join(outL)
    return outStr
  
#...!...!....................
def shotM_2_counts(shot_meas):
    results = list(map(lambda arr: reduce(lambda x, y: str(x) + str(y), arr[::-1], ""), shot_meas))
    counts = dict(zip(results,[results.count(i) for i in results]))
    return counts

#=================================
#=================================
#  M A I N
#=================================
#=================================

inpF="qbart_6q_1248.qasm"
circ=qk.QuantumCircuit.from_qasm_file(inpF)
print(circ)
qdag = circuit_to_dag(circ)
#layers_2_dict(qdag)

quilProgStr=dag_2_quil(qdag)
print(circ)
print('M: conversion finished')
print(quilProgStr.replace('\n','\\n'))
qprg=Program(quilProgStr)

# Raw Quil with newlines
qprg1=Program("DECLARE ro BIT[2]\nH 0\nY 1\nXY(pi) 11 12\nCNOT 1 4\nRY(0.34) 3\nMEASURE 1 ro[1]")


# undefined:  \nU3 (0.1,0.2,0.3) 7
# \nFENCE 0

qprg2 = Program(
    Declare("ro", "BIT", 2),
    Z(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
)

qprg.wrap_in_numshots_loop(1000)
print('\n Quil program:')
print('M:qprg\n',qprg)
device_name = '6q-qvm'
qback = get_qc(device_name, as_qvm=True)  #backend

shot_meas = qback.run(qprg).readout_data.get("ro")
print('M:shot_meas1',shot_meas)
counts= shotM_2_counts(shot_meas)
print('M:counts1',device_name,counts)

print('\nM: switch to noisy device')
device_name ="10q-noisy-qvm"
qback3 = get_qc(device_name)
shot_meas = qback3.run(qprg).readout_data.get("ro")
counts= shotM_2_counts(shot_meas)
counts_ord = sorted(counts.items(),  key=lambda item: item[1], reverse=True)
print('M:counts3',device_name, 'nsol=',len(counts_ord)); pprint(counts_ord[:10])



print('M:done')
